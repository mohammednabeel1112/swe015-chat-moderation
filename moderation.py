"""The inference path used by the app, the test evaluation and the latency benchmark."""

import html
import json
import re
import unicodedata
from pathlib import Path
from time import perf_counter_ns

import joblib
import numpy as np
import torch
from torch import nn
from torch.ao.quantization import (
    default_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    quantize_dynamic,
)
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

NAMES = {
    "int8": "BERT Tiny · INT8",
    "bert": "BERT Tiny · FP32",
    "neural": "Word2Vec + MLP",
    "tfidf": "TF–IDF + logistic regression",
    "rules": "Edit-distance rules",
}
LABELS = ("Clean", "Offensive")
MAX_TOKENS = 64
LATENCY_BUDGET_MS = 50
LEXICON = "asshole bastard bitch crap damn dickhead douche dumbass fuck fucking idiot idiots jerk moron pathetic prick shit stupid trashbag useless worthless".split()
LEET = str.maketrans("0134578", "oieastb")
TURKISH = str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ç": "c", "ö": "o", "ü": "u"})


def normalize(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    text = text.replace("i\u0307", "i").translate(TURKISH).replace("’", "'")
    text = re.sub(r"https?://\S+|www\.\S+", " link ", text)
    text = re.sub(r"(?<!\w)@\w+", " user ", text)
    text = re.sub(
        r"\b\w+\b", lambda m: m[0].translate(LEET) if any(c.isalpha() for c in m[0]) else m[0], text
    )
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    return " ".join(text.split())


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text)


def one_edit(a, b):
    """True when the Levenshtein distance is at most one."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(c != d for c, d in zip(a, b)) <= 1
    shorter, longer = sorted((a, b), key=len)
    i = 0
    while i < len(shorter) and shorter[i] == longer[i]:
        i += 1
    return shorter[i:] == longer[i + 1 :]


def lexicon_hit(tokens, lexicon):
    for word in tokens:
        if word in lexicon:
            return True
        if len(word) >= 5 and any(len(term) >= 5 and one_edit(word, term) for term in lexicon):
            return True
    return False


class MeanEmbeddingMLP(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.layers = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2)
        )

    def forward(self, features):
        return self.layers((features - self.center) / self.scale)


def mean_vector(tokens, vocab, vectors):
    ids = [vocab[t] for t in tokens if t in vocab]
    if not ids:
        return np.zeros(vectors.shape[1], dtype=np.float32)
    return vectors[ids].mean(axis=0)


def quantized_copy(model):
    engines = torch.backends.quantized.supported_engines
    for engine in ("x86", "fbgemm", "qnnpack"):
        if engine in engines:
            torch.backends.quantized.engine = engine
            break
    else:
        raise RuntimeError("This build of PyTorch needs a CPU quantization backend.")
    return quantize_dynamic(
        model.cpu().eval(),
        {
            nn.Linear: default_dynamic_qconfig,
            nn.Embedding: float_qparams_weight_only_qconfig,
        },
        inplace=False,
    )


class Predictor:
    def __init__(self, model_id, root):
        self.model_id = model_id
        path = Path(root) / model_id
        if model_id == "rules":
            settings = json.loads((path / "rules.json").read_text())
            self.lexicon = set(settings["lexicon"])
            self.rates = settings["offensive_rate"]
        elif model_id == "tfidf":
            self.model = joblib.load(path / "pipeline.joblib")
        elif model_id == "neural":
            self.vectors = np.load(path / "vectors.npy", allow_pickle=False)
            self.vocab = {w: i for i, w in enumerate((path / "words.txt").read_text().splitlines())}
            self.model = MeanEmbeddingMLP(self.vectors.shape[1])
            self.model.load_state_dict(
                torch.load(path / "mlp.pt", map_location="cpu", weights_only=True)
            )
            self.model.eval()
        elif model_id in {"bert", "int8"}:
            self.tokenizer = BertTokenizerFast(
                vocab_file=str(path / "vocab.txt"), do_lower_case=True
            )
            if model_id == "bert":
                self.model = BertForSequenceClassification.from_pretrained(
                    path, local_files_only=True, attn_implementation="eager"
                )
            else:
                config = BertConfig.from_json_file(str(path / "config.json"))
                config._attn_implementation = "eager"
                self.model = quantized_copy(BertForSequenceClassification(config))
                state = torch.load(path / "int8.pt", map_location="cpu", weights_only=True)
                self.model.load_state_dict(state)
            self.model.eval()
        else:
            raise ValueError(f"Unknown model {model_id}")

    @torch.inference_mode()
    def classify(self, message):
        start = perf_counter_ns()
        text = normalize(message)
        if self.model_id == "rules":
            label = int(lexicon_hit(words(text), self.lexicon))
            p = self.rates[label]
        elif self.model_id == "tfidf":
            p = float(self.model.predict_proba([text])[0, 1])
            label = int(p >= 0.5)
        else:
            if self.model_id == "neural":
                vec = mean_vector(words(text), self.vocab, self.vectors)
                logits = self.model(torch.from_numpy(vec).unsqueeze(0))
            else:
                inputs = self.tokenizer(
                    text, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
                )
                logits = self.model(**inputs).logits
            p = float(logits.softmax(-1)[0, 1])
            label = int(p >= 0.5)
        result = {
            "label": label,
            "confidence": p if label else 1 - p,
            "display": "Message hidden by moderation" if label else message,
        }
        result["ms"] = (perf_counter_ns() - start) / 1_000_000
        return result
