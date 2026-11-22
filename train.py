"""Train on the official training split and select neural checkpoints on validation."""

import argparse
import copy
import json
import os
import random
import shutil
import tempfile
import zlib
from pathlib import Path

import joblib
import numpy as np
import torch
from filelock import FileLock
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizerFast

from data import CACHE, MODELS, TWEET_URL, checkpoint, split, verified_file
from moderation import (
    LEXICON,
    MAX_TOKENS,
    MeanEmbeddingMLP,
    lexicon_hit,
    mean_vector,
    normalize,
    quantized_copy,
    words,
)

SEED = 42


def macro_f1(y, predicted):
    return float(f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0))


def stable_hash(word):
    return zlib.crc32(word.encode("utf-8"))


def class_weights(y):
    return torch.tensor(len(y) / (2 * np.bincount(y)), dtype=torch.float32)


def train_classical(base, train, y):
    tokens = [words(t) for t in train]
    hits = np.array([lexicon_hit(t, set(LEXICON)) for t in tokens])
    rates = [(int(y[hits == h].sum()) + 1) / (int((hits == h).sum()) + 2) for h in [False, True]]
    (base / "rules" / "rules.json").write_text(
        json.dumps({"lexicon": LEXICON, "offensive_rate": rates}, indent=2)
    )
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30000),
        LogisticRegression(C=2, class_weight="balanced", max_iter=400, random_state=SEED),
    )
    model.fit(train, y)
    joblib.dump(model, base / "tfidf" / "pipeline.joblib", compress=0)
    return tokens


def train_mlp(path, tokens, y, val, val_y):
    w2v = Word2Vec(
        sentences=tokens,
        vector_size=48,
        window=5,
        min_count=2,
        sg=1,
        negative=5,
        workers=1,
        seed=SEED,
        epochs=10,
        hashfxn=stable_hash,
    )
    vocab, vectors = w2v.wv.key_to_index, w2v.wv.vectors
    np.save(path / "vectors.npy", vectors, allow_pickle=False)
    (path / "words.txt").write_text("\n".join(w2v.wv.index_to_key) + "\n", encoding="utf-8")
    x = torch.from_numpy(np.stack([mean_vector(t, vocab, vectors) for t in tokens]))
    vx = torch.from_numpy(np.stack([mean_vector(words(t), vocab, vectors) for t in val]))
    y_tensor = torch.from_numpy(y)
    model = MeanEmbeddingMLP()
    model.center.copy_(x.mean(0))
    model.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-6))
    opt = torch.optim.SGD(model.parameters(), lr=0.018, momentum=0.9, weight_decay=0.0005)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y))
    history, best_score, best_epoch = [], -1, 0
    for epoch in range(1, 19):
        model.train()
        for ids in torch.randperm(len(y)).split(96):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[ids]), y_tensor[ids])
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            score = macro_f1(val_y, model(vx).argmax(1).numpy())
        history.append(score)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), path / "mlp.pt")
    return {"validation_macro_f1": history, "best_epoch": best_epoch, "vocabulary": len(vocab)}


def train_bert(base, train, y, val, val_y):
    source = checkpoint()
    tokenizer = BertTokenizerFast(vocab_file=str(source / "vocab.txt"), do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        source, local_files_only=True, num_labels=2, attn_implementation="eager"
    )

    def encode(texts):
        return tokenizer(
            texts, padding="max_length", truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        )

    def batch(enc, ids):
        # Trim padding in each mini-batch; the same attention mask also guards padding inside it.
        width = int(enc["attention_mask"][ids].sum(1).max())
        return {k: v[ids, :width] for k, v in enc.items()}

    ex, vx = encode(train), encode(val)
    y_tensor = torch.from_numpy(y)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y))
    opt = torch.optim.AdamW(model.parameters(), lr=0.00006, weight_decay=0.01)
    history, best_score, best_epoch, best = [], -1, 0, None
    for epoch in range(1, 4):
        model.train()
        for number, ids in enumerate(torch.randperm(len(y)).split(48), start=1):
            opt.zero_grad(set_to_none=True)
            logit = model(**batch(ex, ids)).logits
            loss_fn(logit, y_tensor[ids]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if number % 80 == 0:
                print(f"BERT epoch {epoch}, batch {number}", flush=True)
        model.eval()
        with torch.inference_mode():
            predicted = torch.cat(
                [
                    model(**batch(vx, idx)).logits.argmax(1)
                    for idx in torch.arange(len(val)).split(64)
                ]
            )
        score = macro_f1(val_y, predicted.numpy())
        history.append(score)
        print(f"BERT epoch {epoch}, validation macro-F1 {score:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval().save_pretrained(base / "bert", safe_serialization=True)
    shutil.copy2(base / "bert" / "config.json", base / "int8" / "config.json")
    for key in ("int8", "bert"):
        shutil.copy2(source / "vocab.txt", base / key / "vocab.txt")
    packed = quantized_copy(model)
    torch.save(packed.state_dict(), base / "int8" / "int8.pt")
    return {
        "validation_macro_f1": history,
        "best_epoch": best_epoch,
        "engine": torch.backends.quantized.engine,
    }


def train_models(dest=MODELS):
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(dest) + ".lock"):
        if (dest / "training.json").is_file():
            print(f"Using completed model directory {dest}", flush=True)
            return
        if dest.exists():
            raise ValueError(
                f"{dest} already exists without its training manifest. Use a new --out directory."
            )
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(int(os.environ.get("MATCHCHAT_TRAIN_THREADS", "4")))
        raw_train, train_y = split("train")
        raw_val, raw_val_y = split("val")
        # Reserve test inputs before fitting, without opening any test labels.
        test_file = verified_file(CACHE / "tweet", TWEET_URL, "test_text.txt")
        seen = {normalize(t) for t in test_file.read_text("utf-8").splitlines()}

        def take_unique(texts, labels):
            kept, targets = [], []
            for raw, label in zip(texts, labels):
                text = normalize(raw)
                if text not in seen:
                    seen.add(text)
                    kept.append(text)
                    targets.append(label)
            return kept, targets

        val, val_y = take_unique(raw_val, raw_val_y)
        train, train_y = take_unique(raw_train, train_y)
        y = np.array(train_y, dtype=np.int64)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            base = Path(tmp) / "ready"
            base.mkdir()
            for item in ("rules", "tfidf", "neural", "bert", "int8"):
                (base / item).mkdir()
            print("Fitting the two classical baselines", flush=True)
            tokens = train_classical(base, train, y)
            print("Training skip-gram embeddings and the MLP", flush=True)
            neural = train_mlp(base / "neural", tokens, y, val, val_y)
            print("Fine-tuning the pretrained BERT encoder and classifier", flush=True)
            bert = train_bert(base, train, y, val, val_y)
            report = {
                "seed": SEED,
                "original_train": len(raw_train),
                "original_val": len(raw_val),
                "train": len(y),
                "val": len(val_y),
                "train_classes": np.bincount(y).tolist(),
                "val_classes": np.bincount(val_y).tolist(),
                "mlp": neural,
                "bert": bert,
            }
            (base / "training.json").write_text(json.dumps(report, indent=2) + "\n")
            base.rename(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=MODELS, help="new model directory")
    train_models(parser.parse_args().out)
