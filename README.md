# Match Chat

Match Chat is an English-language gaming-room moderation prototype for SWE015, use case 4. Type a message, send it, and see the label, model confidence, and processing time. Offensive messages appear as a mask. The side panel selects the model for the next message and records the individual response times. The default model is the quantized BERT checkpoint.

## Install and launch

Tested on Linux x86-64 with Python 3.12 and CPU-only PyTorch 2.7.1. Start in the extracted source directory. On Linux or macOS, create an environment and install the dependencies using these commands.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For Linux x86-64 or Windows x64, install the CPU-only PyTorch wheel first to avoid downloading GPU libraries. On Windows PowerShell use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1` for the first two steps. On macOS, skip the following PyTorch-index command and let `requirements.txt` install the platform wheel.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

```bash
python -m pip install -r requirements.txt
```

**Run the application with one command.**

```bash
python app.py
```

Open `http://127.0.0.1:8000` when the console reports that startup is complete. The first launch downloads the pinned pretrained checkpoint and public data, trains the classifiers, and creates `models/`. Allow several minutes on a CPU. A later launch reuses `models/` and needs no network. The local server binds to loopback. Set the `PORT` environment variable if port 8000 is in use. Training uses four threads by default and `MATCHCHAT_TRAIN_THREADS` can override that setting.

The downloadable coursework ZIP also includes `SWE015_Trained_Models.zip` next to the Git repository. To use the exact weights from the presentation without retraining, extract the model ZIP **into the repository directory** before running `python app.py`. Its top-level `models/` folder should sit beside `app.py`. The model files are excluded from Git. On the command line, from the repository directory, use `python -m zipfile -e ../SWE015_Trained_Models.zip .`.

All weight files stay outside Git. The original Google checkpoint is linked below. The training code reconstructs the fine-tuned and quantized weights and checks the SHA-256 of every download. The `training.json` manifest is written through a staged directory rename to avoid serving a partial model set.

## Training and evaluation

The application and the benchmark call the same `Predictor.classify` method. To evaluate the local models and save a fresh report while keeping the submitted results, run the following command.

```bash
python evaluate.py --out new-results
```

To train another set from scratch and evaluate that particular set separately, use new output directories.

```bash
python train.py --out rebuilt-models
python evaluate.py --models rebuilt-models --out rebuilt-results
```

The training script reserves normalized test texts, then keeps the first occurrence within validation and training, with priority test, validation, training. This removes training/validation duplicates and cross-split overlap. The official test file stays intact for evaluation. No test labels are read during training. Tokenizer truncation is limited to 64 WordPieces including special tokens. Messages sent to the app are limited to 500 characters.

| Retained split | Clean | Offensive | Total |
|---|---:|---:|---:|
| Training | 7,930 | 3,927 | 11,857 |
| Validation | 862 | 458 | 1,320 |
| Test | 620 | 240 | 860 |

### Model ladder

1. **Classical baseline.** TF–IDF word unigrams and bigrams feed class-balanced logistic regression. A separate rule predictor uses exact lexicon hits and Levenshtein distance at most one for words of five or more letters. Rules choose the label deterministically. Their displayed confidence is the smoothed fraction of matching or nonmatching training messages that had that predicted label.
2. **Neural network.** Gensim trains 48-dimensional skip-gram Word2Vec embeddings on training messages. The mean known-word vector is standardized using training statistics. A fully connected PyTorch network uses a hidden layer of 64 ReLU units, 20% dropout and SGD with momentum. Unknown-only messages receive a zero vector before standardization. The checkpoint with the best validation macro-F1 is saved.
3. **Transformer.** All parameters of Google’s English BERT Tiny (two layers, 128 hidden dimensions, two attention heads) and its new two-class head are fine-tuned with AdamW and class-weighted cross-entropy. A small English encoder matches the language of this dataset and the CPU target. The checkpoint with the best validation macro-F1 across three epochs is used.
4. **Quantized Transformer.** Post-training dynamic INT8 quantization packs linear layers. Embeddings use unsigned, row-wise 8-bit weights with floating-point lookup outputs. Remaining attention and normalization operations use floating point. `int8.pt` contains the actual packed state. It is separately loaded and used for every default chat request. All learned classifiers choose the higher-probability class at a 0.5 boundary. Confidence is a model score and was not calibrated on gaming chat.

### Measurement protocol

Macro-F1 averages the F1 scores of both classes. Size is the actual byte total of the predictor’s files divided by 1,000,000, including tokenizers, vocabularies or embeddings needed for that model. The latency test used one CPU thread, batch size 1, 20 warm-ups and the same 200 unique test row indices sampled without replacement for each predictor. A timer covers normalization, features, inference, label, confidence and masking. HTTP and browser rendering are outside that timer. AMD EPYC 9V74, Linux x86-64. The budget is 50 ms per message. A different machine, thread configuration, library build or system load can change timing and sometimes predictions.

| Predictor | Macro-F1 | Size (MB) | Mean (ms) | P95 (ms) |
|---|---:|---:|---:|---:|
| Edit-distance rules | 0.625 | 0.000390 | 0.093 | 0.185 |
| TF–IDF + logistic regression | 0.726 | 1.41 | 0.325 | 0.439 |
| Word2Vec + MLP | 0.631 | 1.72 | 0.124 | 0.235 |
| BERT Tiny FP32 | 0.737 | 17.78 | 1.625 | 2.178 |
| BERT Tiny INT8 | 0.733 | 4.90 | 1.879 | 2.246 |

Quantization cut the BERT bundle by 72.4%. It changed 6 labels across the 860 official test rows. All 200 timed quantized predictions met the CPU budget. The smaller artifact was slower on average in this run.

## How a message is handled

`web/app.js` sends the text and selected model to `POST /api/moderate`. `app.py` validates requests and dispatches them to `moderation.py`. The normalizer replaces handles and URLs, folds case, shortens repetitions, and translates common digit substitutions inside mixed alphanumeric tokens. The server returns a clean message or a fixed masking notice, plus the label, score, model and elapsed time. The browser adds an entry to the chat and the timing log. A flagged response omits the submitted text. The live UI does not persist chat messages.

## What the demonstration shows

| Typed input | Observed behavior |
|---|---|
| Great round everyone, thanks for helping me learn the map. | Quantized BERT displays the clean message. |
| You are a total 1d10t. | The normalizer converts `1d10t` to `idiot`. Quantized BERT masks the insult. |
| Nice save, that was a sick play! | Quantized BERT masks this compliment. The rule model allows the same message. |

The test corpus consists of English social-media posts. A model that flags friendly expressions from a gaming room can silence ordinary conversation. Other examples of input difficulty include quotation, sarcasm and missing context. A next experiment would collect consented gaming conversations, preserve conversation-level splits and measure errors on in-domain labels.

## Where the evidence lives

| File | Contents |
|---|---|
| `data.py` | Pinned sources, SHA-256 verification and official splits |
| `train.py` | Duplicate removal, local Word2Vec training, model fitting, validation checkpointing and quantization |
| `moderation.py` | Normalization, edit matching, model loading, prediction, masking and timing |
| `app.py`, `web/` | FastAPI server and chat interface |
| `evaluate.py` | Reproducible test evaluation and latency sampling |
| `reports/training.json` | Retained class counts, validation histories and selected epochs |
| `reports/results.json` | Raw sizes, macro-F1, per-class F1, confusion matrices, thread count and latency summaries |
| `reports/predictions.csv` | One row per official test example. Row numbers start at 1. Labels are 0 for clean and 1 for offensive. |
| `reports/latency.csv` | All 1,000 timed model/row combinations |

## Data, pretrained weights and tools

- Barbieri et al. (2020), *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*. [Paper](https://aclanthology.org/2020.findings-emnlp.148/) and [pinned offensive-task files](https://github.com/cardiffnlp/tweeteval/tree/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66/datasets/offensive).
- Zampieri et al. (2019), *SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval)*. [Original dataset task](https://aclanthology.org/S19-2010/).
- Google Research, BERT Miniatures. [Pinned English checkpoint and WordPiece vocabulary](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f).
- [PyTorch dynamic quantization](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.quantize_dynamic.html) and [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html).

OpenAI ChatGPT (Codex) assisted with code, debugging, running the experiments, the slides, script and recording. [Piper’s English Lessac voice](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) narrates the demo. The accompanying benchmark files were produced by executing `evaluate.py` against locally trained weights.
