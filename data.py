"""Fetch the pinned public inputs and keep cache writes atomic."""

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
TWEET_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66/datasets/offensive/"
BERT_URL = "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/30b0a37ccaaa32f332884b96992754e246e48c5f/"
DIGESTS = {
    "config.json": "508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39",
    "model.safetensors": "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "train_text.txt": "78a7a32e38b10af7d8970b008bf17f661c8d0a90dad145fa0fa6a944669650db",
    "train_labels.txt": "c0b7d6ebdaa4ebcf6fc557ef1e775d92eda160218a0e3b1dd48eb8234dc892a6",
    "val_text.txt": "816f36d180c35f15a5104838cb73856a0bef42043482fe738f3481b06242a55c",
    "val_labels.txt": "ed2deb776bd1c52fb8221fadd3360e32d9dfe46842d78053528126e46363a258",
    "test_text.txt": "25b08c3333c26190f1023961c4508ec9aab24d4722b1a3ea7a6040724c120547",
    "test_labels.txt": "41d05a7aa0b01f5dafab21b95adb4f979cb4226c046ff315702774d10dac1605",
}


def verified_file(folder, url, name):
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    expected = DIGESTS[name]
    if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
        return dest
    with urlopen(url + name, timeout=120) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {name}")
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as f:
        f.write(content)
    os.replace(f.name, dest)
    return dest


def split(name):
    if name not in {"train", "val", "test"}:
        raise ValueError(name)
    paths = [
        verified_file(CACHE / "tweet", TWEET_URL, f"{name}_{kind}.txt")
        for kind in ("text", "labels")
    ]
    rows = paths[0].read_text("utf-8").splitlines()
    labels = list(map(int, paths[1].read_text("utf-8").splitlines()))
    if len(rows) != len(labels) or set(labels) != {0, 1}:
        raise ValueError(f"Invalid {name} split")
    return rows, labels


def checkpoint():
    directory = CACHE / "bert"
    for name in ("config.json", "model.safetensors", "vocab.txt"):
        verified_file(directory, BERT_URL, name)
    return directory
