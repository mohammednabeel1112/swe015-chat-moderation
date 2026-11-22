"""Launch with python app.py, then open the printed local address."""

import os
from contextlib import asynccontextmanager
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data import MODELS, ROOT
from moderation import NAMES, Predictor


@asynccontextmanager
async def lifespan(app):
    if not (MODELS / "training.json").is_file():
        from train import train_models

        print("Preparing the models on this first launch. Progress appears here.", flush=True)
        train_models()
    torch.set_num_threads(1)
    app.state.models = {key: Predictor(key, MODELS) for key in NAMES}
    yield


app = FastAPI(title="Match Chat", lifespan=lifespan)
app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    model: Literal["int8", "bert", "neural", "tfidf", "rules"] = "int8"


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/models")
def model_list():
    return NAMES


@app.post("/api/moderate")
def moderate(message: Message):
    if not message.text.strip():
        raise HTTPException(422, "Type a message before sending.")
    result = app.state.models[message.model].classify(message.text.strip())
    return {**result, "model": message.model}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
