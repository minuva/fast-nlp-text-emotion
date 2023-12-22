import logging
import os
import numpy as np

from typing import Dict, List, Union
from fastapi import APIRouter
from src.onnx_model import OnnxTransformer
from tokenizers import Tokenizer
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class Request(BaseModel):
    texts: Union[str, List[str]]


# go emotion cfg
emotion_model_name = "MiniLMv2-L6-H384-goemotions-v2-onnx"
tokenizer = Tokenizer.from_file(os.path.join(emotion_model_name, "tokenizer.json"))
tokenizer.enable_padding()
tokenizer.enable_truncation(max_length=256)
emotion_model = OnnxTransformer(
    emotion_model_name,
)


@router.post("/emotions", response_model=List[Dict])
def conversation_emotions(request: Request):
    output = emotion_model.predict(tokenizer, request.texts, batch_size=16)
    output = np.concatenate(output, axis=0)
    scores = 1 / (1 + np.exp(-output))  # Sigmoid
    results = []
    for item in scores:
        labels = []
        scores = []
        for idx, s in enumerate(item):
            labels.append(emotion_model.config["id2label"][str(idx)])
            scores.append(float(s))
        results.append({"labels": labels, "scores": scores})

    return results
