import logging
import os

from typing import Dict, List, Union
from fastapi import APIRouter
from src.onnx_model import OnnxTransformer
from tokenizers import Tokenizer
from pydantic import BaseModel
from rest_api.emotion import get_task_user_agent_emotion
from rest_api.schema import TaskInput

logger = logging.getLogger(__name__)
router = APIRouter()


class Request(BaseModel):
    texts: Union[str, List[str]]


# go emotion cfg
emotion_model_name = "MiniLMv2-goemotions-v2-onnx"
tokenizer = Tokenizer.from_file(os.path.join(emotion_model_name, "tokenizer.json"))
tokenizer.enable_padding(
    pad_token="<pad>",
    pad_id=1,
)
tokenizer.enable_truncation(max_length=256)
emotion_model = OnnxTransformer(
    emotion_model_name,
)


emotion_model_name = "MiniLMv2-goemotions-v2-onnx"
tokenizer = Tokenizer.from_pretrained("minuva/MiniLMv2-goemotions-v2-onnx")
tokenizer.enable_truncation(max_length=256)
emotion_model = OnnxTransformer(
    emotion_model_name,
)

tokenizer.enable_padding(
    pad_token="<pad>",
    pad_id=1,
)


@router.post("/conversation_emotions_plugin", response_model=Dict[str, str])
async def task_emotions(request: TaskInput):
    return get_task_user_agent_emotion(
        request.llm_input, request.llm_output, emotion_model, tokenizer
    )
