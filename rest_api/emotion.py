import numpy as np

from typing import Dict
from tokenizers import Tokenizer
from src.onnx_model import OnnxTransformer

BATCH_SIZE = 16

CLASSES_TO_IGNORE = ["neutral"]


def postprocess(output, model, threshold=0.3):
    scores = 1 / (1 + np.exp(-output))  # Sigmoid
    results = []
    for item in scores:
        labels = []
        scores = []
        for idx, s in enumerate(item):
            if (
                s > threshold
                and model.config["id2label"][str(idx)] not in CLASSES_TO_IGNORE
            ):
                labels.append(model.config["id2label"][str(idx)])
                scores.append(s)
        results.append({"labels": labels, "scores": scores})
    return results


def predict(
    model: OnnxTransformer,
    tokenizer: Tokenizer,
    text: str,
    threshold=0.3,
):
    encoding = tokenizer.encode(text)

    inputs = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
    }
    output = model.predict(inputs)
    result = postprocess(output, model, threshold)

    return result


def get_task_user_agent_emotion(
    llm_input: str, llm_output: str, model, tokenizer
) -> Dict[str, str]:
    user_emotion = predict(model, tokenizer, llm_input)
    agent_emotion = predict(model, tokenizer, llm_output)

    user_label = user_emotion[0]["labels"][0] if user_emotion[0]["labels"] else ""
    agent_label = agent_emotion[0]["labels"][0] if agent_emotion[0]["labels"] else ""

    return {"user_emotion": user_label, "agent_emotion": agent_label}
