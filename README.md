# Fast emotion classification in text

This open-source project delivers an efficient emotion classification system built on FastAPI 🚀. It uses a compact, yet highly accurate model running on onnxruntime for rapid CPU-based processing. It is an ideal solution for applications requiring fast and reliable emotion classification without the need for GPU hardware. More details about the model in the [model page](https://huggingface.co/minuva/MiniLMv2-goemotions-v2).


# Install from source
```bash
git clone https://github.com/minuva/fast-nlp-text-emotion.git
cd fast-nlp-text-emotion
pip install -r requirements.txt
```


# Run locally

Run the following command to start the server (from the root directory):

```bash
chmod +x ./run.sh
./run.sh
```

Check `config.py` for more configuration options.


# Run with Docker

Run the following command to start the server (the root directory):

```bash
docker build --tag emotion .
docker run --network=postlang --network-alias=emotion -p 9612:9612 -it emotion
```

The network and the network alias are used to allow PostHog-LLM to communicate with the emotion classification service.
Since PostHog-LLM is running in a docker container, we connect the two services by adding them to the same network for *fast* and *reliable* communication.


# Example call
```bash
curl -X 'POST' \
  'http://127.0.0.1:9612/emotions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": ["hello", "im not happy"]
}'
```

# Models

| Model | Description |
| --- | --- |
| [minuva/MiniLMv2-goemotions-v2](https://huggingface.co/minuva/MiniLMv2-goemotions-v2) | A small and high quality emotion classification model |
| [minuva/MiniLMv2-goemotions-v2-onnx](https://huggingface.co/minuva/MiniLMv2-goemotions-v2-onnx) | quantized ONNX model |