# Intro

A simple API server using FastAPI for serving a small and high quality emotion classification model with onnxruntime package for fast CPU inference.

# Install from source
```bash
git clone https://github.com/minuva/emotion-prediction-serverless.git
cd emotion-prediction-serverless
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
docker run -p 9612:9612 -it emotion
```

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