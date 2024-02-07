# Intro

A simple API server using FastAPI for serving a small and high quality emotion classification model onnxruntime only for CPU inference to Google Cloud Run.

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

# Deploy to cloun Run

```bash
gcloud projects create emotion-cloudrun
gcloud config set project emotion-cloudrun
docker build --tag gcr.io/emotion-cloudrun/emotion .
docker push gcr.io/emotion-cloudrun/emotion
gcloud run deploy emotion-ml-app --platform managed --region europe-west3 --image gcr.io/emotion-cloudrun/emotion --service-account yourservice-account --allow-unauthenticated
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