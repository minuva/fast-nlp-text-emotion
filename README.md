# Intro

A simple API server using FastAPI for serving a small and high quality emotion classification model onnxruntime only for CPU inference.

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

# Models

| Model | 
| --- |
| [Ngit/MiniLM-L6-toxic-all-labels](https://huggingface.co/Ngit/MiniLMv2-L6-H384-goemotions-v2)
| [Ngit/MiniLM-L6-toxic-all-labels-onnx](https://huggingface.co/Ngit/MiniLMv2-L6-H384-goemotions-v2-onnx)