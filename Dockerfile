FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model required by presidio
RUN python -m spacy download en_core_web_lg

# Pre-download HuggingFace models so they're baked into the image
RUN python -c "\
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest'); \
AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest'); \
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/nli-deberta-v3-small') \
"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "pipeline.server:app", "--host", "0.0.0.0", "--port", "8000"]
