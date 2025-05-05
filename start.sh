#!/bin/bash
export PYTHONPATH=$(pwd)
python -m nltk.downloader punkt stopwords
uvicorn app.api:app --host 0.0.0.0 --port 10000
