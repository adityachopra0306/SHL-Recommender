#!/bin/bash
export PYTHONPATH=$(pwd)
uvicorn app.api:app --host 0.0.0.0 --port 10000
