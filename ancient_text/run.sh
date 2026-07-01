#!/bin/bash
set -e
echo "Starting PaddleOCR inference..."
python /app/inference_paddle.py
echo "Inference completed."