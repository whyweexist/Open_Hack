#!/bin/bash
# ============================================================
# Container startup script
# ============================================================

set -e

echo "============================================"
echo " CCPA Compliance Analyzer"
echo " Starting FastAPI server on port 8000"
echo "============================================"

# Start uvicorn with the FastAPI app
exec python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 120 \
    --log-level info