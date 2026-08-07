#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

command -v python >/dev/null 2>&1 || { echo "Python was not found on PATH."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm was not found on PATH."; exit 1; }

test -f backend/models/model6_best_wer_checkpoint.pt || { echo "Missing backend/models/model6_best_wer_checkpoint.pt"; exit 1; }
test -f backend/models/bangla_vocab_tsv_fixed.json || { echo "Missing backend/models/bangla_vocab_tsv_fixed.json"; exit 1; }

echo "Backend:  http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:5173"

(cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload) &
BACKEND_PID=$!
(cd frontend && npm run dev) &
FRONTEND_PID=$!

trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true' EXIT
wait

