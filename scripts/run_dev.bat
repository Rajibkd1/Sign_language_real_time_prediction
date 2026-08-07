@echo off
setlocal

cd /d "%~dp0.."

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH.
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo npm was not found on PATH.
  exit /b 1
)

if not exist backend\models\model6_best_wer_checkpoint.pt (
  echo Missing backend\models\model6_best_wer_checkpoint.pt
  exit /b 1
)

if not exist backend\models\bangla_vocab_tsv_fixed.json (
  echo Missing backend\models\bangla_vocab_tsv_fixed.json
  exit /b 1
)

echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo.

start "BdSL Vision API" cmd /k "cd /d %cd%\backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
start "BdSL Vision UI" cmd /k "cd /d %cd%\frontend && npm run dev"

endlocal

