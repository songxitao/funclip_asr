# run_cer_test.ps1
# One-click CER benchmark:
#   1. activate asr_ui_env (auto-locate conda if needed)
#   2. start local ASR service on :8002 if not already up
#   3. run cer_eval.py against AISHELL-1 test set (7176 utts)
#   4. print CER, leave service running in background
#
# Usage (new PowerShell window):
#   cd E:\project\funclip-pro
#   .\eval\run_cer_test.ps1
#
# Note: all paths are ASCII to avoid GBK/UTF-8 encoding issues on Windows.

$ErrorActionPreference = "Continue"

# 迁移后：脚本在 eval/，项目根为其父目录
$ROOT       = Split-Path $PSScriptRoot -Parent
$WAV_DIR    = "$ROOT\testset\aishell1_test_extracted\wav"
$TRANSCRIPT = "$ROOT\testset\aishell1_test_extracted\transcript.txt"
$BASE_URL   = "http://localhost:8002"
$SERVICE_LOG= "$ROOT\asr_service_run.log"

# ---- locate conda and activate env ----
$condaRoot = "D:\program files\Miniconda"
$hook = "$condaRoot\shell\condabin\conda-hook.ps1"
if (-not (Test-Path $hook)) {
    $alts = @("D:\Miniconda3","C:\Users\song\Miniconda3","C:\ProgramData\Miniconda3","C:\Program Files\Miniconda3")
    foreach ($a in $alts) {
        if (Test-Path "$a\shell\condabin\conda-hook.ps1") { $condaRoot = $a; $hook = "$a\shell\condabin\conda-hook.ps1"; break }
    }
}
if (Test-Path $hook) { & $hook }
try { conda activate asr_ui_env } catch { Write-Warning "conda activate asr_ui_env failed; make sure conda is initialized for PowerShell" }

# ---- ensure requests is available ----
try { python -c "import requests" } catch {
    Write-Warning "requests is NOT installed in asr_ui_env. Install it first:"
    Write-Warning "  pip install requests"
    exit 1
}

# ---- helper: is service up? ----
function Test-Port($url) {
    try { $r = Invoke-WebRequest -Uri $url -TimeoutSec 2 -ErrorAction SilentlyContinue; return ($r.StatusCode -eq 200) } catch { return $false }
}

# ---- start service if needed ----
if (-not (Test-Port $BASE_URL)) {
    Write-Host "[*] starting ASR service on 8002 ..."
    Start-Process -FilePath "python" -ArgumentList "$ROOT\asr_onnx_service.py" `
        -RedirectStandardOutput $SERVICE_LOG -RedirectStandardError $SERVICE_LOG -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 2
        if (Test-Port $BASE_URL) { $ok = $true; break }
        Write-Host "    waiting for service ... ($($i * 2)s)"
    }
    if (-not $ok) { Write-Error "service did not start. Check $SERVICE_LOG"; exit 1 }
    Write-Host "[*] service ready"
} else {
    Write-Host "[*] service already running on 8002, reusing it"
}

# ---- run CER eval ----
Write-Host "[*] running CER eval on AISHELL-1 test (7176 utts) ..."
python "$ROOT\eval\cer_eval.py" --wav_dir $WAV_DIR --transcript $TRANSCRIPT --base_url $BASE_URL

Write-Host ""
Write-Host "[done] ASR service is still running in background (log: $SERVICE_LOG)."
Write-Host "        To stop it: close its window, or run: Stop-Process -Name python
