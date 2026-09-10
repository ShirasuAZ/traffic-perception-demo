# 把 WSL 里的后端工作副本同步到仓库的 backend/（排除模型、数据、第三方仓库、日志）。
# 用法：powershell -File scripts\sync-backend.ps1        （WSL -> 仓库）
#       powershell -File scripts\sync-backend.ps1 -Reverse（仓库 -> WSL）
param([switch]$Reverse)
$wsl = "\\wsl.localhost\Ubuntu\home\shirasu\traffic"
$repo = Join-Path $PSScriptRoot "..\backend"
$src, $dst = if ($Reverse) { $repo, $wsl } else { $wsl, $repo }
robocopy $src $dst /E /XD third_party weights data logs __pycache__ YOLOX_outputs .pytest_cache node_modules `
  /XF *.pkl *.pth *.mp4 *.webm *.jpg *.png *.zip *.onnx /NFL /NDL /NJH /NJS /NP
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
Write-Host "synced $src -> $dst"
