# Resumable push script for the remaining ICU-XAI content.
# Run this when your network is stable. Each chunk retries up to 10 times.

Set-Location d:\icu-xai

# Network-friendly git settings (idempotent)
git config http.postBuffer 524288000
git config http.lowSpeedLimit 0
git config http.lowSpeedTime 999999
git config http.version HTTP/1.1

function Push-WithRetry {
    param([string]$Label, [int]$MaxTries = 10)
    Write-Host ""
    Write-Host "=== Pushing: $Label ===" -ForegroundColor Cyan
    for ($i = 1; $i -le $MaxTries; $i++) {
        Write-Host "Attempt $i of $MaxTries..."
        git push
        # Verify by comparing local HEAD to remote
        $local  = (git rev-parse HEAD).Trim()
        $remote = (git ls-remote origin main 2>$null | ForEach-Object { ($_ -split "\s+")[0] })
        if ($local -eq $remote) {
            Write-Host "OK: $Label landed on GitHub." -ForegroundColor Green
            return
        }
        Write-Host "Reset detected. Retrying in 5s..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
    Write-Host "FAILED after $MaxTries tries. Run the script again later." -ForegroundColor Red
    exit 1
}

function Commit-And-Push {
    param([string]$Pattern, [string]$Message, [string]$Label)
    git add $Pattern
    $staged = git diff --cached --name-only
    if (-not $staged) { Write-Host "Nothing to add for $Label, skipping."; return }
    git commit -m $Message | Out-Null
    Push-WithRetry -Label $Label
}

# 1. thesis.docx (already committed, just needs push)
$ahead = git rev-list --count '@{u}..HEAD' 2>$null
if ($ahead -gt 0) {
    Push-WithRetry -Label "Pending committed work (thesis.docx)"
}

# 2. thesis.pdf
Commit-And-Push -Pattern "thesis/thesis.pdf" `
    -Message "Compiled thesis (PDF)" `
    -Label "thesis.pdf"

# 3. notebooks
Commit-And-Push -Pattern "notebooks" `
    -Message "Exploratory notebooks" `
    -Label "notebooks"

# 4. outputs JSONs and CSV (small text results)
Commit-And-Push -Pattern "outputs/*.json","outputs/*.csv","outputs/*.npy" `
    -Message "Result JSONs, metrics CSV, SHAP matrices" `
    -Label "outputs (results)"

# 5. outputs/models (trained model weights)
Commit-And-Push -Pattern "outputs/models" `
    -Message "Trained model weights (XGBoost, TCN, Transformer)" `
    -Label "outputs/models"

# 6. outputs/figures
Commit-And-Push -Pattern "outputs/figures" `
    -Message "Publication figures (300 DPI thesis pack + diagnostics)" `
    -Label "outputs/figures"

# 7. data/raw split into two halves (53 MB total -> ~26 MB each)
$rawFiles = Get-ChildItem d:\icu-xai\data\raw -Recurse -File | Sort-Object FullName
$half = [int]($rawFiles.Count / 2)
Write-Host ""
Write-Host "data/raw has $($rawFiles.Count) files, splitting into 2 commits..."
$rawFiles[0..($half-1)]            | ForEach-Object { git add $_.FullName }
git commit -m "PhysioNet 2012 Set-A raw data (part 1)" | Out-Null
Push-WithRetry -Label "data/raw part 1"

$rawFiles[$half..($rawFiles.Count-1)] | ForEach-Object { git add $_.FullName }
git commit -m "PhysioNet 2012 Set-A raw data (part 2)" | Out-Null
Push-WithRetry -Label "data/raw part 2"

# 8. data/processed - push each large file as its own commit
$procFiles = Get-ChildItem d:\icu-xai\data\processed -File | Sort-Object Length
foreach ($f in $procFiles) {
    git add $f.FullName
    git commit -m "Processed data: $($f.Name)" | Out-Null
    $sizeMB = [math]::Round($f.Length / 1MB, 1)
    Push-WithRetry -Label "$($f.Name) ($sizeMB MB)"
}

Write-Host ""
Write-Host "=== ALL DONE ===" -ForegroundColor Green
Write-Host "Repo: https://github.com/ihsaanNaqvi/Explainable-ML-for-ICU-Mortality-Prediction"
