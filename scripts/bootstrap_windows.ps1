$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

foreach ($Command in @("python", "docker", "ollama")) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Command"
    }
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

$Settings = @{}
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^([^#][^=]*)=(.*)$') {
        $Settings[$matches[1].Trim()] = $matches[2].Trim()
    }
}

$GenerationModel = if ($Settings["GENERATION_MODEL"]) { $Settings["GENERATION_MODEL"] } else { "qwen2.5:7b-instruct" }
$ReasoningModel = if ($Settings["REASONING_MODEL"]) { $Settings["REASONING_MODEL"] } else { $GenerationModel }
$VisionModel = if ($Settings["VISION_MODEL"]) { $Settings["VISION_MODEL"] } else { "qwen2.5vl:3b" }
$EnableVision = if ($Settings["ENABLE_VISION"]) { $Settings["ENABLE_VISION"] } else { "true" }

ollama pull $GenerationModel
if ($ReasoningModel -ne $GenerationModel) {
    ollama pull $ReasoningModel
}
if ($EnableVision.ToLower() -eq "true") {
    ollama pull $VisionModel
}

docker compose up -d neo4j

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "Bootstrap complete. Start the API with:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "uvicorn enterprise_graphrag.main:app --host 0.0.0.0 --port 8000"
