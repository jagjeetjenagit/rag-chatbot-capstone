# RAG Chatbot - Quick Setup Script for Windows PowerShell

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "RAG Chatbot - Quick Setup" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Error installing dependencies" -ForegroundColor Red
    exit 1
}

# Setup environment file
Write-Host ""
Write-Host "Setting up environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
} else {
    Copy-Item .env.example .env
    Write-Host "✓ Created .env file from template" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Please edit .env and add your API key!" -ForegroundColor Red
    Write-Host "   1. Open .env in a text editor" -ForegroundColor White
    Write-Host "   2. Add your OPENAI_API_KEY or GOOGLE_API_KEY" -ForegroundColor White
    Write-Host "   3. Save the file" -ForegroundColor White
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Make sure you've added your API key to .env" -ForegroundColor White
Write-Host "2. Run the application:" -ForegroundColor White
Write-Host "   python app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Open your browser to:" -ForegroundColor White
Write-Host "   http://127.0.0.1:7860" -ForegroundColor Cyan
Write-Host ""
Write-Host "For help, see README.md" -ForegroundColor White
Write-Host ""
