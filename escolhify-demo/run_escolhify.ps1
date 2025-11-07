# Escolhify - Easy Launch Script for PowerShell
Write-Host "Starting Escolhify ..." -ForegroundColor Green

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to project directory
Set-Location $ScriptDir

# Run chainlit using the local Python environment
& "$ScriptDir\.venv\Scripts\python.exe" -m chainlit run app.py

# Keep the window open
Read-Host "Press Enter to exit..."