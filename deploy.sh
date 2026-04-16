#!/bin/bash
set -euo pipefail

# RadgnarackAssist Backend Deployment Script
# Runs on VPS via GitHub Actions SSH

echo "=== RadgnarackAssist Deployment Started ==="
echo "Timestamp: $(date)"

# Configuration
PROJECT_DIR="/var/www/RadgnarackAssistBackend"
SERVICE_NAME="radgnarackassist"
APP_PORT=8001
VENV_DIR="$PROJECT_DIR/venv"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo ""
echo "[1/8] Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

echo ""
echo "[2/8] Pulling latest code from main..."
git pull origin main

echo ""
echo "[3/8] Creating virtual environment if missing..."
if [ ! -d "$VENV_DIR" ]; then
    echo "   Creating new virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "   Virtual environment exists"
fi

echo ""
echo "[4/8] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo ""
echo "[5/8] Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[6/8] Creating systemd service file..."
sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=RadgnarackAssist Backend API
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$VENV_DIR/bin"
EnvironmentFile=$PROJECT_DIR/.env

ExecStart=$VENV_DIR/bin/uvicorn app.main:app --host 0.0.0.0 --port $APP_PORT

Restart=always
RestartSec=5
StartLimitInterval=60s
StartLimitBurst=3

StandardOutput=journal
StandardError=journal
SyslogIdentifier=radgnarackassist

[Install]
WantedBy=multi-user.target
EOF

echo "   Service file created at: $SERVICE_FILE"

echo ""
echo "[7/8] Reloading systemd and restarting service..."
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
sleep 2

echo ""
echo "[8/8] Verifying service status..."
sudo systemctl status "$SERVICE_NAME" --no-pager -l

echo ""
echo "=== Deployment Complete ==="
echo "Service: $SERVICE_NAME"
echo "Port: $APP_PORT"
echo "URL: http://$(hostname -I | awk '{print $1}'):$APP_PORT"
