#!/bin/bash
set -e

# RadgnarackAssist VPS Setup Script
# Run this on the VPS before first deployment

echo "=== RadgnarackAssist VPS Setup ==="
echo "VPS: root@72.62.200.116"
echo "Port: 8001"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: Please run as root"
    exit 1
fi

# Create deployment directory
echo "[1/6] Creating deployment directory..."
mkdir -p /var/www/RadgnarackAssistBackend
chown -R root:root /var/www/RadgnarackAssistBackend
cd /var/www/RadgnarackAssistBackend
echo "   ✓ Directory created: /var/www/RadgnarackAssistBackend"

# Check Python version
echo ""
echo "[2/6] Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "   Found Python $PYTHON_VERSION"
else
    echo "   Python 3 not found. Installing..."
    apt update
    apt install -y python3.11 python3.11-venv python3-pip
fi

# Check port availability
echo ""
echo "[3/6] Checking port 8001 availability..."
if ss -tlnp | grep -q ":8001"; then
    echo "   ⚠ WARNING: Port 8001 is already in use!"
    ss -tlnp | grep ":8001"
    echo ""
    echo "   Please update APP_PORT in GitHub secrets and deployment files"
    echo "   Then re-run this script."
    exit 1
else
    echo "   ✓ Port 8001 is available"
fi

# Create systemd service file
echo ""
echo "[4/6] Creating systemd service..."

cat > /etc/systemd/system/radgnarackassist.service << 'EOF'
[Unit]
Description=RadgnarackAssist Backend API
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/var/www/RadgnarackAssistBackend
Environment="PATH=/var/www/RadgnarackAssistBackend/venv/bin"
EnvironmentFile=/var/www/RadgnarackAssistBackend/.env

ExecStart=/var/www/RadgnarackAssistBackend/venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8001

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

systemctl daemon-reload
systemctl enable radgnarackassist
echo "   ✓ Service created and enabled"

# Create placeholder .env file
echo ""
echo "[5/6] Creating placeholder .env file..."
cat > /var/www/RadgnarackAssistBackend/.env << 'EOF'
# RadgnarackAssist Production Environment
# UPDATE THESE VALUES before first deployment!

OPENAI_API_KEY=REPLACE_ME
PINECONE_API_KEY=REPLACE_ME
PINECONE_INDEX_NAME=radgnarack-assist
PINECONE_NAMESPACE=default
TELEGRAM_BOT_TOKEN=REPLACE_ME
TELEGRAM_OWNER_CHAT_ID=6612661432

APP_PORT=8001
FRONTEND_URL=https://radgnarack.rrspark.website
API_DOMAIN=https://api.radgnarackassist.rrspark.website
EOF

chmod 600 /var/www/RadgnarackAssistBackend/.env
echo "   ✓ .env file created (UPDATE VALUES BEFORE DEPLOYMENT!)"

# Summary
echo ""
echo "[6/6] Setup Complete!"
echo ""
echo "=== Next Steps ==="
echo ""
echo "1. UPDATE .env file with real values:"
echo "   nano /var/www/RadgnarackAssistBackend/.env"
echo ""
echo "2. Add GitHub Secrets (10 required):"
echo "   - VPS_HOST: 72.62.200.116"
echo "   - VPS_USER: root"
echo "   - VPS_SSH_KEY: (your private SSH key)"
echo " - DEPLOY_PATH: /var/www/RadgnarackAssistBackend"
echo "   - APP_PORT: 8001"
echo "   - Plus API keys (OPENAI, PINECONE, TELEGRAM)"
echo ""
echo "3. Configure CyberPanel reverse proxy for:"
echo "   Domain: api.radgnarackassist.rrspark.website"
echo "   → Proxy to: 127.0.0.1:8001"
echo ""
echo "4. Push code to GitHub to trigger deployment"
echo ""
echo "=== Useful Commands ==="
echo "Check status:  systemctl status radgnarackassist"
echo "View logs:     journalctl -u radgnarackassist -f"
echo "Start service: systemctl start radgnarackassist"
echo "Stop service:  systemctl stop radgnarackassist"
echo ""
