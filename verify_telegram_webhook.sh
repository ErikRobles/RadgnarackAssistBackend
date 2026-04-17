#!/bin/bash
# Telegram Webhook Verification Script
# Run this on the VPS as root

echo "=== Telegram Webhook Verification ==="
echo ""

# 1. Check if token is in environment
echo "[1/6] Checking TELEGRAM_BOT_TOKEN..."
TOKEN=$(grep TELEGRAM_BOT_TOKEN /etc/radgnarackassist.env 2>/dev/null | cut -d'=' -f2)
if [ -z "$TOKEN" ]; then
    echo "❌ FAIL: TELEGRAM_BOT_TOKEN not found in /etc/radgnarackassist.env"
    exit 1
fi
echo "✓ Token found: ${TOKEN:0:20}..."
echo ""

# 2. Check webhook info
echo "[2/6] Checking getWebhookInfo..."
WEBHOOK_INFO=$(curl -s "https://api.telegram.org/bot${TOKEN}/getWebhookInfo")
echo "Response:"
echo "$WEBHOOK_INFO" | python3 -m json.tool 2>/dev/null || echo "$WEBHOOK_INFO"
echo ""

# 3. Check if webhook URL is set
echo "[3/6] Checking webhook URL..."
WEBHOOK_URL=$(echo "$WEBHOOK_INFO" | grep -o '"url": "[^"]*"' | cut -d'"' -f4)
EXPECTED_URL="https://api.radgnarackassist.rrspark.website/api/telegram/webhook"

if [ -z "$WEBHOOK_URL" ]; then
    echo "❌ FAIL: Webhook URL is EMPTY"
    echo ""
    echo "[4/6] Setting webhook..."
    SET_RESULT=$(curl -s -X POST "https://api.telegram.org/bot${TOKEN}/setWebhook" \
        -H "Content-Type: application/json" \
        -d "{\"url\": \"${EXPECTED_URL}\", \"allowed_updates\": [\"message\"]}")
    echo "setWebhook response:"
    echo "$SET_RESULT" | python3 -m json.tool 2>/dev/null || echo "$SET_RESULT"
elif [ "$WEBHOOK_URL" != "$EXPECTED_URL" ]; then
    echo "❌ FAIL: Webhook URL is WRONG"
    echo "  Current:  $WEBHOOK_URL"
    echo "  Expected: $EXPECTED_URL"
    echo ""
    echo "[4/6] Fixing webhook URL..."
    SET_RESULT=$(curl -s -X POST "https://api.telegram.org/bot${TOKEN}/setWebhook" \
        -H "Content-Type: application/json" \
        -d "{\"url\": \"${EXPECTED_URL}\", \"allowed_updates\": [\"message\"]}")
    echo "setWebhook response:"
    echo "$SET_RESULT" | python3 -m json.tool 2>/dev/null || echo "$SET_RESULT"
else
    echo "✓ Webhook URL is correct: $WEBHOOK_URL"
fi
echo ""

# 5. Verify again
echo "[5/6] Verifying webhook is now set..."
WEBHOOK_INFO=$(curl -s "https://api.telegram.org/bot${TOKEN}/getWebhookInfo")
echo "Updated getWebhookInfo:"
echo "$WEBHOOK_INFO" | python3 -m json.tool 2>/dev/null || echo "$WEBHOOK_INFO"
echo ""

# 6. Check backend can receive
echo "[6/6] Testing backend webhook route..."
echo "Run this in another terminal:"
echo "  sudo journalctl -u radgnarackassist -f"
echo ""
echo "Then send a test message to your bot in Telegram."
echo "You should see [WEBHOOK] logs appear."
echo ""

echo "=== Verification Complete ==="
