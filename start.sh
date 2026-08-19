#!/bin/bash
# start.sh — Starts all services on Oracle Cloud VM.
# Next.js (port 3000) + pipeline-service (port 3001) + Caddy (port 80)

set -e
cd "$(dirname "$0")"

export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"
source "$HOME/.venv/bin/activate"

echo "🚀 Starting Manhwa Recap Studio..."
echo ""

# Kill any existing processes
pkill -f "next-server" 2>/dev/null || true
pkill -f "pipeline-service" 2>/dev/null || true
pkill -f "index.ts" 2>/dev/null || true
sleep 2

# Start pipeline-service (port 3001)
echo "▶ Starting pipeline-service (port 3001)..."
cd mini-services/pipeline-service
nohup bun run start > /home/ubuntu/manhwa-recap-studio-v3/pipeline.log 2>&1 &
PIPELINE_PID=$!
cd ../..

# Start Next.js (port 3000)
echo "▶ Starting Next.js (port 3000)..."
nohup bun .next/standalone/server.js > /home/ubuntu/manhwa-recap-studio-v3/nextjs.log 2>&1 &
NEXT_PID=$!

# Wait for services to start
sleep 5

# Check if they're running
if curl -s http://localhost:3001/internal/health | grep -q "ok"; then
    echo "✅ Pipeline-service is running (PID: $PIPELINE_PID)"
else
    echo "⚠️  Pipeline-service may not be ready yet (check pipeline.log)"
fi

if curl -s -o /dev/null -w "" http://localhost:3000/ 2>/dev/null; then
    echo "✅ Next.js is running (PID: $NEXT_PID)"
else
    echo "⚠️  Next.js may not be ready yet (check nextjs.log)"
fi

# Get public IP
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo "YOUR_VM_IP")
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🎉  Manhwa Recap Studio is running!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  🌐 Website:  http://$PUBLIC_IP"
echo "  📊 API:      http://$PUBLIC_IP/api/stats"
echo "  🔧 Pipeline: http://$PUBLIC_IP:3001/internal/health"
echo ""
echo "  To stop:     pkill -f 'next-server|index.ts'"
echo "  To restart:  bash start.sh"
echo "  Logs:        tail -f nextjs.log pipeline.log"
echo ""
echo "═══════════════════════════════════════════════════════════════"
