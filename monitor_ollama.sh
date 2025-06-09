#!/bin/bash
# Monitor Ollama performance

echo "=== Ollama Status Monitor ==="
echo "Date: $(date)"
echo ""

# Check if Ollama is running
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama process: RUNNING"
    
    # Check API responsiveness
    if curl -s --max-time 5 http://localhost:11434/api/version > /dev/null; then
        echo "✅ API endpoint: RESPONSIVE"
        
        # Get version info
        VERSION=$(curl -s http://localhost:11434/api/version | grep -o '"version":"[^"]*"' | sed 's/"version":"//' | sed 's/"$//')
        echo "📋 Version: $VERSION"
        
        # Check loaded models
        echo ""
        echo "🤖 Loaded models:"
        curl -s http://localhost:11434/api/tags | jq -r '.models[] | "  - \(.name) (\(.size_vb / 1024 / 1024 / 1024 | round)GB)"' 2>/dev/null || echo "  Unable to fetch models"
        
        # Check system resources
        echo ""
        echo "💾 System resources:"
        echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)% used"
        echo "  Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
        echo "  Disk: $(df -h / | awk 'NR==2{printf "%s", $5}')"
        
    else
        echo "❌ API endpoint: NOT RESPONDING"
    fi
else
    echo "❌ Ollama process: NOT RUNNING"
fi

echo ""
echo "=== Recent Ollama Logs ==="
tail -10 /tmp/ollama.log 2>/dev/null || echo "No logs available"
