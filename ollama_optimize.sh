#!/bin/bash
# complete_fix.sh - Complete Ollama Optimization & Service Restart Script

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Print functions
print_header() {
    echo -e "\n${BLUE}🚀 $1${NC}"
    echo "=================================="
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
    # Kill background processes if script is interrupted
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

print_header "Complete Ollama & Services Optimization"
echo -e "${BLUE}This script will:${NC}"
echo "  1. Stop all Ollama instances and clear conflicts"
echo "  2. Start optimized Ollama with enhanced settings"
echo "  3. Ensure Qwen 2.5 model is loaded and optimized"
echo "  4. Test AI performance thoroughly"
echo "  5. Restart all application services"
echo "  6. Test complete integration"
echo "  7. Monitor real-time progress"
echo ""

# Step 1: Stop ALL Ollama instances
print_header "Step 1: Stopping All Ollama Instances"

echo "🛑 Killing all Ollama processes..."
pkill -f ollama 2>/dev/null || true
sleep 3

# Force kill if needed
if pgrep ollama > /dev/null; then
    print_warning "Force killing remaining Ollama processes..."
    pkill -9 -f ollama 2>/dev/null || true
    sleep 2
fi

# Check if port is free
if lsof -ti:11434 > /dev/null 2>&1; then
    print_warning "Port 11434 still in use, killing process..."
    kill -9 $(lsof -ti:11434) 2>/dev/null || true
    sleep 2
fi

print_success "All Ollama instances stopped"

# Step 2: Start Clean Optimized Ollama
print_header "Step 2: Starting Optimized Ollama"

# Clean up old logs
rm -f /tmp/ollama.log

# Set environment variables for optimization
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MAX_QUEUE=5

print_info "Environment variables set:"
print_info "  OLLAMA_HOST=0.0.0.0:11434"
print_info "  OLLAMA_KEEP_ALIVE=30m"
print_info "  OLLAMA_NUM_PARALLEL=2"

# Start Ollama with optimizations
echo "🚀 Starting optimized Ollama..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

print_info "Ollama started with PID: $OLLAMA_PID"
echo $OLLAMA_PID > /tmp/ollama.pid

# Wait for startup
print_info "Waiting for Ollama to initialize..."
for i in {1..15}; do
    if curl -s --connect-timeout 5 http://localhost:11434/api/version > /dev/null 2>&1; then
        print_success "Ollama started successfully"
        break
    fi
    
    if [ $i -eq 15 ]; then
        print_error "Ollama failed to start after 15 attempts"
        echo "📋 Recent logs:"
        tail -10 /tmp/ollama.log
        exit 1
    fi
    
    echo "⏳ Attempt $i/15..."
    sleep 2
done

# Step 3: Verify and Load Model
print_header "Step 3: Model Setup and Verification"

# Check Ollama version
print_info "Getting Ollama info..."
VERSION=$(curl -s http://localhost:11434/api/version 2>/dev/null | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
print_success "Ollama version: ${VERSION:-unknown}"

# Check available models
print_info "Checking available models..."
MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
if [ -n "$MODELS" ]; then
    echo "🤖 Available models:"
    echo "$MODELS" | sed 's/^/  - /'
else
    print_warning "No models found or API issue"
fi

# Ensure Qwen 2.5 is available
if ! ollama list 2>/dev/null | grep -q "qwen2.5:7b"; then
    print_warning "Qwen 2.5 model not found, pulling it now..."
    print_info "This may take several minutes..."
    ollama pull qwen2.5:7b
    print_success "Qwen 2.5 model downloaded"
else
    print_success "Qwen 2.5 model already available"
fi

# Preload the model for faster responses
print_info "Preloading Qwen 2.5 for optimal performance..."
curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "prompt": "System ready",
    "stream": false,
    "keep_alive": "30m"
  }' > /dev/null

print_success "Model preloaded and optimized!"

# Step 4: Performance Test
print_header "Step 4: AI Performance Testing"

print_info "Running comprehensive performance test..."
START_TIME=$(date +%s)

TEST_RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "prompt": "Respond with exactly: Performance test successful",
    "stream": false
  }' | grep -o '"response":"[^"]*"' | cut -d'"' -f4)

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if echo "$TEST_RESPONSE" | grep -q "Performance test successful"; then
    print_success "Performance test PASSED in ${DURATION}s"
    print_info "Response: ${TEST_RESPONSE}"
    
    if [ $DURATION -le 10 ]; then
        print_success "Excellent performance (≤10s)"
    elif [ $DURATION -le 30 ]; then
        print_warning "Good performance (≤30s)"
    else
        print_warning "Slow performance (>${DURATION}s) - consider hardware upgrade"
    fi
else
    print_error "Performance test FAILED"
    print_info "Response: ${TEST_RESPONSE}"
    echo "📋 Recent Ollama logs:"
    tail -5 /tmp/ollama.log
fi

# Step 5: Restart Application Services
print_header "Step 5: Restarting Application Services"

# Stop current services
print_info "Stopping existing application services..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "worker.cv_worker" 2>/dev/null || true
sleep 3

# Ensure logs directory exists
mkdir -p logs

# Start API server
print_info "Starting FastAPI server..."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
API_PID=$!
echo $API_PID > /tmp/api.pid
print_success "API server started with PID: $API_PID"

# Wait for API startup
print_info "Waiting for API server to initialize..."
for i in {1..10}; do
    if curl -s --connect-timeout 5 http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API server is responding"
        break
    fi
    
    if [ $i -eq 10 ]; then
        print_error "API server failed to start"
        echo "📋 Recent API logs:"
        tail -10 logs/api.log
        exit 1
    fi
    
    echo "⏳ Attempt $i/10..."
    sleep 3
done

# Start worker
print_info "Starting CV processing worker..."
python -m worker.cv_worker > logs/worker.log 2>&1 &
WORKER_PID=$!
echo $WORKER_PID > /tmp/worker.pid
print_success "Worker started with PID: $WORKER_PID"

# Wait for worker startup
sleep 5

print_success "All services restarted successfully!"

# Step 6: Integration Testing
print_header "Step 6: Complete Integration Testing"

# Test API health
print_info "Testing API health..."
API_HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null)
API_HEALTH=$(echo "$API_HEALTH_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

if [ "$API_HEALTH" = "healthy" ]; then
    print_success "API Status: healthy"
else
    print_warning "API Status: ${API_HEALTH:-unknown}"
    echo "📋 Health response: $API_HEALTH_RESPONSE"
fi

# Test AI connection through API
print_info "Testing AI integration through API..."
AI_TEST_RESPONSE=$(curl -s http://localhost:8000/test-ai 2>/dev/null)
AI_SUCCESS=$(echo "$AI_TEST_RESPONSE" | grep -o '"success":[^,}]*' | cut -d':' -f2 | tr -d ' ')

if [ "$AI_SUCCESS" = "true" ]; then
    print_success "AI Integration: working"
    AI_MODEL=$(echo "$AI_TEST_RESPONSE" | grep -o '"model":"[^"]*"' | cut -d'"' -f4)
    AI_RESPONSE_TIME=$(echo "$AI_TEST_RESPONSE" | grep -o '"response_time":[^,}]*' | cut -d':' -f2 | tr -d ' ')
    print_info "Model: ${AI_MODEL:-unknown}"
    print_info "Response time: ${AI_RESPONSE_TIME:-unknown}s"
else
    print_warning "AI Integration: ${AI_SUCCESS:-failed}"
    echo "📋 AI test response: $AI_TEST_RESPONSE"
fi

# Check consultant status
print_info "Checking existing consultant status..."
CONSULTANT_RESPONSE=$(curl -s http://localhost:8000/consultants/43b5a041-9231-494c-9577-601c6da6f8e7/status 2>/dev/null)
if [ -n "$CONSULTANT_RESPONSE" ]; then
    CONSULTANT_NAME=$(echo "$CONSULTANT_RESPONSE" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    PROCESSING_PHASE=$(echo "$CONSULTANT_RESPONSE" | grep -o '"processing_phase":"[^"]*"' | cut -d'"' -f4)
    PROCESSING_STATUS=$(echo "$CONSULTANT_RESPONSE" | grep -o '"processing_status":"[^"]*"' | cut -d'"' -f4)
    
    print_success "Consultant found: ${CONSULTANT_NAME:-unknown}"
    print_info "Processing phase: ${PROCESSING_PHASE:-unknown}"
    print_info "Processing status: ${PROCESSING_STATUS:-unknown}"
    
    if [ "$PROCESSING_PHASE" = "partially_processed" ] || [ "$PROCESSING_PHASE" = "analyzing" ]; then
        print_info "🚀 Consultant is ready for or undergoing Phase 2 (AI Analysis)"
    elif [ "$PROCESSING_PHASE" = "completed" ]; then
        print_success "🎉 Consultant is fully processed!"
    fi
else
    print_warning "Could not retrieve consultant status"
fi

# Step 7: Real-time Monitoring Setup
print_header "Step 7: System Monitoring & Status"

# Create monitoring function
monitor_system() {
    echo "📊 Current System Status:"
    echo "========================"
    
    # Ollama status
    if pgrep ollama > /dev/null; then
        echo "🤖 Ollama: RUNNING (PID: $(pgrep ollama))"
    else
        echo "🤖 Ollama: NOT RUNNING"
    fi
    
    # API status
    if pgrep -f "uvicorn app.main:app" > /dev/null; then
        echo "🌐 API Server: RUNNING (PID: $(pgrep -f 'uvicorn app.main:app'))"
    else
        echo "🌐 API Server: NOT RUNNING"
    fi
    
    # Worker status
    if pgrep -f "worker.cv_worker" > /dev/null; then
        echo "⚙️  Worker: RUNNING (PID: $(pgrep -f 'worker.cv_worker'))"
    else
        echo "⚙️  Worker: NOT RUNNING"
    fi
    
    # System resources
    echo ""
    echo "💾 System Resources:"
    echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)% used"
    echo "  Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
    echo "  Disk: $(df -h / | awk 'NR==2{printf "%s", $5}')"
    
    echo ""
    echo "📋 Recent Activity:"
    echo "=== Worker Logs (last 5 lines) ==="
    tail -5 logs/worker.log 2>/dev/null | sed 's/^/  /' || echo "  No worker logs yet"
    
    echo ""
    echo "=== Ollama Logs (last 3 lines) ==="
    tail -3 /tmp/ollama.log 2>/dev/null | sed 's/^/  /' || echo "  No Ollama logs yet"
}

# Show current status
monitor_system

# Final summary
print_header "🎉 Setup Complete!"

echo -e "${GREEN}Your Consultant Matchmaker v2.0 is now optimized and running!${NC}"
echo ""
echo "📍 Service URLs:"
echo "   API Server:    http://localhost:8000"
echo "   Health Check:  http://localhost:8000/health"
echo "   API Docs:      http://localhost:8000/docs"
echo ""
echo "🔧 Service Management:"
echo "   Stop all:      pkill -f 'ollama|uvicorn|worker'"
echo "   Monitor:       tail -f logs/worker.log"
echo "   Ollama logs:   tail -f /tmp/ollama.log"
echo ""
echo "📊 Process IDs saved:"
echo "   Ollama PID:    /tmp/ollama.pid"
echo "   API PID:       /tmp/api.pid"
echo "   Worker PID:    /tmp/worker.pid"
echo ""
echo "🧪 Quick Tests:"
echo "   curl http://localhost:8000/health"
echo "   curl http://localhost:8000/test-ai"
echo ""
echo "📈 Monitoring:"
echo "   ./monitor_ollama.sh"
echo "   tail -f logs/worker.log"
echo ""

# Offer to monitor in real-time
echo -e "${YELLOW}Would you like to monitor logs in real-time? (y/n)${NC}"
read -r MONITOR_CHOICE

if [[ $MONITOR_CHOICE =~ ^[Yy]$ ]]; then
    print_info "Starting real-time monitoring (Press Ctrl+C to stop)..."
    echo ""
    
    # Monitor all logs
    echo "=== Real-time Logs ==="
    (echo "🌐 API Logs:" && tail -f logs/api.log | sed 's/^/API: /') &
    (echo "⚙️  Worker Logs:" && tail -f logs/worker.log | sed 's/^/WORKER: /') &
    (echo "🤖 Ollama Logs:" && tail -f /tmp/ollama.log | sed 's/^/OLLAMA: /') &
    
    # Keep monitoring until user stops
    wait
else
    print_success "Setup complete! Your system is ready for CV processing."
    print_info "Monitor logs with: tail -f logs/worker.log"
fi