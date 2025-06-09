#!/bin/bash
# quick-start.sh - One-Click Setup and Test for Consultant Matchmaker v2.0

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

# Check if .env exists and has required variables
check_env_file() {
    if [[ ! -f ".env" ]]; then
        print_error ".env file not found!"
        print_info "Please run: cp .env.example .env"
        print_info "Then edit .env with your Supabase credentials"
        exit 1
    fi
    
    # Check for required variables
    if ! grep -q "SUPABASE_URL=https://" .env; then
        print_error "SUPABASE_URL not configured in .env"
        print_info "Please update .env with your Supabase project URL"
        exit 1
    fi
    
    if ! grep -q "SUPABASE_SERVICE_KEY=" .env && ! grep -q "SUPABASE_SERVICE_KEY=your-service-key" .env; then
        print_error "SUPABASE_SERVICE_KEY not configured in .env"
        print_info "Please update .env with your Supabase service key"
        exit 1
    fi
    
    print_success "Environment file validated"
}

# Setup Python environment
setup_python() {
    print_header "Setting up Python Environment"
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        print_success "Created virtual environment"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt > /dev/null 2>&1
    print_success "Python dependencies installed"
}

# Setup Ollama and Qwen model
setup_qwen() {
    print_header "Setting up Qwen 2.5 Model"
    
    # Check if Ollama is installed
    if ! command -v ollama &> /dev/null; then
        print_info "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        print_success "Ollama installed"
    else
        print_success "Ollama already installed"
    fi
    
    # Start Ollama if not running
    if ! pgrep -x "ollama" > /dev/null; then
        print_info "Starting Ollama..."
        ollama serve > /dev/null 2>&1 &
        sleep 3
    fi
    
    # Check if model exists
    if ! ollama list | grep -q "qwen2.5:7b"; then
        print_info "Downloading Qwen 2.5 model (this may take a few minutes)..."
        ollama pull qwen2.5:7b
        print_success "Qwen 2.5 model ready"
    else
        print_success "Qwen 2.5 model already available"
    fi
}

# Test connections
test_connections() {
    print_header "Testing Connections"
    
    # Test Ollama
    if curl -s http://localhost:11434/api/version > /dev/null; then
        print_success "Ollama API responding"
    else
        print_error "Ollama API not responding"
        exit 1
    fi
    
    # Test Qwen model
    print_info "Testing Qwen model..."
    source venv/bin/activate
    python3 -c "
import asyncio
import sys
sys.path.append('.')
from app.services.ai_client import ai_client

async def test():
    try:
        result = await ai_client.test_connection()
        if result['success']:
            print('✅ Qwen 2.5 model responding')
        else:
            print(f'❌ Qwen test failed: {result.get(\"error\", \"Unknown error\")}')
            sys.exit(1)
    except Exception as e:
        print(f'❌ Qwen test error: {e}')
        sys.exit(1)

asyncio.run(test())
"
}

# Start services
start_services() {
    print_header "Starting Services"
    
    source venv/bin/activate
    
    # Start API server
    print_info "Starting API server..."
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
    API_PID=$!
    echo $API_PID > .api.pid
    
    # Wait for API to start
    sleep 5
    
    # Test API
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "API server started (PID: $API_PID)"
    else
        print_error "API server failed to start"
        exit 1
    fi
    
    # Start worker
    print_info "Starting background worker..."
    python -m worker.cv_worker > logs/worker.log 2>&1 &
    WORKER_PID=$!
    echo $WORKER_PID > .worker.pid
    
    sleep 2
    print_success "Background worker started (PID: $WORKER_PID)"
}

# Run comprehensive tests
run_tests() {
    print_header "Running System Tests"
    
    # Test 1: Health check
    print_info "Testing health endpoint..."
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
    fi
    
    # Test 2: AI connection
    print_info "Testing AI connection..."
    if curl -s http://localhost:8000/test-ai | grep -q "success"; then
        print_success "AI connection test passed"
    else
        print_warning "AI connection test had issues"
    fi
    
    # Test 3: Create test consultant (if test file exists)
    if [[ -f "test_cv.pdf" ]]; then
        print_info "Testing CV upload..."
        UPLOAD_RESULT=$(curl -s -X POST http://localhost:8000/consultants/upload \
            -F "file=@test_cv.pdf" \
            -F "prior_engagement=false")
        
        if echo "$UPLOAD_RESULT" | grep -q "success"; then
            print_success "CV upload test passed"
            CONSULTANT_ID=$(echo "$UPLOAD_RESULT" | grep -o '"consultant_id":"[^"]*"' | cut -d'"' -f4)
            echo "Test consultant ID: $CONSULTANT_ID" > .test_consultant_id
        else
            print_warning "CV upload test skipped (test_cv.pdf not found)"
        fi
    fi
    
    # Test 4: Project matching
    print_info "Testing project matching..."
    MATCH_RESULT=$(curl -s -X POST http://localhost:8000/projects/match \
        -H "Content-Type: application/json" \
        -d '{"description":"We need a Python developer with 3+ years experience","title":"Python Developer","max_matches":5}')
    
    if echo "$MATCH_RESULT" | grep -q "success"; then
        print_success "Project matching test passed"
    else
        print_warning "Project matching test had issues"
    fi
}

# Show service URLs and next steps
show_info() {
    print_header "🎉 Setup Complete!"
    
    echo -e "${GREEN}Your Consultant Matchmaker v2.0 is ready!${NC}\n"
    
    echo "📍 Service URLs:"
    echo "   API Server:    http://localhost:8000"
    echo "   Health Check:  http://localhost:8000/health"
    echo "   API Docs:      http://localhost:8000/docs"
    echo "   Interactive:   http://localhost:8000/redoc"
    echo ""
    
    echo "🔧 Service Management:"
    echo "   Stop services:    ./quick-start.sh stop"
    echo "   View API logs:    tail -f logs/api.log"
    echo "   View worker logs: tail -f logs/worker.log"
    echo ""
    
    echo "🧪 Quick Tests:"
    echo "   curl http://localhost:8000/health"
    echo "   curl http://localhost:8000/test-ai"
    echo ""
    
    echo "📁 Upload a CV:"
    echo "   curl -X POST http://localhost:8000/consultants/upload \\"
    echo "        -F \"file=@your_resume.pdf\" \\"
    echo "        -F \"prior_engagement=false\""
    echo ""
    
    echo "🎯 Match a project:"
    echo "   curl -X POST http://localhost:8000/projects/match \\"
    echo "        -H \"Content-Type: application/json\" \\"
    echo "        -d '{\"description\":\"Need a Python developer\",\"max_matches\":5}'"
    echo ""
    
    if [[ -f ".test_consultant_id" ]]; then
        CONSULTANT_ID=$(cat .test_consultant_id)
        echo "📋 Test consultant created: $CONSULTANT_ID"
        echo "   Check status: curl http://localhost:8000/consultants/$CONSULTANT_ID/status"
        echo ""
    fi
    
    print_info "Check logs/api.log and logs/worker.log for detailed logs"
    print_info "Visit http://localhost:8000/docs for interactive API documentation"
}

# Stop services
stop_services() {
    print_header "Stopping Services"
    
    if [[ -f ".api.pid" ]]; then
        API_PID=$(cat .api.pid)
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            print_success "Stopped API server (PID: $API_PID)"
        fi
        rm -f .api.pid
    fi
    
    if [[ -f ".worker.pid" ]]; then
        WORKER_PID=$(cat .worker.pid)
        if kill -0 $WORKER_PID 2>/dev/null; then
            kill $WORKER_PID
            print_success "Stopped worker (PID: $WORKER_PID)"
        fi
        rm -f .worker.pid
    fi
    
    # Also kill any Python processes running our app
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "worker.cv_worker" 2>/dev/null || true
    
    print_success "All services stopped"
}

# Main function
main() {
    case "${1:-start}" in
        "stop")
            stop_services
            ;;
        "start"|"")
            echo -e "${BLUE}"
            echo "██████╗ ██╗   ██╗███████╗██╗    ██╗    ██████╗    ██████╗ "
            echo "██╔═══██╗██║  ██║██╔════╝██║    ██║    ╚════██╗  ██╔═████╗"
            echo "██║   ██║██║  ██║█████╗  ██║ █╗ ██║     █████╔╝  ██║██╔██║"
            echo "██║▄▄ ██║██║  ██║██╔══╝  ██║███╗██║    ██╔═══╝   ████╔╝██║"
            echo "╚██████╔╝╚██████╔╝███████╗╚███╔███╔╝    ███████╗  ╚██████╔╝"
            echo " ╚══▀▀═╝  ╚═════╝ ╚══════╝ ╚══╝╚══╝     ╚══════╝   ╚═════╝ "
            echo -e "${NC}"
            echo "        Consultant Matchmaker v2.0 - Quick Start"
            echo ""
            
            # Create logs directory
            mkdir -p logs
            
            # Run setup steps
            check_env_file
            setup_python
            setup_qwen
            test_connections
            start_services
            sleep 3
            run_tests
            show_info
            ;;
        "test")
            source venv/bin/activate
            run_tests
            ;;
        *)
            echo "Usage: $0 [start|stop|test]"
            echo "  start (default): Setup and start all services"
            echo "  stop:            Stop all services"
            echo "  test:            Run system tests"
            ;;
    esac
}

# Run main function
main "$@"