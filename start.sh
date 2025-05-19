#!/bin/bash
# Startup script for Smart Market Analyzer

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Check if running with Docker Compose
if [ "$1" == "docker" ]; then
    echo "Starting with Docker Compose..."
    docker-compose -f $PROJECT_DIR/deployment/docker-compose.yml up
    exit $?
fi

# Set PYTHONPATH to include the project root
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# Ensure data directories exist
mkdir -p $PROJECT_DIR/data/market
mkdir -p $PROJECT_DIR/data/models
mkdir -p $PROJECT_DIR/data/backtest
mkdir -p $PROJECT_DIR/logs

# Start either API server or worker based on argument
if [ "$1" == "api" ] || [ "$1" == "" ]; then
    echo "Starting API server..."
    cd $PROJECT_DIR
    python -c "import sys; print('Python path:', sys.path)"
    python -c "import os; print('Current directory:', os.getcwd())"
    python -c "import os; print('Directory contents:', os.listdir('.'))"
    python -c "import os; print('src directory exists:', os.path.exists('src'))"
    if [ -d "src/api" ]; then
        python -c "import os; print('src/api contents:', os.listdir('src/api'))"
    fi
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
elif [ "$1" == "worker" ]; then
    echo "Starting worker service..."
    cd $PROJECT_DIR
    python worker.py
elif [ "$1" == "both" ]; then
    echo "Starting both API server and worker service..."
    cd $PROJECT_DIR
    # Start API server in background
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    
    # Start worker
    python worker.py &
    WORKER_PID=$!
    
    # Trap SIGINT and SIGTERM to kill both processes
    trap "kill $API_PID $WORKER_PID; exit" SIGINT SIGTERM
    
    # Wait for any process to finish
    wait
else
    echo "Invalid argument: $1"
    echo "Usage: $0 [api|worker|both|docker]"
    exit 1
fi
