#!/bin/bash

echo "===================================="
echo "Cell Detection Tool - Easy Setup"
echo "===================================="
echo ""

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}ERROR: Python is not installed${NC}"
        echo "Please install Python 3.8+ from:"
        echo "  - macOS: brew install python3 or download from python.org"
        echo "  - Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
        echo "  - Other Linux: Use your package manager"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo -e "${GREEN}✅ Python found${NC}"
echo ""

# Always run from the script directory so config files are discovered
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}🔧 Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to create virtual environment${NC}"
        echo "You may need to install python3-venv:"
        echo "  Ubuntu/Debian: sudo apt install python3-venv"
        exit 1
    fi
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

echo ""
echo -e "${BLUE}🔧 Activating virtual environment and installing dependencies...${NC}"

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to activate virtual environment${NC}"
    exit 1
fi

# Upgrade pip first
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
echo "Installing required packages... (this may take a few minutes)"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install requirements${NC}"
    echo "Make sure you have an internet connection"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ All dependencies installed successfully!${NC}"
echo ""
echo -e "${YELLOW}🚀 Starting Cell Detection Tool...${NC}"
echo "The app will open in your web browser at http://localhost:8501"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
echo ""

# Ensure generous upload limits and disable CORS/XSRF for local use
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=1024
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the Streamlit app with explicit flags (works even if config isn't picked up)
streamlit run "$SCRIPT_DIR/streamlit_app.py" \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.maxUploadSize=1024 \
    --server.maxMessageSize=1024 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
