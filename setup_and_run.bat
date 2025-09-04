@echo off
echo ====================================
echo Cell Detection Tool - Easy Setup
echo ====================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found
echo.

:: Check if virtual environment exists
if not exist "venv" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

echo.
echo 🔧 Activating virtual environment and installing dependencies...

:: Activate virtual environment and install packages
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Upgrade pip first
python -m pip install --upgrade pip >nul 2>&1

:: Install requirements
echo Installing required packages... (this may take a few minutes)
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    echo Make sure you have an internet connection
    pause
    exit /b 1
)

echo.
echo ✅ All dependencies installed successfully!
echo.
echo 🚀 Starting Cell Detection Tool...
echo The app will open in your web browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

:: Run the Streamlit app
streamlit run streamlit_app.py --server.port=8501
pause
