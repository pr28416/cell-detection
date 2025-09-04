# 🔬 Cell Detection Tool - Easy Setup Guide

This guide will help you run the Cell Detection Tool on your computer in just a few clicks!

## 📋 What You Need

1. **A computer** (Windows, Mac, or Linux)
2. **Internet connection** (for initial setup only)
3. **Your TIFF microscopy files**

## 🚀 Quick Start (3 Steps!)

### Step 1: Download the Code
1. Click the green "Code" button on this page
2. Select "Download ZIP"
3. Extract the ZIP file to your desktop or Documents folder

### Step 2: Run the Setup Script

#### On Windows:
- Double-click `setup_and_run.bat`
- If Windows asks about running the file, click "Yes" or "Run anyway"

#### On Mac:
- Open Terminal (press Cmd+Space, type "Terminal", press Enter)
- Type: `cd ` (with a space), then drag the downloaded folder into Terminal
- Press Enter, then type: `./setup_and_run.sh`
- Press Enter

#### On Linux:
- Open Terminal
- Navigate to the downloaded folder: `cd /path/to/cell-detection-tool`
- Run: `./setup_and_run.sh`

### Step 3: Use the App!
- Your web browser will automatically open to `http://localhost:8501`
- Upload your TIFF files and start detecting cells!
- **No file size limits** when running locally! 🎉

## 🔧 Troubleshooting

### "Python not found" Error
**Windows:**
1. Download Python from https://python.org/downloads/
2. **IMPORTANT:** Check "Add Python to PATH" during installation
3. Restart your computer and try again

**Mac:**
1. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Install Python: `brew install python3`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Still Having Issues?
1. Make sure you have an internet connection
2. Try running the setup script again
3. On Windows, try "Run as Administrator"

## 💡 Features

✅ **No file size limits** (upload 1GB+ files!)  
✅ **Fast processing** (runs on your computer)  
✅ **Interactive parameter tuning**  
✅ **Slice preview** for quick testing  
✅ **Export results** (images + CSV data)  
✅ **Works offline** after initial setup  

## 🛑 To Stop the App

Press `Ctrl+C` in the terminal/command window where the app is running.

---

**Need help?** The setup script will guide you through any issues and provide helpful error messages!
