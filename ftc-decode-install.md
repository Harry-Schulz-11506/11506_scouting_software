# FTC DECODE Scouting System - Installation Guide
### By Team 11506

A complete neural network-powered scouting system for FTC competitions with live match analysis and AI-powered alliance recommendations.

---

## 📋 Prerequisites

Before installing, ensure you have:

- **Python 3.8 or higher** installed on your system
- **pip** (Python package installer)
- **FTC API Credentials** (Username and API Key from FIRST)
- **Internet connection** for downloading dependencies and accessing FTC API

---

## 🚀 Installation Steps

### Step 1: Download the Files

1. Save the provided `11506_scouting.py` file to a folder on your computer
2. Open a terminal/command prompt in that folder

### Step 2: Install Required Python Packages

Run the following command to install all dependencies:

```bash
pip install flask flask-cors numpy requests scikit-learn
```

**Package breakdown:**
- `flask` - Web server framework
- `flask-cors` - Cross-origin resource sharing support
- `numpy` - Numerical computing library
- `requests` - HTTP library for API calls
- `scikit-learn` - Machine learning library for neural network

### Step 3: Configure FTC API Credentials

You have two options to set your FTC API credentials:

#### Option A: Environment Variables (Recommended)

**Windows (Command Prompt):**
```cmd
set FTC_API_USERNAME=your_username
set FTC_API_KEY=your_api_key
```

**Windows (PowerShell):**
```powershell
$env:FTC_API_USERNAME="your_username"
$env:FTC_API_KEY="your_api_key"
```

**Mac/Linux:**
```bash
export FTC_API_USERNAME=your_username
export FTC_API_KEY=your_api_key
```

#### Option B: Edit the Python File

Open `11506_scouting.py` and modify lines 20-21:

```python
FTC_API_USERNAME = os.environ.get('FTC_API_USERNAME', 'YOUR_USERNAME_HERE')
FTC_API_KEY = os.environ.get('FTC_API_KEY', 'YOUR_API_KEY_HERE')
```

Replace `'YOUR_USERNAME_HERE'` and `'YOUR_API_KEY_HERE'` with your actual credentials.

### Step 4: Get Your FTC API Credentials

1. Go to [https://ftc-events.firstinspires.org/services/API](https://ftc-events.firstinspires.org/services/API)
2. Register for API access if you haven't already
3. Note your **Username** and **Authorization Key**

---

## ▶️ Running the Application

### Start the Server

In your terminal, run:

```bash
python 11506_scouting.py
```

You should see output like:

```
======================================================================
🤖 FTC DECODE Scouting System by Team 11506
======================================================================
📡 Server: http://127.0.0.1:5000
🔧 API Configured: True
🧠 Neural Network: Ready
======================================================================
```

### Access the Application

Open your web browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 📱 Using the Scouting System

### 1. Load Event Data

On the home page:
1. Enter the **Event Code** (e.g., `USILCA1`)
2. Enter your **Team Number** (e.g., `11506`)
3. Enter the **Season** (e.g., `2025`)
4. Click **🚀 Load Event**

The system will:
- Fetch all matches from the FTC API
- Process team statistics
- Train the neural network AI model
- Display live rankings and match data

### 2. Find Alliance Partners

Click **🤝 Alliance** in the top menu:
1. Enter your team number
2. Click **🔍 Find Best Partners**
3. View AI-ranked alliance recommendations with compatibility scores

### 3. Generate Match Strategy

Click **📊 Strategy** in the top menu:
1. Enter two team numbers
2. Click **🎯 Generate Strategy**
3. View AI predictions, expected scores, and win probabilities

---

## 🔧 Troubleshooting

### "Could not fetch event data"
- Verify your FTC API credentials are correct
- Check that the event code exists and matches are available
- Ensure you have internet connectivity

### "No event data loaded"
- You must load an event first before accessing alliance or strategy features
- Click **🏠 Home** and load event data

### Import errors
- Reinstall packages: `pip install --upgrade flask flask-cors numpy requests scikit-learn`
- Check Python version: `python --version` (should be 3.8+)

### Port already in use
- Close other applications using port 5000
- Or modify the port in the last line of the code: `app.run(host='127.0.0.1', port=5001, debug=True)`

### Neural Network not training
- Ensure the event has at least 5 completed matches
- Check that teams have scoring data available

---

## 🎯 Features

✅ **Live Match Feed** - Real-time updates of match results and scores  
✅ **AI Neural Network** - Machine learning predictions for alliance performance  
✅ **Team Rankings** - Automated rankings based on comprehensive statistics  
✅ **Alliance Selection** - AI-powered recommendations for optimal partners  
✅ **Match Strategy** - Detailed breakdowns with win probability calculations  
✅ **Pattern Analysis** - Consistency tracking and performance trends  
✅ **Auto-Refresh** - Automatic data updates every 30 seconds  

---

## 📊 System Requirements

- **RAM:** 2GB minimum, 4GB recommended
- **Disk Space:** 100MB for Python packages
- **Network:** Stable internet connection for API access
- **Browser:** Modern browser (Chrome, Firefox, Safari, Edge)

---

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all installation steps were completed
3. Ensure FTC API credentials are valid and active
4. Check Python and package versions are up to date

---

## 📄 License

Created by FTC Team 11506 for the FIRST Tech Challenge community.

**Good luck at your competitions! 🏆**