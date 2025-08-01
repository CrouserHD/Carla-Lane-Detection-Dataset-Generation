# 🚗 Lane Comparison Tool - Schnellstart

## 🎯 Einfachster Start

### Option 1: Automatischer Launcher (Empfohlen)
```bash
python run_lane_comparison.py
```
- Erkennt automatisch, ob Dashboard-Dependencies verfügbar sind
- Bietet Installation an, falls nötig
- Startet mit oder ohne Dashboard

### Option 2: Direkter Start mit Dashboard
```bash
python main.py --dashboard
```
- Startet direkt mit Web-Dashboard
- Öffnet Browser automatisch
- Benötigt: `flask`, `flask-socketio`, `eventlet`, `requests`

### Option 3: Ohne Dashboard (wie früher)
```bash
python main.py
```
- Läuft ohne Web-Dashboard
- Nur Console-Output
- Keine zusätzlichen Dependencies nötig

## 📋 Alle Startoptionen

### 🌐 Mit Dashboard
```bash
# Standard Dashboard
python main.py --dashboard

# Dashboard auf anderem Port
python main.py --dashboard --port 8080

# Dashboard ohne Browser-Auto-Open
python main.py --dashboard --no-browser

# Dashboard mit Processing-Parametern
python main.py --dashboard --start_index 100 --num_images 50
```

### ⚡ Ohne Dashboard
```bash
# Standard
python main.py

# Mit Processing-Parametern
python main.py --start_index 100 --num_images 50
```

### 🔧 Installations-Befehle
```bash
# Alle Dependencies
pip install -r requirements.txt

# Nur Dashboard-Dependencies
pip install flask flask-socketio eventlet requests

# Nur Core-Dependencies
pip install numpy opencv-python matplotlib pillow tabulate colorama
```

## 🌟 Dashboard-Features

Wenn Sie das Dashboard verwenden, können Sie live sehen:
- **📊 Gesamtfortschritt** - Wie viele Bilder verarbeitet wurden
- **🎯 Algorithmus-Fortschritt** - Individual-Progress pro Algorithmus
- **🖼️ Live-Vorschau** - Aktuelle Comparison-Bilder
- **📈 Performance-Metriken** - F1-Score, Precision, Recall, FPS
- **📝 Live-Logs** - Alle Verarbeitungsschritte
- **⏱️ Geschätzte Restzeit** - Automatische Berechnung

## 🚀 Empfohlener Workflow

1. **Erste Verwendung:**
   ```bash
   python run_lane_comparison.py
   ```
   (Installiert automatisch Dependencies falls nötig)

2. **Regelmäßige Verwendung:**
   ```bash
   python main.py --dashboard
   ```
   (Direkter Start mit Dashboard)

3. **Ohne Dashboard:**
   ```bash
   python main.py
   ```
   (Wenn Sie nur Console-Output wollen)

## 🎉 Viel Spaß beim Live-Monitoring!

Das Dashboard macht Lane Detection zu einem visuellen Erlebnis! 🚗✨
