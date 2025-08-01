# 🚗 Lane Comparison Tool - Web Dashboard

Das Lane Comparison Tool unterstützt jetzt ein **Live-Web-Dashboard** zum Verfolgen des Fortschritts in Echtzeit!

## 📋 Installation

### 1. Dashboard-Abhängigkeiten installieren
```bash
pip install flask flask-socketio eventlet requests
```

Oder alle Abhängigkeiten auf einmal:
```bash
pip install -r requirements.txt
```

### 2. Dashboard verwenden

#### Option A: Mit Dashboard (Empfohlen)
```bash
python main.py --dashboard
```

#### Option B: Mit Dashboard auf anderem Port
```bash
python main.py --dashboard --port 8080
```

#### Option C: Ohne automatisches Browser-Öffnen
```bash
python main.py --dashboard --no-browser
```

#### Option D: Ohne Dashboard (wie vorher)
```bash
python main.py
```

## 🌐 Dashboard-Features

### ✨ Was Sie sehen können:
- **📊 Echtzeit-Fortschritt**: Gesamtfortschritt und pro Algorithmus
- **🖼️ Live-Vorschau**: Aktuelle Verarbeitungsbilder in Echtzeit
- **📈 Performance-Metriken**: F1-Score, Precision, Recall, FPS
- **📝 Live-Logs**: Alle Verarbeitungsschritte in Echtzeit
- **⏱️ Geschätzte Restzeit**: Automatische Berechnung der verbleibenden Zeit
- **🎯 Algorithmspezifischer Fortschritt**: Detaillierte Ansicht pro Algorithmus

### 🎨 Dashboard-Bereiche:
1. **Status-Leiste**: Aktueller Status und Phase
2. **Gesamtfortschritt**: Überblick über alle verarbeiteten Bilder
3. **Algorithmus-Fortschritt**: Individual-Fortschritt pro Algorithmus
4. **Live-Vorschau**: Aktuelle Verarbeitungsbilder
5. **Performance-Metriken**: Detaillierte Statistiken
6. **Live-Logs**: Alle Systemmeldungen

## 🚀 Verwendung

1. **Dashboard starten**:
   ```bash
   python main.py --dashboard
   ```

2. **Browser öffnet sich automatisch** auf: `http://localhost:5000`

3. **Verarbeitung beginnt automatisch** nach 3 Sekunden

4. **Live-Verfolgung**: Sehen Sie in Echtzeit:
   - Welches Bild gerade verarbeitet wird
   - Wie schnell die Algorithmen arbeiten
   - Vorschau der Ergebnisse
   - Detaillierte Logs

## 🔧 Technische Details

### Dashboard-Architektur:
- **Frontend**: HTML5 + JavaScript mit WebSockets
- **Backend**: Flask + SocketIO für Echtzeit-Updates
- **Kommunikation**: RESTful API + WebSocket-Verbindungen

### Ports:
- **Standard**: 5000
- **Anpassbar**: `--port <nummer>`

### Browser-Kompatibilität:
- Chrome/Edge: ✅ Vollständig unterstützt
- Firefox: ✅ Vollständig unterstützt
- Safari: ✅ Vollständig unterstützt

## 🎯 Beispiel-Workflow

```bash
# 1. Dependencies installieren
pip install flask flask-socketio eventlet requests

# 2. Dashboard starten
python main.py --dashboard

# 3. Browser öffnet sich automatisch
# 4. Verarbeitung beginnt automatisch
# 5. Live-Verfolgung des Fortschritts
```

## 📱 Responsive Design

Das Dashboard ist vollständig responsiv und funktioniert auf:
- 🖥️ Desktop-Computern
- 📱 Tablets
- 📲 Smartphones

## 🔍 Dashboard-URLs

- **Hauptseite**: `http://localhost:5000/`
- **API-Endpunkte**: 
  - `http://localhost:5000/api/progress` - Aktueller Fortschritt
  - `http://localhost:5000/api/update` - Fortschritts-Updates (POST)

## 🛠️ Fehlerbehebung

### Dashboard startet nicht:
```bash
pip install flask flask-socketio eventlet requests
```

### Port bereits belegt:
```bash
python main.py --dashboard --port 8080
```

### Browser öffnet sich nicht:
```bash
python main.py --dashboard --no-browser
# Dann manuell öffnen: http://localhost:5000
```

## 🎉 Viel Spaß beim Live-Monitoring!

Das Dashboard macht die Verarbeitung von Lane Detection zu einem visuellen Erlebnis. Sie können in Echtzeit sehen, wie Ihre Algorithmen arbeiten und deren Performance vergleichen!
