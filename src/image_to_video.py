import cv2
import os
import config as cfg
import pandas as pd

# Pfade anpassen:
loading_directory = os.path.join('data', 'debug', cfg.CARLA_TOWN)
metadata_path    = os.path.join('data', 'rawimages', cfg.CARLA_TOWN, 'metadata.csv')
output_video     = os.path.join('data', f"{cfg.CARLA_TOWN}.mp4")

print("Loading images from:", loading_directory)
print("Loading metadata from:", metadata_path)

# Metadaten einlesen
if os.path.exists(metadata_path):
    df = pd.read_csv(metadata_path)
    print("Spalten in CSV:", df.columns.tolist())
    metadata_available = True
else:
    print("WARNUNG: Keine metadata.csv gefunden – kein Overlay.")
    metadata_available = False

# Bilddateien sammeln
images = [f for f in os.listdir(loading_directory) if f.lower().endswith('.jpg')]
images.sort()
print(f"{len(images)} Bilder gefunden.")

if not images:
    raise RuntimeError(f"Keine Bilder im Verzeichnis {loading_directory} gefunden.")

# Videowriter vorbereiten
eerste_frame = cv2.imread(os.path.join(loading_directory, images[0]))
h, w = eerste_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (w, h))

# Für Ordered Mapping: Jede Bildposition -> gleichzeitige Metadatenreihe
for idx, img_name in enumerate(images):
    frame = cv2.imread(os.path.join(loading_directory, img_name))
    if frame is None:
        print("Warnung: Kann Bild nicht laden:", img_name)
        continue

    if metadata_available:
        if idx < len(df):
            row = df.iloc[idx]
            dx = row.get('distance_to_center', 0)
            lw = row.get('lane_width', 0)
            ye = row.get('yaw_error', 0)
            text = f"Offset: {dx:.2f}m | LaneW: {lw:.2f}m | YawErr: {ye:.1f}°"
            # Halbtransparenter Balken
            overlay = frame.copy()
            cv2.rectangle(overlay, (10,10), (w-10,50), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            # Text einblenden
            cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2, cv2.LINE_AA)
        else:
            # Kein entsprechender Metadatensatz
            pass

    out.write(frame)
    # Live-Ansicht
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
print("Fertig! Video gespeichert unter", output_video)
