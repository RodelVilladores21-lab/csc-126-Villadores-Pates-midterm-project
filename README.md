# Motion-Triggered Security Monitor

**CSC 126 – Computer Vision Midterm Project** · Caraga State University, CCIS
**Members:** _Villadores, Pates_ 

A live-webcam application that detects and tracks moving objects, labels
moving people with a Haar face check, and raises an alert (on-screen banner,
saved snapshot, CSV log entry) when sustained motion is seen.

## OpenCV techniques used

| Lab topic | Where in the code | What it does |
|---|---|---|
| a. Image Operations | `preprocess()`, `draw_overlays()` | resize, grayscale, Gaussian blur, flip, drawing boxes/text/trails |
| b. Image Segmentation | `segment_motion()` | MOG2 background subtraction, threshold, morphological open / dilate / close |
| d. Image Analysis | `extract_detections()` | contours, area filter, bounding box, centroid |
| e. Motion & Object Tracking | `CentroidTracker` | persistent IDs, trails, direction of travel |
| c. Haar Cascade | `label_people()` | face check inside moving regions → `PERSON` vs `MOTION` |

## Pipeline

```
webcam frame → resize/gray/blur → MOG2 + threshold + morphology → contours
            → centroid tracker → Haar face check → overlays → alert + log
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run

```bash
python motion_security.py                          # live webcam (index 0)
python motion_security.py --source 1               # another camera
python motion_security.py --source sample_inputs/sample_motion.avi   # recorded video
python make_sample_video.py                        # regenerate the sample video
python evaluate_sample.py                          # accuracy on the sample video
```

**Keys:** `q`/`Esc` quit · `r` reset background · `s` snapshot · `m` show mask
· `f` toggle face check · `z` toggle watch zone

Tip: when the app starts, keep the scene still for ~1 second so the background
model can learn the empty room.

## Outputs

- `snapshots/alert_*.jpg` – annotated frame saved at each alert
- `logs/events.csv` – timestamp, object count, labels, snapshot path
- `screenshots/` – pipeline figures used in the documentation

## Tunable settings (top of `motion_security.py`)

`MIN_CONTOUR_AREA`, `MIN_TRACK_AGE`, `MAX_MATCH_DISTANCE`,
`ALERT_COOLDOWN_SEC`, `WARMUP_FRAMES`.

## Known limitations

Lighting changes and camera shake can trigger false alerts; a very slow or
motionless person fades into the background; Haar face detection needs a
roughly frontal face.
