"""
Motion-Triggered Security Monitor

A live-webcam security application built with classical OpenCV techniques.
It watches a scene, separates moving objects from the static background,
tracks them from frame to frame, and raises an alert (with snapshot + log)
when sustained motion is detected inside a user-defined watch zone.

OpenCV topics used (from the laboratory list):
    a. Image Operations ............ grayscale, Gaussian blur, resize, flip, drawing
    b. Image Segmentation .......... MOG2 background subtraction, thresholding,
                                     morphological opening / dilation
    d. Image Analysis .............. contour extraction, area, bounding box, centroid
    e. Motion & Object Tracking .... centroid tracker with IDs, trails, direction
    c. Haar Cascade (optional) ..... face check inside moving regions to label
                                     "PERSON" vs. generic "MOTION"

Controls (while the window is focused):
    q / ESC  quit            r  reset background model
    s        save snapshot   m  toggle mask view
    f        toggle Haar face check
    z        toggle watch-zone restriction (zone = center 80% of the frame)

Run:
    python motion_security.py                      # default webcam (index 0)
    python motion_security.py --source video.mp4   # recorded video instead
"""

import argparse
import csv
import math
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np



# 1. CONFIGURATION

FRAME_WIDTH = 640            # every frame is resized to this width for speed
WARMUP_FRAMES = 30           # frames used to learn the background before alerts
MIN_CONTOUR_AREA = 800       # ignore blobs smaller than this (noise, in pixels)
MAX_MATCH_DISTANCE = 80      # max centroid jump (px) to treat as the same object
MAX_MISSING_FRAMES = 15      # drop a track after this many frames unseen
MIN_TRACK_AGE = 8            # frames an object must persist to count as real
ALERT_COOLDOWN_SEC = 3.0     # minimum seconds between two saved alerts
TRAIL_LENGTH = 30            # number of past centroids drawn as a trail

SNAPSHOT_DIR = "snapshots"
LOG_FILE = os.path.join("logs", "events.csv")

# 2. OBJECT TRACKER  (Topic e: Motion and Object Tracking)

class TrackedObject:
    """Holds the state of one tracked moving object."""

    def __init__(self, object_id, centroid, bbox):
        self.object_id = object_id
        self.centroid = centroid
        self.bbox = bbox
        self.trail = deque([centroid], maxlen=TRAIL_LENGTH)
        self.age = 1               # frames since first seen
        self.missing = 0           # consecutive frames not matched
        self.label = "MOTION"      # becomes "PERSON" if a face is found

    def update(self, centroid, bbox):
        self.centroid = centroid
        self.bbox = bbox
        self.trail.append(centroid)
        self.age += 1
        self.missing = 0

    def direction(self):
        """Return a compass-style direction from the recent trail."""
        if len(self.trail) < 6:
            return "-"
        dx = self.trail[-1][0] - self.trail[-6][0]
        dy = self.trail[-1][1] - self.trail[-6][1]
        if math.hypot(dx, dy) < 8:
            return "still"
        horiz = "right" if dx > 0 else "left"
        vert = "down" if dy > 0 else "up"
        return horiz if abs(dx) >= abs(dy) else vert


class CentroidTracker:
    """Greedy nearest-centroid tracker that assigns persistent IDs."""

    def __init__(self):
        self.next_id = 1
        self.objects = {}

    def update(self, detections):
        """
        detections: list of (centroid, bbox) found in the current frame.
        Returns the dict of currently tracked objects {id: TrackedObject}.
        """
        # Nothing detected: every existing track has now been missed once more.
        if not detections:
            for obj in list(self.objects.values()):
                obj.missing += 1
                if obj.missing > MAX_MISSING_FRAMES:
                    del self.objects[obj.object_id]
            return self.objects

        # Build all (distance, track_id, detection_index) pairs, closest first.
        pairs = []
        for tid, obj in self.objects.items():
            for di, (centroid, _) in enumerate(detections):
                dist = math.hypot(obj.centroid[0] - centroid[0],
                                  obj.centroid[1] - centroid[1])
                if dist <= MAX_MATCH_DISTANCE:
                    pairs.append((dist, tid, di))
        pairs.sort()

        used_tracks, used_dets = set(), set()
        for dist, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            centroid, bbox = detections[di]
            self.objects[tid].update(centroid, bbox)
            used_tracks.add(tid)
            used_dets.add(di)

        # Tracks that found no match are missed this frame.
        for tid in list(self.objects):
            if tid not in used_tracks:
                self.objects[tid].missing += 1
                if self.objects[tid].missing > MAX_MISSING_FRAMES:
                    del self.objects[tid]

        # Detections that matched no track become new objects.
        for di, (centroid, bbox) in enumerate(detections):
            if di not in used_dets:
                self.objects[self.next_id] = TrackedObject(self.next_id, centroid, bbox)
                self.next_id += 1

        return self.objects


# 3. PROCESSING STEPS

def preprocess(frame):
    """Topic a (Image Operations): resize, grayscale and blur the frame."""
    h, w = frame.shape[:2]
    scale = FRAME_WIDTH / float(w)
    frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)   # suppress sensor noise
    return frame, gray, blurred


def segment_motion(subtractor, blurred):
    """
    Topic b (Image Segmentation): produce a clean binary mask of moving pixels.
    1) MOG2 background subtraction  -> foreground mask (shadows = 127)
    2) Threshold at 200             -> removes shadows, keeps true foreground
    3) Morphological opening        -> removes tiny speckles
    4) Dilation                     -> grows blobs so nearby pieces touch
    5) Morphological closing        -> fills gaps so one person = one blob
    """
    fg_mask = subtractor.apply(blurred)
    _, mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def extract_detections(mask, zone):
    """
    Topic d (Image Analysis): find contours and measure each blob.
    Returns a list of (centroid, bbox) for blobs big enough and inside the zone.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        centroid = (x + w // 2, y + h // 2)
        if zone is not None:
            zx1, zy1, zx2, zy2 = zone
            if not (zx1 <= centroid[0] <= zx2 and zy1 <= centroid[1] <= zy2):
                continue
        detections.append((centroid, (x, y, w, h)))
    return detections


def load_face_cascade():
    """Topic c (Haar Cascade): load the pretrained frontal-face classifier."""
    try:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(path)
    except AttributeError:
        # Some OpenCV builds (e.g. 5.x core) do not ship the cascade module.
        print("[WARN] CascadeClassifier unavailable - face check disabled. "
              "Install opencv-python 4.x (see requirements.txt).")
        return None
    if cascade.empty():
        print("[WARN] Haar cascade file not found - face check disabled.")
        return None
    return cascade


def label_people(cascade, gray, tracks):
    """Run the Haar face detector only inside each moving region (fast)."""
    for obj in tracks.values():
        x, y, w, h = obj.bbox
        pad = 20
        roi = gray[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            continue
        faces = cascade.detectMultiScale(roi, scaleFactor=1.2,
                                         minNeighbors=5, minSize=(30, 30))
        obj.label = "PERSON" if len(faces) > 0 else "MOTION"



# 4. VISUALISATION  (Topic a: drawing overlays)

def draw_overlays(frame, tracks, zone, alarm, fps, show_mask_hint):
    """Draw zone, bounding boxes, IDs, trails, and the status banner."""
    if zone is not None:
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (255, 200, 0), 1)
        cv2.putText(frame, "WATCH ZONE", (zone[0] + 5, zone[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    for obj in tracks.values():
        if obj.age < MIN_TRACK_AGE or obj.missing > 0:
            continue   # only draw confirmed, currently visible objects
        x, y, w, h = obj.bbox
        color = (0, 0, 255) if obj.label == "PERSON" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"ID {obj.object_id} {obj.label} {obj.direction()}",
                    (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pts = list(obj.trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)
        cv2.circle(frame, obj.centroid, 4, (0, 0, 255), -1)

    banner_color = (0, 0, 255) if alarm else (0, 150, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), banner_color, -1)
    status = "ALERT: MOTION DETECTED" if alarm else "MONITORING - area clear"
    cv2.putText(frame, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"{fps:4.1f} FPS", (frame.shape[1] - 90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame



# 5. EVENT LOGGING

def log_event(frame, tracks):
    """Save a snapshot and append one row to the CSV log."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"alert_{stamp}.jpg")
    cv2.imwrite(path, frame)

    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "objects", "labels", "snapshot"])
        confirmed = [o for o in tracks.values() if o.age >= MIN_TRACK_AGE]
        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                         len(confirmed),
                         "|".join(o.label for o in confirmed), path])
    print(f"[ALERT] {stamp}  objects={len(confirmed)}  saved {path}")



# 6. MAIN LOOP

def parse_args():
    parser = argparse.ArgumentParser(description="Motion-triggered security monitor")
    parser.add_argument("--source", default="0",
                        help="webcam index (default 0) or path to a video file")
    parser.add_argument("--no-display", action="store_true",
                        help="run without a window (for testing on servers)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = run until quit)")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        return 1

    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25,
                                                    detectShadows=True)
    tracker = CentroidTracker()
    cascade = load_face_cascade()
    use_haar = cascade is not None
    use_zone = False
    show_mask = False
    is_webcam = isinstance(source, int)

    frame_count, alerts_saved = 0, 0
    last_alert, prev_time, fps = 0.0, time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        if is_webcam:
            frame = cv2.flip(frame, 1)       # mirror view feels natural on webcam

        # --- Pipeline: preprocess -> segment -> analyse -> track -> classify ---
        frame, gray, blurred = preprocess(frame)
        h, w = frame.shape[:2]
        zone = (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)) if use_zone else None

        mask = segment_motion(subtractor, blurred)
        detections = extract_detections(mask, zone) if frame_count > WARMUP_FRAMES else []
        tracks = tracker.update(detections)
        if use_haar:
            label_people(cascade, gray, tracks)

        confirmed = [o for o in tracks.values() if o.age >= MIN_TRACK_AGE and o.missing == 0]
        alarm = len(confirmed) > 0

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        output = draw_overlays(frame.copy(), tracks, zone, alarm, fps, show_mask)
        if alarm and now - last_alert > ALERT_COOLDOWN_SEC:
            log_event(output, tracks)
            last_alert = now
            alerts_saved += 1

        if not args.no_display:
            if show_mask:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Motion Security Monitor", np.hstack([output, mask_bgr]))
            else:
                cv2.imshow("Motion Security Monitor", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                subtractor = cv2.createBackgroundSubtractorMOG2(500, 25, True)
                tracker = CentroidTracker()
                frame_count = 0
                print("[INFO] Background model reset.")
            elif key == ord("s"):
                log_event(output, tracks)
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("f") and cascade is not None:
                use_haar = not use_haar
            elif key == ord("z"):
                use_zone = not use_zone

        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"[DONE] frames={frame_count}  alerts_saved={alerts_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())