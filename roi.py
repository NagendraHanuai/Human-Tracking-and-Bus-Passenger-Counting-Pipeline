# process_seg_json_optimized_ROI_fullframe.py
import cv2
import os
import time
import threading
import queue
import psutil
import numpy as np
import statistics
import sys
import json
from datetime import datetime
from ultralytics import YOLO

# ============================================================================
# USER CONFIGURATION – Adjust these to match your scene
# ============================================================================
MODEL_PATH = 'yolo26x.pt'                # YOLO model file
VIDEO_PATH = r"NORM0002.MP4"          # Input video
OUTPUT_PATH = "Jul_14_A1_ROI.MP4"  # Output video (full frame)
JSON_OUTPUT_PATH = "Jul_14_A1_ROI.json"   # JSON log

CLASSES = [0]                            # Class IDs to track (0=person, 5=bus, etc.)
CONFIDENCE = 0.30
# MAX_WIDTH is now ignored – output video keeps original resolution
# MAX_WIDTH = 3840                         # (kept for reference, not used)

# ---------- ROI DEFINITION (normalised 0‑1) ----------
# This polygon defines the area where detections are considered.
# The model will only process this region; outside is ignored.
ROI_POLYGON_NORM = [
    (0.5 - 300/640/2, 0.5 - 300/480/2),   # top‑left (box corner)
    (0.5 + 300/640/2, 0.5 - 300/480/2),   # top‑right
    (0.5 + 300/640/2, 0.5),               # right‑box‑mid
    (1.0, 0.5),                           # right‑mid (frame edge)
    (1.0, 1.0),                           # bottom‑right
    (0.0, 1.0),                           # bottom‑left
    (0.0, 0.5),                           # left‑mid
    (0.5 - 300/640/2, 0.5),               # left‑box‑mid
]
# If you want a simple rectangle instead, use:
# ROI_POLYGON_NORM = [(0.2,0.2), (0.8,0.2), (0.8,0.8), (0.2,0.8)]

ROI_MASK_OUTSIDE = False                  # **** KEEP FALSE – we never black out the output ****
ROI_DRAW_POLYGON = False                  # Draw the ROI border on output (visual guide)

# Counting line definitions (normalized 0-1)
SINGLE_COUNTING_ZONES = [{
    "name": "main",
    "entry": ((0.48, 0.53), (0.50, 0.44)),   # IN (green)
    "exit":  ((0.50, 0.54), (0.52, 0.45)),   # OUT (red)
}]
DOUBLE_COUNTING_ZONES = [{
    "name": "near_door",
    "entry": ((0.365, 0.910), (0.435, 0.810)),
    "exit":  ((0.425, 0.920), (0.495, 0.820)),
}]
LINE_LAYOUT_MODE = "double"               # "single", "double", or "auto"
AUTO_SINGLE_LINE_MIN_PERSONS = 10
INITIAL_VISIBLE_PERSONS_ARE_IN = True     # Seed initial people as "inside"
INITIAL_INSIDE_SEED_SECONDS = 2

# Tracking parameters
TRACKER_TYPE = "botsort.yaml"           # "botsort.yaml" or "bytetrack.yaml"
LINE_CROSS_MIN_MOVE = 8                   # Minimum movement (pixels) to trigger crossing
EVENT_DISPLAY_SECONDS = 2

# ID relinking thresholds (when tracker loses an ID)
ID_RELINK_IOU = 0.25                     # More forgiving than 0.20
ID_RELINK_CENTER_DIST = 0.60             # Allow slightly larger displacement
MAX_RELINK_MISSING_SECONDS = 30

# Performance
PROGRESS_BAR_WIDTH = 30
PROGRESS_UPDATE_INTERVAL = 0.5
SHOW_COUNTING_LINES = False               # Set True for visual debugging
SHOW_DETECTIONS = True                    # Draw bounding boxes and IDs

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

# --- Global state ---
json_results = []
frame_index = 0
counted_person_ids = set()
raw_track_to_person_id = {}
person_last_boxes = {}
person_last_seen = {}
next_person_id = 1
person_last_centers = {}
counted_in_ids = set()
counted_out_ids = set()
in_count = 0
out_count = 0
recent_events = []
last_progress_update = 0
start_time = 0

# --- Load model ---
model = YOLO(MODEL_PATH, task="detect")
model.fuse()  # Fuse layers for speed

# --- Open video ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_length_sec = frame_count / fps if fps > 0 else 0

print(f"🎬 Video: {frame_count} frames, {fps:.2f} FPS, {video_length_sec:.2f} sec, {width}x{height}")

# Output video will have the same resolution as input (no resizing)
out_width, out_height = width, height

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (out_width, out_height))

# Queues (larger for smoother threading)
decode_queue = queue.Queue(maxsize=60)
infer_queue = queue.Queue(maxsize=60)

# Benchmark lists
decode_times = []
infer_times = []
memory_usage = []

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def format_duration(seconds):
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def update_progress_bar(force=False):
    global last_progress_update
    now = time.time()
    if not force and now - last_progress_update < PROGRESS_UPDATE_INTERVAL:
        return
    last_progress_update = now
    processed = min(frame_index, frame_count) if frame_count > 0 else frame_index
    total = frame_count if frame_count > 0 else max(processed, 1)
    percent = processed / total if total > 0 else 0
    filled = int(PROGRESS_BAR_WIDTH * percent)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    elapsed = max(now - start_time, 1e-6)
    proc_fps = processed / elapsed
    rem = max(total - processed, 0)
    eta = rem / proc_fps if proc_fps > 0 else 0
    occ = max(in_count - out_count, 0)
    sys.stdout.write(f"\rProcessing [{bar}] {processed}/{total} {percent*100:6.2f}% | "
                     f"FPS {proc_fps:5.2f} | ETA {format_duration(eta)} | "
                     f"IN {in_count} OUT {out_count} OCC {occ}")
    sys.stdout.flush()

def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def center_distance_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx, acy = (ax1+ax2)/2, (ay1+ay2)/2
    bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
    dist = ((acx-bcx)**2 + (acy-bcy)**2)**0.5
    avg_w = ((ax2-ax1)+(bx2-bx1))/2
    avg_h = ((ay2-ay1)+(by2-by1))/2
    scale = max(avg_w, avg_h, 1)
    return dist / scale

# ---------- ROI polygon helpers ----------
def point_in_polygon(px, py, polygon):
    """Ray casting algorithm. polygon: list of (x,y) tuples."""
    inside = False
    n = len(polygon)
    x, y = px, py
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def get_roi_polygon(frame_shape):
    """Convert normalised polygon to pixel coordinates for the given frame."""
    h, w = frame_shape[:2]
    return [(int(x * w), int(y * h)) for (x, y) in ROI_POLYGON_NORM]

# --- NEW: compute bounding rectangle of polygon ---
def get_roi_bbox(polygon):
    """Return (minx, miny, maxx, maxy) of the polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)

def crop_and_mask_frame(frame, polygon):
    """
    Crop the frame to the bounding rectangle of the polygon,
    then mask out pixels outside the polygon (set to black).
    Returns (cropped_masked, offset_x, offset_y, crop_width, crop_height)
    where offset is the top-left corner of the crop in original frame coords.
    """
    minx, miny, maxx, maxy = get_roi_bbox(polygon)
    # Ensure integer and within frame bounds
    minx = max(0, int(minx))
    miny = max(0, int(miny))
    maxx = min(frame.shape[1], int(maxx))
    maxy = min(frame.shape[0], int(maxy))
    if maxx <= minx or maxy <= miny:
        # Invalid ROI – return a small black image
        return np.zeros((10, 10, 3), dtype=np.uint8), 0, 0, 10, 10
    crop = frame[miny:maxy, minx:maxx]
    h, w = crop.shape[:2]
    # Shift polygon to crop coordinates
    shifted_poly = [(x - minx, y - miny) for (x, y) in polygon]
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(shifted_poly, dtype=np.int32)], 255)
    # Apply mask
    masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
    return masked_crop, minx, miny, w, h

def draw_roi_polygon(frame, polygon):
    """Draw the ROI border on the frame."""
    cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (0, 255, 255), 3)

# ----------------------------------------------------------------------------
# ID relinking (same as original)
# ----------------------------------------------------------------------------
def get_corrected_person_id(raw_track_id, bbox, current_frame_person_ids):
    """Attempt to relink a raw tracker ID to a persistent person ID."""
    global next_person_id
    if raw_track_id is not None and raw_track_id in raw_track_to_person_id:
        return raw_track_to_person_id[raw_track_id]

    best_id = None
    best_score = -1
    max_missing_frames = int(max(fps, 1) * MAX_RELINK_MISSING_SECONDS)
    for pid, last_box in person_last_boxes.items():
        if pid in current_frame_person_ids:
            continue
        if frame_index - person_last_seen.get(pid, frame_index) > max_missing_frames:
            continue
        iou = bbox_iou(bbox, last_box)
        dist = center_distance_ratio(bbox, last_box)
        if iou >= ID_RELINK_IOU or dist <= ID_RELINK_CENTER_DIST:
            score = iou + (1 - min(dist, 1))
            if score > best_score:
                best_score = score
                best_id = pid

    if best_id is None:
        best_id = next_person_id
        next_person_id += 1

    if raw_track_id is not None:
        raw_track_to_person_id[raw_track_id] = best_id
    return best_id

# --- Drawing helpers ---
def draw_detection(frame, bbox, person_id, conf):
    if not SHOW_DETECTIONS:
        return
    x1, y1, x2, y2 = map(int, bbox)
    # label = f"HUMAN:{person_id}"
    label = f"HUMAN"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,80,0), 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    ly1 = max(y1 - th - 12, 0)
    ly2 = ly1 + th + 8
    cv2.rectangle(frame, (x1, ly1), (x1+tw+8, ly2), (255,80,0), -1)
    cv2.putText(frame, label, (x1+4, ly2-4), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255,255,255), 3, cv2.LINE_AA)

def text_with_outline(frame, text, org, font, scale, color, thickness=2):
    cv2.putText(frame, text, org, font, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)

# --- Line operations ---
def normalized_line_to_pixels(line_norm, frame_shape):
    h, w = frame_shape[:2]
    return tuple((int(x*w), int(y*h)) for x,y in line_norm)

def normalized_zones_to_pixels(zones, frame_shape):
    return [{
        "name": z["name"],
        "entry": normalized_line_to_pixels(z["entry"], frame_shape),
        "exit":  normalized_line_to_pixels(z["exit"], frame_shape),
    } for z in zones]

def select_counting_zones(frame_person_count):
    if LINE_LAYOUT_MODE == "single":
        return SINGLE_COUNTING_ZONES, "single"
    if LINE_LAYOUT_MODE == "double":
        return DOUBLE_COUNTING_ZONES, "double"
    if frame_person_count >= AUTO_SINGLE_LINE_MIN_PERSONS:
        return SINGLE_COUNTING_ZONES, "single"
    return DOUBLE_COUNTING_ZONES, "double"

def point_side(point, line):
    (x1,y1),(x2,y2) = line
    px,py = point
    return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)

def orientation(a,b,c):
    v = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
    if abs(v) < 1e-6: return 0
    return 1 if v>0 else 2

def on_segment(a,b,c):
    return (min(a[0],c[0]) <= b[0] <= max(a[0],c[0]) and
            min(a[1],c[1]) <= b[1] <= max(a[1],c[1]))

def segments_intersect(p1,q1,p2,q2):
    o1 = orientation(p1,q1,p2)
    o2 = orientation(p1,q1,q2)
    o3 = orientation(p2,q2,p1)
    o4 = orientation(p2,q2,q1)
    if o1 != o2 and o3 != o4: return True
    if o1==0 and on_segment(p1,p2,q1): return True
    if o2==0 and on_segment(p1,q2,q1): return True
    if o3==0 and on_segment(p2,p1,q2): return True
    if o4==0 and on_segment(p2,q1,q2): return True
    return False

def crossed_line(prev_center, current_center, line):
    if prev_center is None:
        return False
    move = np.linalg.norm(np.array(current_center) - np.array(prev_center))
    if move < LINE_CROSS_MIN_MOVE:
        return False
    prev_side = point_side(prev_center, line)
    curr_side = point_side(current_center, line)
    if prev_side == 0 or curr_side == 0:
        return segments_intersect(prev_center, current_center, line[0], line[1])
    return prev_side * curr_side < 0 and segments_intersect(prev_center, current_center, line[0], line[1])

# --- Event handling ---
def add_event(person_id, event_type):
    expiry = frame_index + int(max(fps,1) * EVENT_DISPLAY_SECONDS)
    recent_events.append({"person_id": person_id, "event_type": event_type, "expiry_frame": expiry})

# --- Dashboard ---
def draw_dashboard(frame, stats):
    total = stats['in_count'] + stats['out_count']
    lines = [
        f"IN Count: {stats['in_count']}",
        f"OUT Count: {stats['out_count']}",
        f"Total: {total}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 3.0
    thick = 7
    pad = 16
    gap = 9
    sizes = [cv2.getTextSize(l, font, fs, thick)[0] for l in lines]
    bw = max(s[0] for s in sizes) + 2*pad
    bh = sum(s[1] for s in sizes) + 2*pad + gap*(len(lines)-1)
    x1 = max(frame.shape[1] - bw - 20, 0)
    y1 = 20
    x2 = frame.shape[1] - 20
    y2 = y1 + bh
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (245,245,245), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (30,30,30), 2)
    ty = y1 + pad + sizes[0][1]
    for line, size in zip(lines, sizes):
        cv2.putText(frame, line, (x2 - pad - size[0], ty),
                    font, fs, (20,20,20), thick, cv2.LINE_AA)
        ty += size[1] + gap
    return frame

def draw_virtual_lines(frame, pixel_zones, layout_name):
    if not SHOW_COUNTING_LINES:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    for zone in pixel_zones:
        cv2.line(frame, zone["entry"][0], zone["entry"][1], (0,255,0), 6, cv2.LINE_AA)
        cv2.line(frame, zone["exit"][0], zone["exit"][1], (0,0,255), 6, cv2.LINE_AA)
        ex = (zone["entry"][0][0]+zone["entry"][1][0])//2
        ey = (zone["entry"][0][1]+zone["entry"][1][1])//2
        ox = (zone["exit"][0][0]+zone["exit"][1][0])//2
        oy = (zone["exit"][0][1]+zone["exit"][1][1])//2
        text_with_outline(frame, "ENTRY (IN)", (ex-45, ey-12), font, 0.55, (255,255,255), 2)
        text_with_outline(frame, "EXIT (OUT)", (ox-40, oy+26), font, 0.55, (255,255,255), 2)
        cv2.arrowedLine(frame, (ex, ey-45), (ex, ey+45), (0,255,0), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(frame, (ox, oy+45), (ox, oy-45), (0,0,255), 3, cv2.LINE_AA, tipLength=0.25)

# ============================================================================
# Threading functions
# ============================================================================
def decode_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        decode_queue.put(frame)
    decode_queue.put(None)

def run_inference():
    global frame_index, in_count, out_count, json_results
    while True:
        frame = decode_queue.get()
        if frame is None:
            break
        start_t = time.time()

        # ---- 1. Get ROI polygon and crop/mask ----
        roi_polygon = get_roi_polygon(frame.shape)
        masked_crop, offset_x, offset_y, crop_w, crop_h = crop_and_mask_frame(frame, roi_polygon)

        # ---- 2. Run inference on the cropped + masked image ----
        # Use model.track to keep ByteTrack. The tracker will operate in crop coordinates.
        results = model.track(
            masked_crop,
            classes=CLASSES,
            conf=CONFIDENCE,
            device=0,
            persist=True,
            tracker=TRACKER_TYPE,
            verbose=False
        )[0]
        if hasattr(results, "masks") and results.masks is not None:
            results.masks = None

        # ---- 3. Prepare frame data for JSON ----
        frame_data = {
            "frame_id": frame_index,
            "timestamp": time.time(),
            "frame_person_count": 0,
            "total_unique_person_count": len(counted_person_ids),
            "in_count": in_count,
            "out_count": out_count,
            "occupancy": max(in_count - out_count, 0),
            "line_layout": None,
            "counting_zones": [],
            "person_ids": [],
            "events": [],
            "detections": []
        }

        detected_boxes = results.boxes if results.boxes is not None else []
        active_zones, line_layout = select_counting_zones(len(detected_boxes))
        pixel_zones = normalized_zones_to_pixels(active_zones, frame.shape)
        frame_data["line_layout"] = line_layout
        frame_data["counting_zones"] = active_zones

        current_person_ids = set()

        # ---- 4. Process each detection ----
        if len(detected_boxes) > 0:
            for box in detected_boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                # bbox in crop coordinates
                crop_bbox = list(map(float, box.xyxy[0]))
                cx1, cy1, cx2, cy2 = crop_bbox
                track_id = int(box.id[0]) if box.id is not None else None

                # Convert to full-frame coordinates
                x1 = cx1 + offset_x
                y1 = cy1 + offset_y
                x2 = cx2 + offset_x
                y2 = cy2 + offset_y
                full_bbox = [x1, y1, x2, y2]

                # ---- ROI safety filter ----
                foot_x = (x1 + x2) / 2
                foot_y = y2
                if not point_in_polygon(foot_x, foot_y, roi_polygon):
                    continue   # skip if outside (should not happen due to mask, but keep for safety)

                # Assign persistent person ID
                person_id = get_corrected_person_id(track_id, full_bbox, current_person_ids)
                if person_id in current_person_ids:
                    continue   # duplicate detection
                is_new_person = person_id not in counted_person_ids
                current_person_ids.add(person_id)
                counted_person_ids.add(person_id)
                person_last_boxes[person_id] = full_bbox
                person_last_seen[person_id] = frame_index

                foot_center = (foot_x, foot_y)
                event_type = None
                event_zone = None

                # Check for line crossing
                if person_id is not None:
                    prev_center = person_last_centers.get(person_id)
                    # Seed initial inside
                    seed_inside = (INITIAL_VISIBLE_PERSONS_ARE_IN and
                                   is_new_person and
                                   frame_index <= int(max(fps,1)*INITIAL_INSIDE_SEED_SECONDS))
                    if seed_inside and person_id not in counted_in_ids:
                        in_count += 1
                        counted_in_ids.add(person_id)
                        event_type = "IN"
                        event_zone = "initial_inside"
                        add_event(person_id, event_type)
                    else:
                        for zone in pixel_zones:
                            if (person_id not in counted_in_ids and
                                crossed_line(prev_center, foot_center, zone["entry"])):
                                in_count += 1
                                counted_in_ids.add(person_id)
                                event_type = "IN"
                                event_zone = zone["name"]
                                add_event(person_id, event_type)
                                break
                            if (person_id not in counted_out_ids and
                                crossed_line(prev_center, foot_center, zone["exit"])):
                                out_count += 1
                                counted_out_ids.add(person_id)
                                event_type = "OUT"
                                event_zone = zone["name"]
                                add_event(person_id, event_type)
                                break
                    person_last_centers[person_id] = foot_center

                # Store detection info (full-frame coordinates)
                det = {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "raw_track_id": track_id,
                    "person_id": person_id,
                    "foot_center": [foot_x, foot_y],
                    "confidence": round(conf, 4),
                    "bbox_xyxy": full_bbox
                }
                frame_data["detections"].append(det)
                if event_type is not None:
                    frame_data["events"].append({
                        "person_id": person_id,
                        "event_type": event_type,
                        "zone": event_zone
                    })

        # ---- 5. Update stats ----
        frame_person_count = len(current_person_ids)
        occupancy = max(in_count - out_count, 0)
        visible_inside = min(frame_person_count, occupancy)
        visible_outside = max(frame_person_count - visible_inside, 0)
        frame_data["frame_person_count"] = frame_person_count
        frame_data["total_unique_person_count"] = len(counted_person_ids)
        frame_data["in_count"] = in_count
        frame_data["out_count"] = out_count
        frame_data["occupancy"] = occupancy
        frame_data["inside_bus"] = visible_inside
        frame_data["outside_bus"] = visible_outside
        frame_data["person_ids"] = sorted(current_person_ids)

        json_results.append(frame_data)
        frame_index += 1

        infer_times.append(time.time() - start_t)
        memory_usage.append(get_memory_usage())

        # ---- 6. Annotate full frame ----
        annotated = frame.copy()

        # Draw ROI border (visual guide)
        if ROI_DRAW_POLYGON:
            draw_roi_polygon(annotated, roi_polygon)

        # Draw detections (using full-frame coordinates)
        for det in frame_data["detections"]:
            if det["person_id"] is not None:
                draw_detection(annotated, det["bbox_xyxy"], det["person_id"], det["confidence"])

        draw_virtual_lines(annotated, pixel_zones, line_layout)

        # Clean expired events
        global recent_events
        recent_events = [e for e in recent_events if e["expiry_frame"] >= frame_index]

        # Dashboard
        elapsed_frame = max(time.time() - start_t, 1e-6)
        stats = {
            "frame": frame_data["frame_id"],
            "persons_detected": frame_person_count,
            "inside_bus": visible_inside,
            "outside_bus": visible_outside,
            "in_count": in_count,
            "out_count": out_count,
            "occupancy": occupancy,
            "total_unique": len(counted_person_ids),
            "line_layout": line_layout,
            "fps": 1 / elapsed_frame
        }
        annotated = draw_dashboard(annotated, stats)

        # Output video keeps original resolution – no resize
        infer_queue.put(annotated)
        update_progress_bar()

    infer_queue.put(None)

def write_frames():
    while True:
        frame = infer_queue.get()
        if frame is None:
            break
        out.write(frame)

# ============================================================================
# Main execution
# ============================================================================
start_time = time.time()

t1 = threading.Thread(target=decode_frames)
t2 = threading.Thread(target=run_inference)
t3 = threading.Thread(target=write_frames)

t1.start(); t2.start(); t3.start()
t1.join(); t2.join(); t3.join()

update_progress_bar(force=True)
print()

cap.release()
out.release()

if not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) < 1024:
    raise RuntimeError("❌ Output video corrupted or empty")

elapsed = time.time() - start_time

# --- Save JSON ---
final_json = {
    "meta": {
        "model_path": MODEL_PATH,
        "video_path": VIDEO_PATH,
        "output_video": OUTPUT_PATH,
        "generated_at": datetime.now().isoformat(),
        "fps": fps,
        "resolution": f"{width}x{height}",
        "total_frames": frame_count,
        "confidence_threshold": CONFIDENCE,
        "tracker": TRACKER_TYPE,
        "classes": CLASSES,
        "unique_person_count": len(counted_person_ids),
        "line_layout_mode": LINE_LAYOUT_MODE,
        "single_counting_zones": SINGLE_COUNTING_ZONES,
        "double_counting_zones": DOUBLE_COUNTING_ZONES,
        "in_count": in_count,
        "out_count": out_count,
        "occupancy": max(in_count - out_count, 0),
        "id_relink_iou": ID_RELINK_IOU,
        "id_relink_center_dist": ID_RELINK_CENTER_DIST,
        "max_relink_missing_seconds": MAX_RELINK_MISSING_SECONDS,
        "device": str(model.device),
        "roi_polygon_norm": ROI_POLYGON_NORM,
        "roi_mask_outside": ROI_MASK_OUTSIDE,
        "roi_inference_only": True,   # added flag
    },
    "frames": json_results
}
with open(JSON_OUTPUT_PATH, "w") as f:
    json.dump(final_json, f, indent=4)

print(f"📄 JSON saved to: {os.path.abspath(JSON_OUTPUT_PATH)}")

# --- Benchmark ---
latency_ms = [t*1000 for t in infer_times]
p50 = np.percentile(latency_ms, 50)
p95 = np.percentile(latency_ms, 95)
avg_lat = np.mean(latency_ms)
avg_fps = frame_count / elapsed if elapsed>0 else 0
fps_per_frame = [1/t for t in infer_times if t>0]
avg_mem = np.mean(memory_usage) if memory_usage else 0
max_mem = np.max(memory_usage) if memory_usage else 0
window = min(30, len(fps_per_frame))
sliding = fps_per_frame[-window:] if window>0 else fps_per_frame

print("\n===== OPTIMIZED BENCHMARK =====")
print(f"Frames: {frame_count}, Time: {elapsed:.2f}s, Avg FPS: {avg_fps:.2f}")
print(f"Latency (ms) avg: {avg_lat:.2f}, p50: {p50:.2f}, p95: {p95:.2f}")
print(f"RAM avg: {avg_mem:.2f} MB, max: {max_mem:.2f} MB")
print(f"FPS stability (last {window} frames): min {np.min(sliding):.2f}, max {np.max(sliding):.2f}")
print(f"Unique persons: {len(counted_person_ids)}, IN: {in_count}, OUT: {out_count}, Occupancy: {max(in_count-out_count,0)}")
print(f"✅ Output video: {os.path.abspath(OUTPUT_PATH)}")
print("=================================")