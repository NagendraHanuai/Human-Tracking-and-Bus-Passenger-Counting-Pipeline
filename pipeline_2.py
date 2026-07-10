# process_seg_json.py
import cv2
from ultralytics import YOLO
import os
import time
import threading
import queue
import psutil
import numpy as np
import statistics
import sys

# ===================== ADDED =====================
import json
from datetime import datetime
# =================================================

# ---------------- CONFIG ----------------
MODEL_PATH = 'yolo26x.pt'







VIDEO_PATH = r"videos_data/NORM0002.MP4"
OUTPUT_PATH = "videos_data/results_data/A1_NORM0002.MP4"
JSON_OUTPUT_PATH = "videos_data/results_data/A1_NORM0002.json"

CONFIDENCE = 0.30
MAX_WIDTH = 3840

# Ignore people detected through bus windows / outside view.  Each box is
# normalized as (x1, y1, x2, y2). A detection is rejected only when its center
# is inside one of these zones and its feet are still above that zone bottom.
OUTSIDE_VIEW_ZONES = [
    (0.00, 0.00, 0.36, 0.58),  # left windows / road
    (0.39, 0.25, 0.49, 0.68),  # left door-window outside view
    (0.78, 0.00, 1.00, 0.74),  # right windows / road
]
MIN_PERSON_BOX_AREA_RATIO = 0.00025
MIN_PERSON_HEIGHT_RATIO = 0.055

# Virtual counting line coordinates are normalized (x, y), so they work after
# resolution changes. Tune these points once for your camera view.
LINE_LAYOUT_MODE = "single"  # "auto", "single", or "double"
AUTO_SINGLE_LINE_MIN_PERSONS = 10
INITIAL_VISIBLE_PERSONS_ARE_IN = True
INITIAL_INSIDE_SEED_SECONDS = 2

SINGLE_COUNTING_ZONES = [
    {
        "name": "main",
        "entry": ((0.48, 0.53), (0.50, 0.44)),  # Green line: ENTRY (IN)
        # "exit": ((0.50, 0.54), (0.52, 0.45)),   # Red line: EXIT (OUT)
        "exit": ((0.425, 0.920), (0.495, 0.820)),
    }
]

DOUBLE_COUNTING_ZONES = [
    {
        "name": "near_door",
        # "entry": ((0.465, 0.835), (0.305, 0.500)),
        # "exit": ((0.430, 0.865), (0.380, 0.500)),

        # "entry": ((0.28, 0.68), (0.38, 0.52)),
        # "exit": ((0.35, 0.82), (0.46, 0.58)),

        # "entry": ((0.425, 0.835),(0.495, 0.735)),

        # "exit": ((0.485, 0.845),(0.555, 0.745)),

        #  "entry": (
        #     (0.395, 0.865),
        #     (0.465, 0.765)
        # ),

        # # EXIT (Red)
        # "exit": (
        #     (0.455, 0.875),
        #     (0.525, 0.775)
        # ),

        # "entry": (
        #     (0.380, 0.885),
        #     (0.450, 0.785)
        # ),

        # # EXIT (Red)
        # "exit": (
        #     (0.440, 0.895),
        #     (0.510, 0.795)
        # ),

        "entry": (
            (0.365, 0.910),
            (0.435, 0.810)
        ),

        # EXIT (Red)
        "exit": (
            (0.425, 0.920),
            (0.495, 0.820)
        ),




    },
    {
        "name": "far_door",
        "entry": ((0.500, 0.705), (0.535, 0.585)),
        "exit": ((0.535, 0.720), (0.570, 0.585)),
    },
]

LINE_CROSS_MIN_MOVE = 8
EVENT_DISPLAY_SECONDS = 2
PROGRESS_BAR_WIDTH = 30
PROGRESS_UPDATE_INTERVAL = 0.5

# Merge a new tracker ID back into an old person when the person is seated in
# almost the same place. This fixes false count increases caused by ID switches.
ID_RELINK_IOU = 0.12
ID_RELINK_CENTER_DIST = 0.95
ID_RELINK_FEATURE_SIM = 0.55
MAX_RELINK_MISSING_SECONDS = 600
RAW_TRACK_JUMP_CENTER_DIST = 2.20
# ----------------------------------------



# ===================== ADDED =====================
# JSON_OUTPUT_PATH = OUTPUT_PATH.replace(".MP4", ".json")
# JSON_OUTPUT_PATH = os.path.splitext(OUTPUT_PATH)[0] + ".json"

json_results = []
frame_index = 0
counted_person_ids = set()
raw_track_to_person_id = {}
person_last_boxes = {}
person_last_seen = {}
next_person_id = 1
person_last_centers = {}
person_features = {}
counted_in_ids = set()
counted_out_ids = set()
in_count = 0
out_count = 0
recent_events = []
last_progress_update = 0
# =================================================

# --- Load YOLO model ---
model = YOLO(MODEL_PATH, task="detect")

# --- Open video ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_length_sec = frame_count / fps if fps > 0 else 0

print(f"🎬 Video Info: {frame_count} frames, {fps:.2f} FPS, length {video_length_sec:.2f} sec, "
      f"resolution {width}x{height}")

# --- Resize settings ---
if width > MAX_WIDTH:
    scale = MAX_WIDTH / width
    out_width = int(width * scale)
    out_height = int(height * scale)
else:
    out_width, out_height = width, height

# --- Video writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (out_width, out_height))

# --- Queues ---
decode_queue = queue.Queue(maxsize=50)
infer_queue = queue.Queue(maxsize=50)

# --- Benchmark lists ---
decode_times = []
infer_times = []
memory_usage = []

# --- Helper function: memory ---
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return mem.rss / (1024**2)  # MB


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


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
    elapsed_progress = max(now - start_time, 1e-6)
    processing_fps = processed / elapsed_progress
    remaining = max(total - processed, 0)
    eta = remaining / processing_fps if processing_fps > 0 else 0

    message = (
        f"\rProcessing [{bar}] {processed}/{total} "
        f"{percent * 100:6.2f}% | FPS {processing_fps:5.2f} | "
        f"ETA {format_duration(eta)} | IN {in_count} OUT {out_count} "
        f"OCC {max(in_count - out_count, 0)}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0


def center_distance_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2
    distance = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    avg_width = ((ax2 - ax1) + (bx2 - bx1)) / 2
    avg_height = ((ay2 - ay1) + (by2 - by1)) / 2
    scale = max(avg_width, avg_height, 1)
    return distance / scale


def normalized_bbox(bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (x1 / w, y1 / h, x2 / w, y2 / h)


def bbox_center_and_area_ratios(bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = normalized_bbox(bbox, frame_shape)
    box_w = max(x2 - x1, 0)
    box_h = max(y2 - y1, 0)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2, y2, box_w * box_h, box_h)


def is_outside_view_detection(bbox, frame_shape):
    cx, cy, _, foot_y, area_ratio, height_ratio = bbox_center_and_area_ratios(
        bbox, frame_shape
    )
    if area_ratio < MIN_PERSON_BOX_AREA_RATIO or height_ratio < MIN_PERSON_HEIGHT_RATIO:
        return True

    for zx1, zy1, zx2, zy2 in OUTSIDE_VIEW_ZONES:
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2 and foot_y <= zy2:
            return True
    return False


def extract_person_feature(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten()


def feature_similarity(feature_a, feature_b):
    if feature_a is None or feature_b is None:
        return 0.0
    return float(cv2.compareHist(
        feature_a.astype("float32"),
        feature_b.astype("float32"),
        cv2.HISTCMP_CORREL
    ))


def remember_person_feature(person_id, feature):
    if feature is None:
        return
    previous = person_features.get(person_id)
    if previous is None:
        person_features[person_id] = feature
    else:
        person_features[person_id] = (previous * 0.75) + (feature * 0.25)


def get_corrected_person_id(raw_track_id, bbox, feature, current_frame_person_ids):
    global next_person_id

    if raw_track_id is not None and raw_track_id in raw_track_to_person_id:
        mapped_person_id = raw_track_to_person_id[raw_track_id]
        mapped_box = person_last_boxes.get(mapped_person_id)
        if mapped_box is None:
            return mapped_person_id
        if center_distance_ratio(bbox, mapped_box) <= RAW_TRACK_JUMP_CENTER_DIST:
            return mapped_person_id

    best_person_id = None
    best_score = -1
    max_missing_frames = int(max(fps, 1) * MAX_RELINK_MISSING_SECONDS)

    for person_id, last_box in person_last_boxes.items():
        if person_id in current_frame_person_ids:
            continue
        if frame_index - person_last_seen.get(person_id, frame_index) > max_missing_frames:
            continue

        iou = bbox_iou(bbox, last_box)
        dist_ratio = center_distance_ratio(bbox, last_box)
        appearance = feature_similarity(feature, person_features.get(person_id))
        if (iou >= ID_RELINK_IOU or
                dist_ratio <= ID_RELINK_CENTER_DIST or
                appearance >= ID_RELINK_FEATURE_SIM):
            score = (iou * 2.0) + (1 - min(dist_ratio, 2) / 2) + max(appearance, 0)
            if score > best_score:
                best_score = score
                best_person_id = person_id

    if best_person_id is None:
        best_person_id = next_person_id
        next_person_id += 1

    if raw_track_id is not None:
        raw_track_to_person_id[raw_track_id] = best_person_id

    return best_person_id


def draw_detection(frame, bbox, person_id, conf):
    x1, y1, x2, y2 = map(int, bbox)
    label = f"Human"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 3
    color = (255, 80, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_y1 = max(y1 - text_size[1] - baseline - 8, 0)
    label_y2 = label_y1 + text_size[1] + baseline + 8
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_size[0] + 8, label_y2), color, -1)
    cv2.putText(frame, label, (x1 + 4, label_y2 - baseline - 4),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def text_with_outline(frame, text, org, font, scale, color, thickness=2):
    cv2.putText(frame, text, org, font, scale, (0, 0, 0),
                thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def normalized_line_to_pixels(line_norm, frame_shape):
    h, w = frame_shape[:2]
    return tuple((int(x * w), int(y * h)) for x, y in line_norm)


def normalized_zones_to_pixels(zones, frame_shape):
    pixel_zones = []
    for zone in zones:
        pixel_zones.append({
            "name": zone["name"],
            "entry": normalized_line_to_pixels(zone["entry"], frame_shape),
            "exit": normalized_line_to_pixels(zone["exit"], frame_shape),
        })
    return pixel_zones


def select_counting_zones(frame_person_count):
    if LINE_LAYOUT_MODE == "single":
        return SINGLE_COUNTING_ZONES, "single"
    if LINE_LAYOUT_MODE == "double":
        return DOUBLE_COUNTING_ZONES, "double"
    if frame_person_count >= AUTO_SINGLE_LINE_MIN_PERSONS:
        return SINGLE_COUNTING_ZONES, "single"
    return DOUBLE_COUNTING_ZONES, "double"


def point_side(point, line):
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def orientation(a, b, c):
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-6:
        return 0
    return 1 if value > 0 else 2


def on_segment(a, b, c):
    return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
            min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))


def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


def crossed_line(prev_center, current_center, line):
    if prev_center is None:
        return False

    move = ((current_center[0] - prev_center[0]) ** 2 +
            (current_center[1] - prev_center[1]) ** 2) ** 0.5
    if move < LINE_CROSS_MIN_MOVE:
        return False

    prev_side = point_side(prev_center, line)
    current_side = point_side(current_center, line)
    if prev_side == 0 or current_side == 0:
        return segments_intersect(prev_center, current_center, line[0], line[1])
    return prev_side * current_side < 0 and segments_intersect(
        prev_center, current_center, line[0], line[1]
    )


def add_event(person_id, event_type):
    expiry_frame = frame_index + int(max(fps, 1) * EVENT_DISPLAY_SECONDS)
    recent_events.append({
        "person_id": person_id,
        "event_type": event_type,
        "expiry_frame": expiry_frame
    })


def draw_virtual_lines(frame, pixel_zones, layout_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for zone in pixel_zones:
        entry_line = zone["entry"]
        exit_line = zone["exit"]

        cv2.line(frame, entry_line[0], entry_line[1], (0, 255, 0), 6, cv2.LINE_AA)
        cv2.line(frame, exit_line[0], exit_line[1], (0, 0, 255), 6, cv2.LINE_AA)

        entry_label_x = (entry_line[0][0] + entry_line[1][0]) // 2
        entry_label_y = (entry_line[0][1] + entry_line[1][1]) // 2
        exit_label_x = (exit_line[0][0] + exit_line[1][0]) // 2
        exit_label_y = (exit_line[0][1] + exit_line[1][1]) // 2
        text_with_outline(
            frame,
            "ENTRY (IN)",
            (entry_label_x - 45, entry_label_y - 12),
            font,
            0.55,
            (255, 255, 255),
            2
        )
        text_with_outline(
            frame,
            "EXIT (OUT)",
            (exit_label_x - 40, exit_label_y + 26),
            font,
            0.55,
            (255, 255, 255),
            2
        )

        # Direction arrows: down across green for entry, up across red for exit.
        ex = (entry_line[0][0] + entry_line[1][0]) // 2
        ey = (entry_line[0][1] + entry_line[1][1]) // 2
        ox = (exit_line[0][0] + exit_line[1][0]) // 2
        oy = (exit_line[0][1] + exit_line[1][1]) // 2
        cv2.arrowedLine(frame, (ex, ey - 45), (ex, ey + 45), (0, 255, 0), 3,
                        cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(frame, (ox, oy + 45), (ox, oy - 45), (0, 0, 255), 3,
                        cv2.LINE_AA, tipLength=0.25)


def draw_dashboard(frame, stats):
    total = stats['occupancy'] + stats['out_count']

    lines = [
        # f"IN Count: {stats['in_count']}",
        f"OUT Count: {stats['out_count']}",
        f"Occupancy: {stats['occupancy']}",
        f"Total: {total}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    thickness = 7
    padding = 16
    line_gap = 9

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_width = max(size[0] for size in text_sizes) + padding * 2
    box_height = sum(size[1] for size in text_sizes) + padding * 2 + line_gap * (len(lines) - 1)
    x1 = max(frame.shape[1] - box_width - 20, 0)
    y1 = 20
    x2 = frame.shape[1] - 20
    y2 = y1 + box_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), 2)

    text_y = y1 + padding + text_sizes[0][1]
    for line, size in zip(lines, text_sizes):
        cv2.putText(frame, line, (x2 - padding - size[0], text_y),
                    font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)
        text_y += size[1] + line_gap

    return frame


def draw_recent_events(frame):
    active_events = [event for event in recent_events if event["expiry_frame"] >= frame_index]
    recent_events[:] = active_events[-3:]
    if not recent_events:
        return

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # y = 70
    # for event in recent_events:
    #     if event["event_type"] == "IN":
    #         text = f"Person {event['person_id']} ENTERED"
    #         color = (0, 255, 0)
    #     else:
    #         text = f"Person {event['person_id']} EXITED"
    #         color = (0, 0, 255)
    #     text_with_outline(frame, text, (30, y), font, 1.1, color, 3)
    #     y += 45

# --- Thread: Decode ---
def decode_frames():
    while True:
        start_t = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        decode_queue.put(frame)
        decode_times.append(time.time() - start_t)
    decode_queue.put(None)  # Signal end

# --- Thread: Inference ---
def run_inference():
    global frame_index, in_count, out_count
    while True:
        frame = decode_queue.get()
        if frame is None:
            break
        start_t = time.time()
        # YOLO tracking with BoT-SORT keeps persistent person IDs across frames.
        # results = model.predict(frame, conf=CONFIDENCE, device="cpu")[0]
        results = model.track(
            frame,
            classes=[0],
            conf=CONFIDENCE,
            device=0,
            persist=True,
            tracker="botsort.yaml",
            verbose=False
        )[0]  # 0 = first GPU

        # OPTIONAL: Disable segmentation masks if model is a -seg model
        if hasattr(results, "masks") and results.masks is not None:
            results.masks = None   # turn off masks

# ===================== ADDED =====================
# -------- JSON FRAME DATA EXTRACTION -------------
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
            "detections": [],
            "ignored_detections": []
        }

        person_candidates = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                bbox = list(map(float, box.xyxy[0]))
                x1, y1, x2, y2 = bbox
                track_id = int(box.id[0]) if box.id is not None else None
                foot_center = ((x1 + x2) / 2, y2)

                candidate = {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "raw_track_id": track_id,
                    "bbox": bbox,
                    "foot_center": foot_center,
                    "confidence": conf,
                }
                if is_outside_view_detection(bbox, frame.shape):
                    frame_data["ignored_detections"].append({
                        "class_id": cls_id,
                        "class_name": class_name,
                        "raw_track_id": track_id,
                        "foot_center": [foot_center[0], foot_center[1]],
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "reason": "outside_bus_view"
                    })
                    continue
                person_candidates.append(candidate)

        detected_box_count = len(person_candidates)
        active_zones, line_layout = select_counting_zones(detected_box_count)
        pixel_zones = normalized_zones_to_pixels(active_zones, frame.shape)
        frame_data["line_layout"] = line_layout
        frame_data["counting_zones"] = active_zones
        current_person_ids = set()
        current_frame_detection_count = 0
        for candidate in person_candidates:
            cls_id = candidate["class_id"]
            class_name = candidate["class_name"]
            conf = candidate["confidence"]
            bbox = candidate["bbox"]
            x1, y1, x2, y2 = bbox
            track_id = candidate["raw_track_id"]
            foot_center = candidate["foot_center"]
            feature = extract_person_feature(frame, bbox)

            if track_id is not None:
                person_id = get_corrected_person_id(
                    track_id,
                    bbox,
                    feature,
                    current_person_ids
                )
                if person_id in current_person_ids:
                    continue
                is_new_person = person_id not in counted_person_ids
                current_person_ids.add(person_id)
                counted_person_ids.add(person_id)
                person_last_boxes[person_id] = bbox
                person_last_seen[person_id] = frame_index
                remember_person_feature(person_id, feature)
            else:
                person_id = None
                is_new_person = False
            current_frame_detection_count += 1

            event_type = None
            event_zone = None
            if person_id is not None:
                previous_center = person_last_centers.get(person_id)
                seed_initial_inside = (
                    INITIAL_VISIBLE_PERSONS_ARE_IN and
                    is_new_person and
                    frame_index <= int(max(fps, 1) * INITIAL_INSIDE_SEED_SECONDS)
                )

                if seed_initial_inside and person_id not in counted_in_ids:
                    in_count += 1
                    counted_in_ids.add(person_id)
                    event_type = "IN"
                    add_event(person_id, event_type)
                    event_zone = "initial_inside"
                else:
                    for zone in pixel_zones:
                        if (person_id not in counted_in_ids and
                                crossed_line(previous_center, foot_center, zone["entry"])):
                            in_count += 1
                            counted_in_ids.add(person_id)
                            event_type = "IN"
                            event_zone = zone["name"]
                            add_event(person_id, event_type)
                            break
                        if (person_id not in counted_out_ids and
                                crossed_line(previous_center, foot_center, zone["exit"])):
                            out_count += 1
                            counted_out_ids.add(person_id)
                            event_type = "OUT"
                            event_zone = zone["name"]
                            add_event(person_id, event_type)
                            break
                person_last_centers[person_id] = foot_center

            if event_type is not None:
                frame_data["events"].append({
                    "person_id": person_id,
                    "event_type": event_type,
                    "zone": event_zone
                })

            frame_data["detections"].append({
                "class_id": cls_id,
                "class_name": class_name,
                "raw_track_id": track_id,
                "person_id": person_id,
                "foot_center": [foot_center[0], foot_center[1]],
                "confidence": round(conf, 4),
                "bbox_xyxy": [x1, y1, x2, y2]
            })

        frame_person_count = len(current_person_ids) if current_person_ids else current_frame_detection_count
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
# =================================================

        infer_times.append(time.time() - start_t)
        memory_usage.append(get_memory_usage())
        
        # Annotate frame with corrected person IDs, not raw tracker IDs.
        annotated_frame = frame.copy()
        for detection in frame_data["detections"]:
            if detection["person_id"] is not None:
                draw_detection(
                    annotated_frame,
                    detection["bbox_xyxy"],
                    detection["person_id"],
                    detection["confidence"]
                )
        draw_virtual_lines(annotated_frame, pixel_zones, line_layout)
        draw_recent_events(annotated_frame)
        if width > MAX_WIDTH:
            annotated_frame = cv2.resize(annotated_frame, (out_width, out_height))
            print(out_width,out_height)
        elapsed_frame_time = max(time.time() - start_t, 1e-6)
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
            "fps": 1 / elapsed_frame_time
        }
        annotated_frame = draw_dashboard(annotated_frame, stats)
        infer_queue.put(annotated_frame)
        update_progress_bar()
    infer_queue.put(None)

# --- Thread: Write ---
def write_frames():
    while True:
        annotated_frame = infer_queue.get()
        if annotated_frame is None:
            break
        out.write(annotated_frame)

# --- Run threads ---
start_time = time.time()
t1 = threading.Thread(target=decode_frames)
t2 = threading.Thread(target=run_inference)
t3 = threading.Thread(target=write_frames)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

update_progress_bar(force=True)
print()

cap.release()
out.release()

# cv2.destroyAllWindows()

if not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) < 1024:
    raise RuntimeError("❌ Output video corrupted or empty")

elapsed = time.time() - start_time

# # ===================== ADDED =====================
# -------- SAVE JSON OUTPUT -----------------------
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
        "tracker": "botsort.yaml",
        "unique_person_count": len(counted_person_ids),
        "line_layout_mode": LINE_LAYOUT_MODE,
        "auto_single_line_min_persons": AUTO_SINGLE_LINE_MIN_PERSONS,
        "initial_visible_persons_are_in": INITIAL_VISIBLE_PERSONS_ARE_IN,
        "initial_inside_seed_seconds": INITIAL_INSIDE_SEED_SECONDS,
        "single_counting_zones": SINGLE_COUNTING_ZONES,
        "double_counting_zones": DOUBLE_COUNTING_ZONES,
        "in_count": in_count,
        "out_count": out_count,
        "occupancy": max(in_count - out_count, 0),
        "outside_view_zones": OUTSIDE_VIEW_ZONES,
        "min_person_box_area_ratio": MIN_PERSON_BOX_AREA_RATIO,
        "min_person_height_ratio": MIN_PERSON_HEIGHT_RATIO,
        "id_relink_iou": ID_RELINK_IOU,
        "id_relink_center_dist": ID_RELINK_CENTER_DIST,
        "id_relink_feature_sim": ID_RELINK_FEATURE_SIM,
        "max_relink_missing_seconds": MAX_RELINK_MISSING_SECONDS,
        "raw_track_jump_center_dist": RAW_TRACK_JUMP_CENTER_DIST,
        "device": str(model.device)
    },
    "frames": json_results
}

with open(JSON_OUTPUT_PATH, "w") as f:
    json.dump(final_json, f, indent=4)

print(f"📄 JSON saved to: {os.path.abspath(JSON_OUTPUT_PATH)}")
# =================================================

# --- Benchmark calculations ---
# Latency
latency_ms = [t*1000 for t in infer_times]
p50 = np.percentile(latency_ms, 50)
p95 = np.percentile(latency_ms, 95)
avg_latency = np.mean(latency_ms)

# FPS
avg_fps = frame_count / elapsed
fps_per_frame = [1/t for t in infer_times if t > 0]

# RAM
avg_mem = np.mean(memory_usage) if memory_usage else 0
max_mem = np.max(memory_usage) if memory_usage else 0

# Stability (last 30 frames)
fps_window = 30
fps_sliding = [1/t for t in infer_times[-fps_window:]] if len(infer_times) >= fps_window else fps_per_frame

# --- Print report ---
print("\n===== MODEL BENCHMARK REPORT =====")
print(f"Frames processed: {frame_count}")
print(f"length {video_length_sec:.2f} sec")
print(f"Total processing time (s): {elapsed:.2f}")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Median FPS per frame: {np.median(fps_per_frame):.2f}")
print(f"Latency (ms) - avg: {avg_latency:.2f}, p50: {p50:.2f}, p95: {p95:.2f}")
print(f"RAM usage (MB) - avg: {avg_mem:.2f}, max: {max_mem:.2f}")
print(f"FPS stability (last {fps_window} frames) - min: {np.min(fps_sliding):.2f}, max: {np.max(fps_sliding):.2f}")
print(f"Unique persons counted by corrected persistent ID: {len(counted_person_ids)}")
print(f"IN Count: {in_count}")
print(f"OUT Count: {out_count}")
print(f"Final occupancy: {max(in_count - out_count, 0)}")
print(f"✅ Saved threaded annotated video to: {os.path.abspath(OUTPUT_PATH)}")
print("=================================")
