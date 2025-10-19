# detector.py
import cv2
from ultralytics import YOLO
import math
import os
import base64

# --- Configuration Parameters from interface.py ---
YOLO_MODEL_CONF_THRESHOLD = 0.04
YOLO_MODEL_IOU_THRESHOLD = 0.5
CODE_DETECTION_CONF_THRESHOLD = 0.25
TRACKING_IOU_THRESHOLD = 0.03
TRACKING_FRAMES_TO_LOOK_BACK = 5
MIN_BBOX_GEOMETRIC_MEAN = 10
N_SAME_FOR_SWARM = 4
N_BULK_FOR_SWARM = 8 # New parameter for bulk re-labeling
SWARM_CONF_THRESHOLD = 0.40

# Reference values for distance estimation
SEAGULL_REF_BBOX_GEOMETRIC_MEAN_PIXELS = 42
SEAGULL_REF_DISTANCE_M = 40.0

# --- Model Initialization ---
def load_model_and_wingspans(model_path, wingspans_file):
    """Loads the YOLO model and wingspan data."""
    model = None
    average_wingspans_m = {}

    # Load Model
    try:
        print(f"Loading YOLO model from {model_path}...")
        # This import is here to avoid a circular dependency if it were at the top level
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(model_path).to(device)
        print(f"Model loaded successfully on '{device}'.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")

    # Load Wingspans
    if os.path.exists(wingspans_file):
        with open(wingspans_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    bird_name, wingspan_str = line.split(':', 1)
                    try:
                        wingspan_cm = float(wingspan_str.strip())
                        average_wingspans_m[bird_name.strip().lower()] = wingspan_cm / 100.0
                    except ValueError:
                        print(f"Could not parse wingspan for {bird_name}: {wingspan_str}")
        print(f"Loaded and converted wingspans (in meters): {average_wingspans_m}")
    else:
        print(f"Wingspans file not found at {wingspans_file}. Distance estimation will be disabled.")

    return model, average_wingspans_m

def get_video_total_frames(video_path):
    """Gets the total number of frames in a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file at {video_path}")
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Warning: Video file at {video_path} has no frames.")
            return None
        return total_frames
    except Exception as e:
        print(f"Error getting total frames for {video_path}: {e}")
        return None
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

def run_detection_on_frame(frame, model):
    """Runs bird detection and processes results."""
    if model is None:
        return []
    try:
        results = model(frame, conf=YOLO_MODEL_CONF_THRESHOLD, iou=YOLO_MODEL_IOU_THRESHOLD)
        raw_detections = []
        for r in results:
            for box in r.boxes:
                raw_detections.append({
                    'bbox': box.xyxy[0].tolist(),
                    'class': model.names[int(box.cls)].lower(),
                    'confidence': float(box.conf),
                    'bbox_width_pixels': float(box.xywh[0][2]),
                    'bbox_height_pixels': float(box.xywh[0][3])
                })
        return raw_detections
    except Exception as e:
        print(f"Error during raw detection on frame: {e}")
        return []

def annotate_frame(frame, detections_for_annotation):
    """Overlays annotations onto a video frame."""
    annotated_frame = frame.copy()
    for detection in detections_for_annotation:
        bbox = detection['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        label_lines = []
        if detection.get('tracked_id') is not None:
            label_lines.append(f"ID {detection['tracked_id']}: {detection['class']} ({detection['confidence']:.2f})")
        else:
            label_lines.append(f"{detection['class']} ({detection['confidence']:.2f})")

        if detection.get('distance_m') is not None:
            label_lines.append(f"Dist: {detection['distance_m']:.2f}m")

        if detection.get('visibility_count') is not None:
            label_lines.append(f"Visible: {detection['visibility_count']}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        line_height = cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] + 5

        text_y_start = y1 - (line_height * len(label_lines)) - 5
        if text_y_start < 0:
            text_y_start = y2 + 5

        for i, line in enumerate(label_lines):
            text_y = text_y_start + i * line_height
            cv2.putText(annotated_frame, line, (x1, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return annotated_frame

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x_inter1 = max(box1[0], box2[0])
    y_inter1 = max(box1[1], box2[1])
    x_inter2 = min(box1[2], box2[2])
    y_inter2 = min(box1[3], box2[3])
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = float(box1_area + box2_area - inter_area)
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def process_video_frame_with_tracking(video_path, frame_index, model, average_wingspans_m, tracking_state):
    """
    Main processing function that reads a frame, detects, tracks, estimates distance,
    and annotates.
    """
    if model is None:
        return None, [], tracking_state, "Model is not loaded."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, [], tracking_state, "Could not open video file."

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, [], tracking_state, f"Could not read frame at index {frame_index}."

    # --- Raw Detection ---
    raw_detections_from_model = run_detection_on_frame(frame, model)

    # --- Base Filtering ---
    # Start with all detections that meet the basic confidence threshold.
    # This ensures we don't lose valid detections due to swarm logic.
    base_detections = [det for det in raw_detections_from_model if det['confidence'] >= CODE_DETECTION_CONF_THRESHOLD]

    # --- Conditional Filtering (Swarm Logic) ---
    core_detections = []
    swarm_candidates = []
    core_detection_counts_by_type = {}

    for det in raw_detections_from_model:
        bbox_width = det['bbox_width_pixels']
        bbox_height = det['bbox_height_pixels']
        geometric_mean = math.sqrt(bbox_width * bbox_height) if bbox_width > 0 and bbox_height > 0 else 0
        det['geometric_mean'] = geometric_mean
        confidence = det['confidence']

        # A detection is "core" if it's both large enough AND confident enough.
        if geometric_mean >= MIN_BBOX_GEOMETRIC_MEAN and confidence >= CODE_DETECTION_CONF_THRESHOLD:
            core_detections.append(det)
            bird_type = det['class']
            core_detection_counts_by_type[bird_type] = core_detection_counts_by_type.get(bird_type, 0) + 1
        else:
            # Otherwise, it's a candidate to be included if a swarm is detected.
            swarm_candidates.append(det)

    # --- Swarm Logic: Check for bulk swarm first, then normal swarm ---
    bulk_swarm_class = None
    # Find the most numerous core bird type
    for bird_type, count in core_detection_counts_by_type.items():
        # If a class meets the bulk threshold AND it's not 'unknown_bird', it triggers the rule.
        if count >= N_BULK_FOR_SWARM and bird_type != 'unknown_bird':
            bulk_swarm_class = bird_type
            # We take the first valid class that triggers the rule.
            break

    if bulk_swarm_class:
        # Bulk Swarm Rule: Re-label ALL original detections to the dominant class.
        # This is an aggressive override.
        for det in raw_detections_from_model:
            det['class'] = bulk_swarm_class
        detections_for_tracking = raw_detections_from_model
    else:
        # If no bulk swarm, start with our baseline good-confidence detections.
        detections_for_tracking = base_detections.copy()
        # Then, check for a normal swarm to ADD more detections.
        normal_swarm_class = None
        for bird_type, count in core_detection_counts_by_type.items():
            if count >= N_SAME_FOR_SWARM:
                normal_swarm_class = bird_type
                break

        if normal_swarm_class:
            # If a swarm is detected, add the low-confidence candidates back in,
            # but re-label them to match the swarm.
            for candidate in swarm_candidates:
                candidate['class'] = normal_swarm_class
            detections_for_tracking.extend(swarm_candidates)

    # The list `detections_for_tracking` now contains all the detections we want to process.

    # --- Object Tracking Logic ---
    current_frame_tracking_updates = {}
    matched_detections_indices = set()

    for i, current_det in enumerate(detections_for_tracking):
        best_match_id = None
        best_match_iou = 0
        for tracked_id, tracked_info in tracking_state.items():
            if (tracked_info.get('class') == current_det['class'] and
                    tracked_info.get('last_visible_frame', -TRACKING_FRAMES_TO_LOOK_BACK - 1) >= frame_index - TRACKING_FRAMES_TO_LOOK_BACK):
                iou = calculate_iou(current_det['bbox'], tracked_info['bbox'])
                if iou > best_match_iou and iou > TRACKING_IOU_THRESHOLD:
                    best_match_iou = iou
                    best_match_id = tracked_id

        if best_match_id is not None:
            matched_detections_indices.add(i)
            # It's important to copy the tracked info to avoid modifying the state before all checks are done.
            updated_info = tracking_state[best_match_id].copy()

            # Check for visibility count before updating last_visible_frame
            if updated_info.get('last_visible_frame', -2) == frame_index - 1:
                updated_info['visibility_count'] = updated_info.get('visibility_count', 0) + 1
            else:
                updated_info['visibility_count'] = 1

            # Now, update the rest of the information for the tracked object
            updated_info.update({
                'bbox': current_det['bbox'],
                'confidence': current_det['confidence'],
                'last_visible_frame': frame_index,
                'is_visible': True,
                'max_bbox_geometric_mean': max(
                    updated_info.get('max_bbox_geometric_mean', 0.0),
                    current_det['geometric_mean']
                )
            })

            if 'bbox_size_history' not in updated_info:
                updated_info['bbox_size_history'] = []
            updated_info['bbox_size_history'].append(current_det['geometric_mean'])
            tracking_state[best_match_id] = updated_info
            current_frame_tracking_updates[best_match_id] = updated_info

    # Add new detections
    all_ids = list(tracking_state.keys())
    next_id = max(all_ids + [0]) + 1 if all_ids else 1
    for i, current_det in enumerate(detections_for_tracking):
        if i not in matched_detections_indices:
            new_id = next_id
            new_info = {
                'id': new_id,
                'class': current_det['class'],
                'bbox': current_det['bbox'],
                'confidence': current_det['confidence'],
                'bbox_size_history': [current_det['geometric_mean']],
                'visibility_count': 1,
                'last_visible_frame': frame_index,
                'max_bbox_geometric_mean': current_det['geometric_mean'],
                'is_visible': True
            }
            tracking_state[new_id] = new_info
            current_frame_tracking_updates[new_id] = new_info
            next_id += 1

    # Mark unmatched as not visible and remove old ones
    ids_to_remove = []
    for tracked_id, tracked_info in tracking_state.items():
        if tracked_id not in current_frame_tracking_updates:
            tracked_info['is_visible'] = False
            if frame_index - tracked_info.get('last_visible_frame', 0) > TRACKING_FRAMES_TO_LOOK_BACK:
                ids_to_remove.append(tracked_id)

    for tracked_id in ids_to_remove:
        if tracked_id in tracking_state:
            del tracking_state[tracked_id]

    # --- Prepare for Annotation ---
    detections_for_annotation = [
        info for info in current_frame_tracking_updates.values() if info.get('is_visible', False)
    ]

    # --- Distance Estimation ---
    perform_distance_estimation = (
        average_wingspans_m and 'seagull' in average_wingspans_m and
        SEAGULL_REF_BBOX_GEOMETRIC_MEAN_PIXELS > 0 and
        average_wingspans_m['seagull'] > 0
    )

    if perform_distance_estimation:
        seagull_ref_wingspan_m = average_wingspans_m['seagull']
        for det_info in detections_for_annotation:
            bird_name = det_info['class']
            if bird_name in average_wingspans_m:
                bird_real_wingspan_m = average_wingspans_m[bird_name]
                max_geometric_mean_pixels = det_info.get('max_bbox_geometric_mean', 0.0)

                if max_geometric_mean_pixels > 0 and bird_real_wingspan_m > 0:
                    estimated_distance_m = (bird_real_wingspan_m / seagull_ref_wingspan_m) * \
                                          (SEAGULL_REF_BBOX_GEOMETRIC_MEAN_PIXELS / max_geometric_mean_pixels) * \
                                          SEAGULL_REF_DISTANCE_M
                    det_info['distance_m'] = estimated_distance_m

    # --- Annotation and Encoding ---
    annotated_frame = annotate_frame(frame, detections_for_annotation)

    _, buffer = cv2.imencode('.jpg', annotated_frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')

    # Prepare detection list for frontend
    frontend_detections = []
    for det in detections_for_annotation:
        frontend_detections.append({
            "class": det.get('class'),
            "confidence": det.get('confidence'),
            "tracked_id": det.get('id'),
            "distance_m": det.get('distance_m'),
            "visibility_count": det.get('visibility_count'),
            "box": det.get('bbox')
        })

    return encoded_frame, frontend_detections, tracking_state, None

def analyze_video_chunk(video_path, start_frame, num_frames, model, average_wingspans_m, initial_tracking_state):
    """
    Analyzes a chunk of video frames and returns an aggregated report.
    This function now correctly uses the process_video_frame_with_tracking function
    to ensure swarm logic is applied.
    """
    if model is None:
        return {"error": "Model not loaded"}, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}, None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    end_frame = min(start_frame + num_frames, total_frames)
    
    # Make a deep copy to avoid modifying the main tracker state during analysis
    import copy
    local_tracking_state = copy.deepcopy(initial_tracking_state)

    aggregated_detections = {}

    for frame_index in range(start_frame, end_frame):
        _, frame_detections, local_tracking_state, error = process_video_frame_with_tracking(
            video_path, frame_index, model, average_wingspans_m, local_tracking_state
        )
        if error:
            print(f"Skipping frame {frame_index} due to error: {error}")
            continue

        for det in frame_detections:
            bird_class = det['class']
            aggregated_detections[bird_class] = aggregated_detections.get(bird_class, 0) + 1

    return aggregated_detections, local_tracking_state