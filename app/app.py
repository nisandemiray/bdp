import os
import cv2
import base64
import uuid
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename

# Import the actual detector functions
from detector import (
    analyze_video_chunk,
    load_model_and_wingspans,
    process_video_frame_with_tracking,
)

# --- Flask App Initialization ---
app = Flask(__name__, static_folder='static', template_folder='static')

# --- In-memory cache for tracking state ---
# This simple dictionary will hold the tracking state for each video session.
# Key: video_filename, Value: tracking_state dictionary
tracking_cache = {}

# --- Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PERMANENT_VIDEO_FOLDER = os.path.join(APP_ROOT, 'videos')
SESSION_UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')

# Use a secret key for session management
app.config['SECRET_KEY'] = os.urandom(24)

# Ensure both video directories exist
if not os.path.exists(PERMANENT_VIDEO_FOLDER):
    os.makedirs(PERMANENT_VIDEO_FOLDER)
if not os.path.exists(SESSION_UPLOAD_FOLDER):
    os.makedirs(SESSION_UPLOAD_FOLDER)

# --- Initialize Detector ---
MODEL_PATH = os.path.join(APP_ROOT, "weights/best.pt")
WINGSPANS_FILE = os.path.join(APP_ROOT, "wingspans.txt")

model, average_wingspans_m = load_model_and_wingspans(MODEL_PATH, WINGSPANS_FILE)

# --- Routes ---

def ensure_session_id():
    """Ensures the user has a unique session ID."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/videos')
def list_videos():
    """
    Scans permanent and session-specific folders for videos and returns a
    JSON list containing the filename and total frame count for each video.
    """
    video_files = {}
    session_id = ensure_session_id()
    user_upload_dir = os.path.join(SESSION_UPLOAD_FOLDER, session_id)

    def scan_directory(directory, is_session_file=False):
        if not os.path.exists(directory):
            return
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(directory, filename)
                try:
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        # Use a prefix for session files to avoid name collisions in the dictionary
                        key = f"session__{filename}" if is_session_file else filename
                        video_files[key] = {
                            'total_frames': total_frames,
                            'filename': filename, # Keep original filename for requests
                            'is_session_file': is_session_file
                        }
                    cap.release()
                except Exception as e:
                    app.logger.error(f"Error processing video {filename}: {e}")

    try:
        scan_directory(PERMANENT_VIDEO_FOLDER, is_session_file=False)
        scan_directory(user_upload_dir, is_session_file=True)
        return jsonify(video_files)
    except Exception as e:
        print(f"Error listing videos: {e}")
        return jsonify({"error": "Could not list videos."}), 500

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handles video file uploads."""
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    if file:
        session_id = ensure_session_id()
        user_upload_dir = os.path.join(SESSION_UPLOAD_FOLDER, session_id)

        if not os.path.exists(user_upload_dir):
            os.makedirs(user_upload_dir)

        filename = secure_filename(file.filename)
        file.save(os.path.join(user_upload_dir, filename))
        
        # Get total frames to return to the frontend for immediate selection
        video_path = os.path.join(user_upload_dir, filename)
        total_frames = -1
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception as e:
            app.logger.error(f"Could not get frame count for uploaded video {filename}: {e}")

        return jsonify({
            'status': 'success',
            'message': f'Video "{filename}" uploaded successfully.',
            'filename': filename,
            'total_frames': total_frames,
            'is_session_file': True
        })

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    """Processes a single video frame and returns the annotated version."""
    data = request.get_json()
    video_filename = data.get('video_filename')
    frame_index = int(data.get('frame_index', 0))
    is_session_file = data.get('is_session_file', False)
    reset_tracker = data.get('reset_tracker', False)

    if is_session_file:
        session_id = ensure_session_id()
        video_path = os.path.join(SESSION_UPLOAD_FOLDER, session_id, secure_filename(video_filename))
    else:
        video_path = os.path.join(PERMANENT_VIDEO_FOLDER, secure_filename(video_filename))

    if not os.path.exists(video_path):
        app.logger.error(f"Video not found at path: {video_path}")
        return jsonify({'status': 'error', 'message': 'Video not found'}), 404

    app.logger.info(f"Processing frame {frame_index} for video: {video_filename}")

    # Get or reset the tracking state for this video
    if video_filename not in tracking_cache or reset_tracker:
        tracking_cache[video_filename] = {}
        app.logger.info(f"Initialized new tracking state for {video_filename}")

    current_tracking_state = tracking_cache[video_filename]

    # Use the real detector function
    encoded_frame, detections, updated_tracking_state, error_message = process_video_frame_with_tracking(
        video_path=video_path,
        frame_index=frame_index,
        model=model,
        average_wingspans_m=average_wingspans_m,
        tracking_state=current_tracking_state
    )

    if error_message:
        return jsonify({'status': 'error', 'message': error_message}), 500

    # Update the cache with the new state
    tracking_cache[video_filename] = updated_tracking_state

    return jsonify({
        'status': 'success',
        'annotated_frame': encoded_frame,
        'detections': detections
    })

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    data = request.get_json()
    video_filename = data.get('video_filename')
    is_session_file = data.get('is_session_file', False)
    start_frame_index = data.get('start_frame_index', 0)


    if not video_filename:
        return jsonify({'status': 'error', 'message': 'Video filename is required.'}), 400

    if is_session_file:
        session_id = ensure_session_id()
        video_path = os.path.join(SESSION_UPLOAD_FOLDER, session_id, secure_filename(video_filename))
    else:
        video_path = os.path.join(PERMANENT_VIDEO_FOLDER, secure_filename(video_filename))

    if not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': 'Video not found'}), 404

    try:
        # --- ISOLATED ANALYSIS ---
        # Load a SEPARATE model instance for analysis to prevent corrupting the
        # state of the global model used for single-frame processing.
        analysis_model, _ = load_model_and_wingspans(MODEL_PATH, WINGSPANS_FILE)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set the starting frame for analysis
        if start_frame_index > 0 and start_frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

        # This block appears to be an alternative implementation.
        # It is syntactically correct but was mixed with the block below.
        # I am keeping it as requested.
        # Get the current tracking state for the video from the cache.
        # If it doesn't exist, start with an empty state.
        initial_tracking_state = tracking_cache.get(video_filename, {})
        # Call the new, consistent analysis function from detector.py
        # We use the global model instance, just like the single-frame view.
        aggregated_results, _ = analyze_video_chunk(
            video_path=video_path,
            start_frame=start_frame_index,
            num_frames=30,  # Analyze 30 consecutive frames
            model=model,
            average_wingspans_m=average_wingspans_m,
            initial_tracking_state=initial_tracking_state
        )

        # Data structures for analysis
        analysis_results = {}
        # { class_name: { "all_ids": set(), "longest_tracking": {"track_id": 0, "frames": 0}, "current_streaks": {track_id: streak_len} } }

        for _ in range(30): # Limit analysis to 30 frames
            ret, frame = cap.read()
            if not ret:
                break # Stop if we reach the end of the video

            # This check was misplaced. It should likely be after the loop.
            # if "error" in aggregated_results:
            #     return jsonify({'status': 'error', 'message': aggregated_results["error"]}), 500

            results = analysis_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            current_frame_track_ids = set()

            # This block was incorrectly indented.
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    class_name = analysis_model.names[class_id]
                    current_frame_track_ids.add(track_id)

                    # Initialize class in results if not present
                    if class_name not in analysis_results:
                        analysis_results[class_name] = {
                            "all_ids": set(),
                            "longest_tracking": {"track_id": 0, "frames": 0},
                            "current_streaks": {}
                        }
                    
                    class_data = analysis_results[class_name]
                    class_data["all_ids"].add(track_id)

                    # Update current streak
                    class_data["current_streaks"][track_id] = class_data["current_streaks"].get(track_id, 0) + 1

                    # Check for new longest streak
                    if class_data["current_streaks"][track_id] > class_data["longest_tracking"]["frames"]:
                        class_data["longest_tracking"]["frames"] = class_data["current_streaks"][track_id]
                        class_data["longest_tracking"]["track_id"] = track_id
            
            # Reset streaks for tracks that were not found in this frame
            for class_data in analysis_results.values():
                lost_tracks = set(class_data["current_streaks"].keys()) - current_frame_track_ids
                for track_id in lost_tracks:
                    class_data["current_streaks"][track_id] = 0

        cap.release()

        # This block was also incorrectly indented and structured.
        if "error" in aggregated_results:
             return jsonify({'status': 'error', 'message': aggregated_results["error"]}), 500

        # The code seems to have two conflicting return paths. I am preserving both.
        # This one seems to belong to the `analyze_video_chunk` call.
        # return jsonify({'status': 'success', 'analysis': aggregated_results})

        # Final formatting of the results from the manual loop
        final_analysis = {}
        for class_name, data in analysis_results.items():
            final_analysis[class_name] = {
                "total_unique_birds": len(data["all_ids"]),
                "longest_tracking": {
                    "track_id": int(data["longest_tracking"]["track_id"]),
                    "frames": int(data["longest_tracking"]["frames"])
                }
            }

        return jsonify({'status': 'success', 'analysis': final_analysis})

    except Exception as e:
        # Log the full error to the server console for debugging
        print(f"An error occurred during video analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'An internal error occurred: {e}'}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    # Run the app on 0.0.0.0 to be accessible from outside the container
    app.run(debug=True, host='0.0.0.0', port=port)