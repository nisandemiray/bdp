document.addEventListener('DOMContentLoaded', () => {
    const videoList = document.getElementById('video-list');
    const uploadForm = document.getElementById('upload-form');
    const videoFileInput = document.getElementById('video-file-input');
    const uploadStatus = document.getElementById('upload-status');
    const annotatedFrame = document.getElementById('annotated-frame');
    const frameSlider = document.getElementById('frame-slider');
    const frameInput = document.getElementById('frame-input');
    const totalFramesDisplay = document.getElementById('total-frames-display');
    const prevFrameBtn = document.getElementById('prev-frame-btn');
    const nextFrameBtn = document.getElementById('next-frame-btn');
    const playBtn = document.getElementById('play-btn');
    const pauseBtn = document.getElementById('pause-btn');
    const resetTrackerBtn = document.getElementById('reset-tracker-btn');
    const detectionsLog = document.getElementById('detections-log');
    const viewerTitle = document.getElementById('viewer-title');
    const loadingSpinner = document.getElementById('loading-spinner');
    const analyzeVideoBtn = document.getElementById('analyze-video-btn');
    const analysisResults = document.getElementById('analysis-results');
    const analysisSpinner = document.getElementById('analysis-spinner');

    let currentVideo = null;
    let totalFrames = 0;
    let currentFrameIndex = 0;
    let isProcessing = false;
    let isPlaying = false;
    let playbackIntervalId = null;
    let isCurrentVideoSessionFile = false; // Track if the current video is a session file

    // --- Video List Management ---

    async function fetchVideos() {
        try {
            const response = await fetch('/videos');
             // The response is a dictionary where keys can be 'session__filename.mp4'
            if (!response.ok) throw new Error('Failed to fetch videos');
            const videos = await response.json();
            
            videoList.innerHTML = '';
            if (Object.keys(videos).length === 0) {
                videoList.innerHTML = '<li>No videos found.</li>';
                return;
            }

            // The key is the unique identifier (e.g., 'session__video.mp4'), the value is the video object
            for (const key in videos) {
                const video = videos[key];
                const li = document.createElement('li');
                li.textContent = `${video.filename} (${video.is_session_file ? 'session' : 'permanent'})`;
                li.dataset.filename = video.filename; // Use the actual filename
                li.dataset.totalFrames = video.total_frames;
                li.dataset.isSessionFile = video.is_session_file;
                li.addEventListener('click', () => selectVideo(video.filename, video.total_frames, video.is_session_file));
                videoList.appendChild(li);
            }
        } catch (error) {
            console.error('Error fetching videos:', error);
            videoList.innerHTML = '<li>Error loading videos.</li>';
        }
    }

    function selectVideo(filename, frames, isSessionFile) {
        currentVideo = filename;
        totalFrames = frames;
        isCurrentVideoSessionFile = isSessionFile;

        document.querySelectorAll('#video-list li').forEach(li => {
            li.classList.toggle('active', li.dataset.filename === filename && (li.dataset.isSessionFile === String(isSessionFile)));
        });

        viewerTitle.textContent = `Viewing: ${filename}`;
        frameSlider.max = totalFrames - 1;
        frameSlider.value = 0;
        frameSlider.disabled = false;
        frameInput.disabled = false;
        frameInput.max = totalFrames;
        prevFrameBtn.disabled = false;
        nextFrameBtn.disabled = false;
        playBtn.disabled = false;
        resetTrackerBtn.disabled = false;
        analyzeVideoBtn.disabled = false;

        currentFrameIndex = 0;
        processFrame(currentFrameIndex, true); // Process first frame and reset tracker
    }

    // --- Frame Processing ---

    async function processFrame(frameIndex, resetTracker = false) {
        if (!currentVideo || isProcessing) return;

        isProcessing = true;
        currentFrameIndex = frameIndex;
        loadingSpinner.style.display = 'block';
        annotatedFrame.style.opacity = 0.5;

        try {
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_filename: currentVideo,
                    frame_index: frameIndex,
                    reset_tracker: resetTracker,
                    is_session_file: isCurrentVideoSessionFile // Send the session flag
                })
            });

            if (!response.ok) {
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.indexOf("application/json") !== -1) {
                    const errorData = await response.json();
                    throw new Error(errorData.message || 'Processing failed');
                }
                throw new Error(`Server returned an error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            if (data.status === 'success') {
                annotatedFrame.src = `data:image/jpeg;base64,${data.annotated_frame}`;
                updateDetections(data.detections);
                updateControls(frameIndex);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error processing frame:', error);
            detectionsLog.innerHTML = `<p style="color: var(--error-color);">Error: ${error.message}</p>`;
        } finally {
            isProcessing = false;
            loadingSpinner.style.display = 'none';
            annotatedFrame.style.opacity = 1;
        }
    }

    function updateControls(frameIndex) {
        frameSlider.value = frameIndex;
        frameInput.value = frameIndex + 1;
        totalFramesDisplay.textContent = `/ ${totalFrames}`;
        prevFrameBtn.disabled = frameIndex <= 0;
        nextFrameBtn.disabled = frameIndex >= totalFrames - 1;
        playBtn.disabled = frameIndex >= totalFrames - 1;
        if (isPlaying && frameIndex >= totalFrames - 1) {
            pause();
        }
    }

    function updateDetections(detections) {
        if (!detections || detections.length === 0) {
            detectionsLog.innerHTML = '<p>No detections in this frame.</p>';
            return;
        }

        let html = '<ul>';
        detections.forEach(det => {
            html += `<li>ID ${det.tracked_id}: ${det.class} - Distance: ${det.distance_m.toFixed(2)}m</li>`;
        });
        html += '</ul>';
        detectionsLog.innerHTML = html;
    }

    // --- Video Upload ---

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (videoFileInput.files.length === 0) {
            showUploadStatus('Please select a file.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('video', videoFileInput.files[0]);

        showUploadStatus('Uploading...', 'info');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                showUploadStatus(result.message, 'success');
                uploadForm.reset();
                fetchVideos(); // Refresh the list to show the new video
                selectVideo(result.filename, result.total_frames, result.is_session_file);
            } else {
                throw new Error(result.message || 'Upload failed. The server responded, but the operation was not successful.');
            }
        } catch (error) {
            showUploadStatus(`Error: ${error.message}`, 'error');
        }
    });

    function showUploadStatus(message, type) {
        uploadStatus.textContent = message;
        uploadStatus.className = type;
        setTimeout(() => {
            uploadStatus.textContent = '';
            uploadStatus.className = '';
        }, 5000);
    }

    // --- Event Listeners for Controls ---

    function play() {
        if (isPlaying) return;
        isPlaying = true;
        playBtn.style.display = 'none';
        pauseBtn.style.display = 'inline-block';
        pauseBtn.disabled = false;

        playbackIntervalId = setInterval(() => {
            if (currentFrameIndex < totalFrames - 1) {
                processFrame(currentFrameIndex + 1);
            } else {
                pause();
            }
        }, 200); // 5 FPS, adjust as needed
    }

    function pause() {
        if (!isPlaying) return;
        isPlaying = false;
        clearInterval(playbackIntervalId);
        playbackIntervalId = null;
        playBtn.style.display = 'inline-block';
        pauseBtn.style.display = 'none';
    }

    frameSlider.addEventListener('input', (e) => {
        const frameIndex = parseInt(e.target.value, 10);
        updateControls(frameIndex);
        frameInput.value = frameIndex + 1;
    });

    frameSlider.addEventListener('change', (e) => {
        const frameIndex = parseInt(e.target.value, 10);
        pause();
        processFrame(frameIndex);
    });

    frameInput.addEventListener('change', (e) => {
        let frameIndex = parseInt(e.target.value, 10) - 1;
        if (isNaN(frameIndex)) return;

        // Clamp the value
        if (frameIndex < 0) frameIndex = 0;
        if (frameIndex >= totalFrames) frameIndex = totalFrames - 1;

        pause();
        processFrame(frameIndex);
    });

    prevFrameBtn.addEventListener('click', () => {
        if (currentFrameIndex > 0) {
            pause();
            processFrame(currentFrameIndex - 1);
        }
    });

    nextFrameBtn.addEventListener('click', () => {
        if (currentFrameIndex < totalFrames - 1) {
            pause();
            processFrame(currentFrameIndex + 1);
        }
    });

    playBtn.addEventListener('click', play);
    pauseBtn.addEventListener('click', pause);

    resetTrackerBtn.addEventListener('click', () => {
        if (currentVideo) {
            pause();
            processFrame(currentFrameIndex, true); // Process current frame with reset
        }
    });

    // --- Video Analysis ---

    analyzeVideoBtn.addEventListener('click', async () => {
        if (!currentVideo) return;

        analysisSpinner.style.display = 'block';
        analyzeVideoBtn.disabled = true;
        analysisResults.innerHTML = '<p>Analyzing video... This may take a while.</p>';

        try {
            const response = await fetch('/analyze_video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_filename: currentVideo,
                    // Also tell the analyzer if it's a session file
                    is_session_file: isCurrentVideoSessionFile,
                    start_frame_index: currentFrameIndex // Send the starting frame
                })
            });

            if (!response.ok) {
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.indexOf("application/json") !== -1) {
                    const errorData = await response.json();
                    throw new Error(errorData.message || 'Analysis failed');
                }
                throw new Error(`Server returned an error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            if (data.status === 'success') {
                displayAnalysisResults(data.analysis);
            } else {
                throw new Error(data.message);
            }

        } catch (error) {
            analysisResults.innerHTML = `<p style="color: var(--error-color);">Error: ${error.message}</p>`;
        } finally {
            analysisSpinner.style.display = 'none';
            analyzeVideoBtn.disabled = false;
        }
    });

    function displayAnalysisResults(analysisData) {
        if (!analysisData || Object.keys(analysisData).length === 0) {
            analysisResults.innerHTML = '<p>Analysis complete. No objects were tracked.</p>';
            return;
        }

        let tableHTML = `
            <table>
                <thead>
                    <tr>
                        <th>Bird Class</th>
                        <th>Total Unique Birds</th>
                        <th>Longest Track (ID)</th>
                        <th>Longest Track (Frames)</th>
                    </tr>
                </thead>
                <tbody>
        `;

        for (const birdClass in analysisData) {
            const data = analysisData[birdClass];
            tableHTML += `
                <tr>
                    <td>${birdClass.charAt(0).toUpperCase() + birdClass.slice(1)}</td>
                    <td>${data.total_unique_birds}</td>
                    <td>${data.longest_tracking.track_id}</td>
                    <td>${data.longest_tracking.frames}</td>
                </tr>`;
        }

        tableHTML += '</tbody></table>';
        analysisResults.innerHTML = tableHTML;
    }

    // --- Initial Load ---
    fetchVideos();
});