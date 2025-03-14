import os
import uuid
import subprocess
import threading
import tempfile
import shutil
import time
import mediapipe as mp
import cv2
import numpy as np
import whisper
from pydub import AudioSegment
from flask import Flask, render_template, request, jsonify, send_file
from waitress import serve

# Create necessary directories
os.makedirs('uploads', exist_ok=True)

# Create processed directory if it doesn't exist
if not os.path.exists('processed'):
    os.makedirs('processed')

# Create a global temp directory
temp_dir = tempfile.mkdtemp()

# Initialize Flask app
app = Flask(__name__, 
            static_folder='src/static',
            template_folder='src/templates')

# Global variable to store processing status
processing_tasks = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('src/static', path)

@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    video_url = data.get('video_url')
    start_time = data.get('start_time', 0)
    duration = data.get('duration', 60)
    debug_frames = data.get('debug_frames', False)
    karaoke_mode = data.get('karaoke_mode', True)
    
    if not video_url:
        return jsonify({'error': 'No video URL provided'}), 400
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # Create a directory for this task
    task_dir = os.path.join(temp_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Also create a permanent directory in the processed folder
    permanent_dir = os.path.join('processed', task_id)
    os.makedirs(permanent_dir, exist_ok=True)
    
    # Start the processing task in a background thread
    thread = threading.Thread(
        target=process_video_task, 
        args=(task_id, video_url, start_time, duration, debug_frames, karaoke_mode)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(processing_tasks[task_id])

@app.route('/download/<task_id>', methods=['GET'])
def download_video(task_id):
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
        
    if processing_tasks[task_id]['status'] != 'completed':
        return jsonify({'error': 'Video processing not completed'}), 404
    
    output_path = processing_tasks[task_id].get('output_path')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': f'Video file not found at {output_path}'}), 404
    
    try:
        # Return the copied file
        return send_file(output_path, as_attachment=True, 
                        download_name=f'processed_video_{task_id}.mp4')
    except Exception as e:
        return jsonify({'error': f'Error preparing download: {str(e)}'}), 500

def process_video_task(task_id, video_url, start_time, duration, debug_frames=False, karaoke_mode=True):
    # Create a permanent directory for this task
    task_dir = os.path.join(temp_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Also create a permanent directory in the processed folder
    permanent_dir = os.path.join('processed', task_id)
    os.makedirs(permanent_dir, exist_ok=True)
    
    # Define output paths
    temp_output_file = os.path.join(task_dir, 'final.mp4')
    permanent_output_file = os.path.join(permanent_dir, 'final.mp4')
    
    try:
        # Update task status
        update_task_status(task_id, 'Downloading video...', 10)
        
        # Download video using yt-dlp
        video_path = download_youtube_video(video_url, task_dir)
        
        # Update task status
        update_task_status(task_id, 'Trimming video...', 30)
        
        # Trim video to specified start time and duration
        trimmed_path = trim_video(video_path, start_time, duration, task_dir)
        
        # Update task status
        update_task_status(task_id, 'Creating vertical crop...', 50)
        
        # Create a 9:16 vertical crop
        cropped_path = create_vertical_crop(trimmed_path, task_dir, debug_frames)
        
        # Generate subtitles
        try:
            update_task_status(task_id, 'Generating subtitles...', 70)
            subtitle_path = generate_subtitles(trimmed_path, task_dir, karaoke_mode)
        except Exception as e:
            print(f"Failed to generate subtitles: {str(e)}. The video may not have proper audio.")
            raise
        
        # Add subtitles to the video
        update_task_status(task_id, 'Adding subtitles to video...', 90)
        final_path = add_subtitles_to_video(cropped_path, subtitle_path, temp_output_file)
        
        # Log the final path for debugging
        print(f"Task {task_id} completed. Final video path: {final_path}")
        print(f"File exists: {os.path.exists(final_path)}, Size: {os.path.getsize(final_path) if os.path.exists(final_path) else 'N/A'}")
        
        # Copy the file to the permanent location
        if os.path.exists(final_path):
            import shutil
            shutil.copy2(final_path, permanent_output_file)
            print(f"Copied to permanent location: {permanent_output_file}")
            print(f"Permanent file exists: {os.path.exists(permanent_output_file)}")
        
        # Update task status with the permanent file path
        update_task_status(task_id, 'Processing complete!', 100, permanent_output_file)
        
    except Exception as e:
        # Update task status with error
        error_msg = f'Error: {str(e)}'
        print(f"Task {task_id} failed: {error_msg}")
        update_task_status(task_id, error_msg, -1)
        raise
    finally:
        # Don't clean up temporary files for now to help with debugging
        pass

def download_youtube_video(url, output_dir):
    try:
        output_path = os.path.join(output_dir, 'video.mp4')
        
        # First attempt with more options to bypass restrictions
        cmd = [
            'yt-dlp', 
            '--no-check-certificates',
            '--geo-bypass',
            '--force-ipv4',
            '--no-warnings',
            '--ignore-errors',
            '--extractor-retries', '3',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 
            '--cookies', 'src/cookies.txt',  
            '-o', output_path, 
            url
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except subprocess.CalledProcessError as e:
            print(f"First download attempt failed: {str(e)}")
            print(f"Output: {e.stdout.decode() if e.stdout else ''}")
            print(f"Error: {e.stderr.decode() if e.stderr else ''}")
        
        # Second attempt with simpler format selection
        cmd = [
            'yt-dlp', 
            '--no-check-certificates',
            '--geo-bypass',
            '-f', 'best[ext=mp4]/best', 
            '--cookies', 'src/cookies.txt',  
            '-o', output_path, 
            url
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except subprocess.CalledProcessError as e:
            print(f"Second download attempt failed: {str(e)}")
            print(f"Output: {e.stdout.decode() if e.stdout else ''}")
            print(f"Error: {e.stderr.decode() if e.stderr else ''}")
        
        # Third attempt with format 18 (360p) which is often more accessible
        cmd = [
            'yt-dlp', 
            '--no-check-certificates',
            '--geo-bypass',
            '-f', '18/best', 
            '--cookies', 'src/cookies.txt',  
            '-o', output_path, 
            url
        ]
        
        subprocess.run(cmd, check=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            raise Exception("Downloaded file is empty or does not exist")
            
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

def trim_video(video_path, start_time, duration, output_dir):
    output_path = os.path.join(output_dir, 'trimmed.mp4')
    
    # Use ffmpeg to trim the video
    cmd = [
        'ffmpeg', '-i', video_path, 
        '-ss', str(start_time), 
        '-t', str(duration), 
        '-c:v', 'libx264', '-c:a', 'aac', 
        '-strict', 'experimental', output_path
    ]
    
    subprocess.run(cmd, check=True)
    return output_path

def create_vertical_crop(video_path, output_dir, debug_frames=False):
    output_path = os.path.join(output_dir, 'processed.mp4')
    
    # Detect faces using MediaPipe
    print("Detecting faces with MediaPipe...")
    face_bbox, debug_video_path = detect_faces_mediapipe(video_path, output_dir, debug_frames)
    
    # If debug mode is on and we have a debug video, return that instead
    if debug_frames and debug_video_path:
        return debug_video_path
    
    if face_bbox:
        # If a face is detected, center the crop on the face
        x, y, w, h = face_bbox
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate face center
        face_center_x = x + w // 2
        
        # Calculate crop dimensions (9:16 aspect ratio)
        crop_width = int(video_height * (9/16))
        
        # Calculate crop position centered on face
        crop_x = max(0, min(face_center_x - crop_width // 2, video_width - crop_width))
        
        # Use ffmpeg to create a 9:16 crop centered on the detected face
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'crop={crop_width}:{video_height}:{crop_x}:0',
            '-c:a', 'copy', output_path
        ]
    else:
        # Fallback to center crop if no face is detected
        print("No face detected, using center crop...")
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', 'crop=ih*(9/16):ih:iw/2-ih*(9/16)/2:0',
            '-c:a', 'copy', output_path
        ]
    
    subprocess.run(cmd, check=True)
    return output_path

def generate_subtitles(video_path, output_dir, karaoke_mode=True):
    """
    Generate subtitles for a video using Whisper speech recognition.
    Returns the path to the generated subtitle file.
    """
    try:
        # Try to use Whisper for transcription
        print("Generating subtitles with Whisper...")
        return extract_audio_and_transcribe(video_path, output_dir, karaoke_mode)
    except Exception as e:
        print(f"Whisper transcription failed: {str(e)}")
        print("Falling back to simple subtitle generation...")
        return generate_simple_subtitles(video_path, output_dir)

def extract_audio_and_transcribe(video_path, output_dir, karaoke_mode=True):
    """
    Extract audio from video and transcribe it using Whisper.
    Returns the path to the generated ASS subtitle file.
    """
    # Extract audio from video
    audio_path = os.path.join(output_dir, 'audio.wav')
    
    # Use ffmpeg to extract audio
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        audio_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Load Whisper model (large for best accuracy)
    model = whisper.load_model("medium")
    
    # Transcribe audio
    print("Transcribing audio with Whisper...")
    result = model.transcribe(audio_path)
    
    # Get detected language
    detected_language = result.get("language", "en")
    print(f"Detected language: {detected_language}")
    
    # Generate ASS subtitle file
    ass_path = os.path.join(output_dir, 'subtitles.ass')
    
    # Convert Whisper segments to ASS format
    with open(ass_path, 'w', encoding='utf-8') as f:
        # Write ASS header
        f.write("[Script Info]\n")
        f.write("Title: Auto-generated subtitles\n")
        f.write("ScriptType: v4.00+\n")
        f.write("WrapStyle: 0\n")
        f.write("ScaledBorderAndShadow: yes\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n\n")
        
        # Write styles
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        # Default style (white text)
        f.write("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,3,2,10,10,30,1\n")
        
        # Highlighted style for karaoke (red text)
        f.write("Style: Highlight,Arial,48,&H000000FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,3,2,10,10,30,1\n\n")
        
        # Write events
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        # Write subtitle lines
        for i, segment in enumerate(result["segments"]):
            start_time = format_ass_time(segment["start"])
            end_time = format_ass_time(segment["end"])
            text = segment["text"].strip()
            
            if karaoke_mode and len(segment.get("words", [])) > 0:
                # Karaoke mode with word-level timing
                words = segment.get("words", [])
                
                # If words timing is available
                if words and all(["start" in word for word in words]):
                    full_text = ""
                    last_end = segment["start"]
                    
                    for word in words:
                        word_text = word.get("word", "").strip()
                        if not word_text:
                            continue
                            
                        word_start = word.get("start", last_end)
                        word_end = word.get("end", word_start + 0.5)
                        
                        # Add karaoke effect timing
                        word_start_time = format_ass_time(word_start)
                        word_end_time = format_ass_time(word_end)
                        
                        # Add a dialogue event for this word with highlight style
                        f.write(f"Dialogue: 0,{word_start_time},{word_end_time},Highlight,,0,0,0,,{word_text}\n")
                        
                        last_end = word_end
                    
                    # Add the full text in default style
                    f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
                else:
                    # No word-level timing, just add the full text
                    f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
            else:
                # Add the full text in default style if karaoke mode is off
                f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
            
        return ass_path

def format_ass_time(seconds):
    """Convert seconds to ASS time format (H:MM:SS.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    centiseconds = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{seconds_int:02d}.{centiseconds:02d}"

def generate_simple_subtitles(video_path, output_dir):
    """
    Generate simple subtitles with timestamps.
    Used as a fallback if Whisper transcription fails.
    """
    # Get video duration using ffprobe
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())
    
    # Create a simple SRT file with timestamps
    srt_path = os.path.join(output_dir, 'subtitles.srt')
    with open(srt_path, 'w', encoding='utf-8') as f:
        # Write SRT header
        f.write("1\n")
        f.write("00:00:00,000 --> 00:00:05,000\n")
        f.write("[Video content at 0-5 seconds]\n\n")
        
        segment_count = 2
        for start_time in range(5, int(duration), 5):
            end_time = min(start_time + 5, duration)
            
            f.write(f"{segment_count}\n")
            f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            f.write(f"[Video content at {start_time}-{end_time} seconds]\n\n")
            
            segment_count += 1
    
    return srt_path

def format_time(seconds):
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"

def add_subtitles_to_video(video_path, subtitle_path, output_path):
    """
    Add subtitles to a video. Supports both SRT and ASS subtitle formats.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Determine subtitle format based on file extension
    subtitle_ext = os.path.splitext(subtitle_path)[1].lower()
    
    # Convert backslashes to forward slashes and escape the path properly
    subtitle_path_escaped = subtitle_path.replace('\\', '/').replace(':', '\\:')
    
    # Create a temporary subtitle file with a simpler path if needed
    temp_subtitle_path = None
    if ':' in subtitle_path or '\\' in subtitle_path:
        # Create a simple temporary directory with a short path
        temp_dir = os.path.join(os.getcwd(), 'temp_subs')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy the subtitle file to the temporary directory
        temp_subtitle_path = os.path.join(temp_dir, f'subs{subtitle_ext}')
        with open(subtitle_path, 'r', encoding='utf-8') as src:
            with open(temp_subtitle_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        
        # Use the temporary subtitle path
        subtitle_path = temp_subtitle_path
        subtitle_path_escaped = subtitle_path.replace('\\', '/').replace(':', '\\:')
    
    try:
        # Try using the subtitles filter with the escaped path
        if subtitle_ext == '.ass':
            # For ASS subtitles
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-vf', f"ass='{subtitle_path_escaped}'",
                '-c:a', 'copy', output_path
            ]
        else:
            # For SRT subtitles
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-vf', f"subtitles='{subtitle_path_escaped}':force_style='FontSize=24,Alignment=2'",
                '-c:a', 'copy', output_path
            ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"First subtitle method failed: {e.stderr.decode() if e.stderr else ''}")
            print("Trying alternative approach...")
            
            # If that fails, try an alternative approach
            if subtitle_ext == '.ass':
                # Try a different method for ASS
                cmd = [
                    'ffmpeg', '-i', video_path, '-f', 'ass', '-i', subtitle_path,
                    '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text',
                    output_path
                ]
            else:
                # Try a different method for SRT
                cmd = [
                    'ffmpeg', '-i', video_path, '-f', 'srt', '-i', subtitle_path,
                    '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text',
                    output_path
                ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # If that also fails, just copy the video without subtitles
                print("Both subtitle methods failed, copying video without subtitles...")
                subprocess.run([
                    'ffmpeg', '-i', video_path, 
                    '-c:v', 'copy', '-c:a', 'copy', 
                    output_path
                ], check=True)
    
    finally:
        # Clean up temporary files
        if temp_subtitle_path and os.path.exists(temp_subtitle_path):
            try:
                os.remove(temp_subtitle_path)
                os.rmdir(os.path.dirname(temp_subtitle_path))
            except:
                pass
    
    return output_path

def detect_faces_mediapipe(video_path, output_dir, debug_frames=False):
    """
    Detect faces in a video using MediaPipe Face Detection.
    Returns:
    - The bounding box of the most frequently detected face (x, y, w, h)
    - Path to debug video if debug_frames is True, otherwise None
    """
    # Initialize MediaPipe face detector
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames (process 1 frame per second)
    sample_interval = int(fps)
    
    # Store detected faces
    all_faces = []
    
    # Setup for debug video
    debug_video_path = None
    debug_video_writer = None
    
    if debug_frames:
        debug_video_path = os.path.join(output_dir, 'debug_faces.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        debug_video_writer = cv2.VideoWriter(
            debug_video_path, fourcc, fps, (frame_width, frame_height)
        )
    
    # Process frames with MediaPipe
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if frame_idx % sample_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = face_detection.process(rgb_frame)
                
                # Create a debug frame if needed
                debug_frame = frame.copy() if debug_frames else None
                
                # Check if faces were detected
                if results.detections:
                    for detection in results.detections:
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Convert relative coordinates to absolute
                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        w = int(bbox.width * frame_width)
                        h = int(bbox.height * frame_height)
                        
                        # Store face with frame index
                        all_faces.append((x, y, w, h, frame_idx))
                        
                        # Draw detection on debug frame
                        if debug_frames:
                            # Draw bounding box
                            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # Add confidence score
                            confidence = detection.score[0]
                            cv2.putText(debug_frame, f"Face: {confidence:.2f}", 
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add frame number and progress
                if debug_frames:
                    progress = (frame_idx / total_frames) * 100
                    cv2.putText(debug_frame, f"Frame: {frame_idx}/{total_frames} ({progress:.1f}%)", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Write frame to debug video
                    debug_video_writer.write(debug_frame)
            
            # For non-sampled frames in debug mode, still add to video
            elif debug_frames:
                debug_video_writer.write(frame)
                
            frame_idx += 1
            
            # Progress indication
            if frame_idx % (sample_interval * 10) == 0:
                print(f"Processed {frame_idx} frames ({frame_idx/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    if debug_video_writer:
        debug_video_writer.release()
    
    # If no faces detected, return None
    if not all_faces:
        return None, debug_video_path
    
    # Group faces by proximity (consider faces within 20% of frame width to be the same person)
    proximity_threshold = frame_width * 0.2
    
    # Sort faces by frame index
    all_faces.sort(key=lambda f: f[4])
    
    # Group faces
    face_groups = []
    for face in all_faces:
        x, y, w, h, frame_idx = face
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Check if this face is close to any existing group
        found_group = False
        for group in face_groups:
            # Calculate average center of the group
            group_center_x = sum(gx + gw // 2 for gx, gy, gw, gh, _ in group) / len(group)
            group_center_y = sum(gy + gh // 2 for gx, gy, gw, gh, _ in group) / len(group)
            
            # Check distance
            distance = ((face_center_x - group_center_x) ** 2 + (face_center_y - group_center_y) ** 2) ** 0.5
            if distance < proximity_threshold:
                group.append(face)
                found_group = True
                break
        
        # If not close to any group, create a new group
        if not found_group:
            face_groups.append([face])
    
    # Find the group with the most faces
    if face_groups:
        largest_group = max(face_groups, key=len)
        
        # Calculate average bounding box for the largest group
        avg_x = sum(x for x, _, _, _, _ in largest_group) // len(largest_group)
        avg_y = sum(y for _, y, _, _, _ in largest_group) // len(largest_group)
        avg_w = sum(w for _, _, w, _, _ in largest_group) // len(largest_group)
        avg_h = sum(h for _, _, _, h, _ in largest_group) // len(largest_group)
        
        return (avg_x, avg_y, avg_w, avg_h), debug_video_path
    
    return None, debug_video_path

def update_task_status(task_id, message, progress, output_path=None):
    processing_tasks[task_id] = {
        'status': 'processing' if progress != 100 else 'completed',
        'progress': progress,
        'message': message,
        'output_path': output_path
    }

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
    pass
