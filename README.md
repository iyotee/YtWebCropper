# YouTube Video Processor

A web application that allows users to process YouTube videos with face tracking, 9:16 aspect ratio cropping, and automatic subtitle generation.

## Features

- Input YouTube video URL with customizable start time and duration
- Automatic video trimming based on specified parameters
- Face detection and tracking for intelligent 9:16 aspect ratio cropping using OpenCV
- Simple subtitle generation with timestamps
- Clean and intuitive user interface

## Requirements

- Python 3.8+
- FFmpeg installed on your system
- Various Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/youtube-video-processor.git
cd youtube-video-processor
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Install FFmpeg:
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

4. Update mediapipe once more.
```
pip install --upgrade mediapipe
```

## Usage

1. Start the application:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter a YouTube URL, optional start time, and duration (default is 60 seconds)

4. Click "Process Video" and wait for the processing to complete

5. Download your processed video when ready

## How It Works

1. **Video Download**: The application uses yt-dlp to download the YouTube video.

2. **Video Trimming**: FFmpeg is used to trim the video to the specified start time and duration.

3. **Face Detection**: OpenCV's Haar Cascade classifier is used to detect faces in each frame of the video.

4. **Intelligent Cropping**:
   - If one face is detected, the frame is cropped centered on the face with a 9:16 aspect ratio.
   - If no faces are detected, the crop is centered in the middle of the frame.
   - If multiple faces are detected, the crop is centered in the middle of the frame (future versions may use more advanced methods to identify the main speaker).

5. **Subtitle Generation**: A simple timestamp-based subtitle generation is implemented. In a production environment, you would integrate with a proper speech-to-text service.

6. **Final Output**: The cropped video frames are combined with the original audio and the generated subtitles to create the final video.

## Troubleshooting

- **FFmpeg Errors**: Ensure FFmpeg is properly installed and accessible in your PATH.
- **OpenCV Installation Issues**: If you encounter issues with OpenCV, try installing an older version or check system dependencies.
- **Memory Issues**: Processing large videos may require significant memory. Consider reducing the video duration for very large videos.

## License

MIT
