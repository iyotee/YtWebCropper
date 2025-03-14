document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const videoForm = document.getElementById('video-form');
    const inputSection = document.querySelector('.input-section');
    const progressSection = document.querySelector('.progress-section');
    const resultSection = document.querySelector('.result-section');
    const errorSection = document.querySelector('.error-section');
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    const statusMessage = document.getElementById('status-message');
    const downloadBtn = document.getElementById('download-btn');
    const processNewBtn = document.getElementById('process-new-btn');
    const tryAgainBtn = document.getElementById('try-again-btn');
    const errorMessage = document.getElementById('error-message');
    const debugFramesCheckbox = document.getElementById('debug-frames');
    const karaokeModeCheckbox = document.getElementById('karaoke-mode');

    // Current task ID
    let currentTaskId = null;
    let statusCheckInterval = null;

    // Form submission
    videoForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show progress section
        inputSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        errorSection.style.display = 'none';
        
        // Get form values
        const videoUrl = document.getElementById('video-url').value;
        const startTime = parseInt(document.getElementById('start-time').value) || 0;
        const duration = parseInt(document.getElementById('duration').value) || 60;
        const debugFrames = debugFramesCheckbox.checked;
        const karaokeModeEnabled = karaokeModeCheckbox.checked;
        
        // Create request data
        const requestData = {
            video_url: videoUrl,
            start_time: startTime,
            duration: duration,
            debug_frames: debugFrames,
            karaoke_mode: karaokeModeEnabled
        };
        
        try {
            // Send request to process video
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                currentTaskId = data.task_id;
                // Start checking status
                statusCheckInterval = setInterval(checkStatus, 2000);
            } else {
                showError(data.error || 'Failed to start processing');
            }
        } catch (error) {
            showError('Network error. Please try again.');
        }
    });
    
    // Check processing status
    async function checkStatus() {
        if (!currentTaskId) return;
        
        try {
            const response = await fetch(`/status/${currentTaskId}`);
            const data = await response.json();
            
            if (response.ok) {
                updateProgress(data);
                
                if (data.status === 'completed') {
                    clearInterval(statusCheckInterval);
                    showResult();
                } else if (data.status === 'error') {
                    clearInterval(statusCheckInterval);
                    showError(data.message || 'An error occurred during processing');
                }
            } else {
                showError('Failed to get status update');
                clearInterval(statusCheckInterval);
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }
    
    // Update progress UI
    function updateProgress(data) {
        // Update progress bar
        const progress = data.progress || 0;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;
        
        // Update status message
        statusMessage.textContent = data.message || 'Processing...';
    }
    
    // Show result section
    function showResult() {
        progressSection.style.display = 'none';
        resultSection.style.display = 'block';
        
        // Set download link
        downloadBtn.href = `/download/${currentTaskId}`;
    }
    
    // Show error section
    function showError(message) {
        progressSection.style.display = 'none';
        errorSection.style.display = 'block';
        errorMessage.textContent = message;
    }
    
    // Process new video button
    processNewBtn.addEventListener('click', () => {
        resetUI();
    });
    
    // Try again button
    tryAgainBtn.addEventListener('click', () => {
        resetUI();
    });
    
    // Reset UI to initial state
    function resetUI() {
        inputSection.style.display = 'block';
        progressSection.style.display = 'none';
        resultSection.style.display = 'none';
        errorSection.style.display = 'none';
        
        // Reset progress
        progressFill.style.width = '0%';
        progressText.textContent = '0%';
        statusMessage.textContent = 'Starting process...';
        
        // Clear task ID and interval
        currentTaskId = null;
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
    }
});
