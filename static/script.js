document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const loadingState = document.getElementById('loading-state');
    const resultCard = document.getElementById('result-card');
    const errorCard = document.getElementById('error-card');

    // Result elements
    const docTypeEl = document.getElementById('res-doc-type');
    const writingTypeEl = document.getElementById('res-writing-type');
    const confidenceBar = document.getElementById('res-confidence-bar');
    const confidenceText = document.getElementById('res-confidence-text');
    const processingTimeEl = document.getElementById('res-processing-time');

    // Buttons
    const resetBtn = document.getElementById('reset-btn');
    const errorResetBtn = document.getElementById('error-reset-btn');
    const errorMessage = document.getElementById('error-message');

    // Click to upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
    });

    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });

    resetBtn.addEventListener('click', resetUI);
    errorResetBtn.addEventListener('click', resetUI);

    function resetUI() {
        resultCard.classList.add('hidden');
        errorCard.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        fileInput.value = '';
        confidenceBar.style.width = '0%';
    }

    function showLoading() {
        uploadZone.classList.add('hidden');
        loadingState.classList.remove('hidden');
    }

    function hideLoading() {
        loadingState.classList.add('hidden');
    }

    async function handleFile(file) {
        // Validate file type locally first
        const validTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showError('Invalid file type. Please upload a PDF, PNG, or JPG.');
            return;
        }

        if (file.size > 10 * 1024 * 1024) { // 10MB
            showError('File is too large. Maximum size is 10MB.');
            return;
        }

        showLoading();

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Get selected approach
            const approach = document.querySelector('input[name="approach"]:checked').value;
            
            const response = await fetch(`/api/v1/classify?all_pages=false&approach=${approach}`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'An error occurred while queueing.');
            }

            // Start polling the status endpoint
            pollStatus(data.job_id);
        } catch (error) {
            console.error('Error:', error);
            showError(error.message);
        }
    }

    async function pollStatus(jobId) {
        try {
            const response = await fetch(`/api/v1/status/${jobId}`);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to check job status.');
            }

            if (data.status === 'completed') {
                displayResults(data.result);
            } else if (data.status === 'error') {
                showError(data.error || 'An error occurred during processing.');
            } else {
                // Still processing, poll again in 5 seconds
                setTimeout(() => pollStatus(jobId), 5000);
            }
        } catch (error) {
            console.error('Polling Error:', error);
            showError(error.message);
        }
    }

    function displayResults(data) {
        hideLoading();
        resultCard.classList.remove('hidden');

        docTypeEl.textContent = data.document_type;
        writingTypeEl.textContent = data.writing_type;
        const languageEl = document.getElementById('res-language');
        if (languageEl) {
            languageEl.textContent = data.language || 'Unknown';
        }

        const summaryEl = document.getElementById('res-summary');
        if (summaryEl) {
            summaryEl.textContent = data.summary || 'No details extracted.';
        }

        processingTimeEl.textContent = `${data.processing_time}s`;

        // Animate confidence bar
        const confidencePercent = Math.round(data.confidence * 100);
        setTimeout(() => {
            confidenceBar.style.width = `${confidencePercent}%`;

            // Color coding based on confidence
            if (confidencePercent > 80) {
                confidenceBar.style.background = 'linear-gradient(90deg, #6366f1, #10b981)';
            } else if (confidencePercent > 50) {
                confidenceBar.style.background = 'linear-gradient(90deg, #6366f1, #f59e0b)';
            } else {
                confidenceBar.style.background = 'linear-gradient(90deg, #6366f1, #ef4444)';
            }
        }, 100); // Slight delay to ensure CSS transition triggers

        confidenceText.textContent = `${confidencePercent}%`;
    }

    function showError(msg) {
        hideLoading();
        errorMessage.textContent = msg;
        errorCard.classList.remove('hidden');
    }
});
