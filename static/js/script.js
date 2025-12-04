// Global variables
let selectedFile = null;
let isLoggedIn = false;
let cameraStream = null;
let currentMode = null; // 'upload' or 'camera'

// Check login status on page load
document.addEventListener('DOMContentLoaded', async () => {
    await checkLoginStatus();
    updateNavLinks();
    loadHistory();
    setupDetectionOptions();
});

// Check if user is logged in
async function checkLoginStatus() {
    try {
        const response = await fetch('/api/check-session');
        const data = await response.json();
        
        if (data.logged_in) {
            isLoggedIn = true;
            document.getElementById('loginBtn').style.display = 'none';
            document.getElementById('profileDropdown').style.display = 'block';
            document.getElementById('profileUsername').textContent = data.username;
        } else {
            isLoggedIn = false;
            document.getElementById('loginBtn').style.display = 'block';
            document.getElementById('profileDropdown').style.display = 'none';
        }
    } catch (error) {
        console.error('Error checking login status:', error);
    }
}

// Smooth scroll to section
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    section.scrollIntoView({ behavior: 'smooth' });
}

// Update active nav link on scroll
function updateNavLinks() {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');
    
    window.addEventListener('scroll', () => {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (window.pageYOffset >= sectionTop - 100) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ============================================================================
// DETECTION MODE SELECTION
// ============================================================================

function setupDetectionOptions() {
    const uploadOption = document.getElementById('uploadOption');
    const cameraOption = document.getElementById('cameraOption');
    const uploadMode = document.getElementById('uploadMode');
    const cameraMode = document.getElementById('cameraMode');
    
    if (!uploadOption || !cameraOption) {
        console.error('❌ Detection option buttons not found');
        return;
    }
    
    // Set upload mode as default on page load
    currentMode = 'upload';
    uploadMode.style.display = 'block';
    uploadOption.classList.add('active');
    
    uploadOption.addEventListener('click', () => {
        console.log('📤 Upload mode selected');
        currentMode = 'upload';
        uploadMode.style.display = 'block';
        cameraMode.style.display = 'none';
        uploadOption.classList.add('active');
        cameraOption.classList.remove('active');
        stopCamera();
    });
    
    cameraOption.addEventListener('click', () => {
        console.log('📹 Camera mode selected');
        currentMode = 'camera';
        uploadMode.style.display = 'none';
        cameraMode.style.display = 'block';
        cameraOption.classList.add('active');
        uploadOption.classList.remove('active');
        startCamera();
    });
    
    console.log('✅ Detection options setup complete');
}

// ============================================================================
// CAMERA FUNCTIONALITY
// ============================================================================

async function startCamera() {
    try {
        console.log('🎥 Starting camera...');
        const cameraVideo = document.getElementById('cameraVideo');
        
        if (!cameraVideo) {
            console.error('❌ Camera video element not found');
            return;
        }
        
        // Check if page is served over HTTPS or localhost
        const isSecure = window.location.protocol === 'https:' || 
                        window.location.hostname === 'localhost' || 
                        window.location.hostname === '127.0.0.1' ||
                        window.location.hostname.startsWith('192.168.');
        
        if (!isSecure) {
            alert('⚠️ Camera requires HTTPS or localhost.\n\nPlease access the app via:\n• https://... (secure connection)\n• http://localhost:5000\n• http://127.0.0.1:5000\n\nOr use "Upload Photo" option instead.');
            return;
        }
        
        // Check browser compatibility
        console.log('Browser info:', {
            hasMediaDevices: !!navigator.mediaDevices,
            hasGetUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
            protocol: window.location.protocol,
            hostname: window.location.hostname
        });
        
        // Check if getUserMedia is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera API not supported in this browser. Please use Chrome, Edge, or Firefox.');
        }
        
        console.log('📞 Requesting camera access...');
        
        // Request camera access with basic constraints
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user', // Front camera for selfies
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
        
        console.log('✅ Camera stream obtained');
        
        cameraVideo.srcObject = cameraStream;
        
        // Wait for video to be ready
        cameraVideo.onloadedmetadata = () => {
            cameraVideo.play();
            console.log('✅ Camera started successfully');
            console.log(`📹 Camera resolution: ${cameraVideo.videoWidth}x${cameraVideo.videoHeight}`);
        };
        
    } catch (error) {
        console.error('❌ Camera error:', error);
        console.error('Error name:', error.name);
        console.error('Error message:', error.message);
        
        let errorMsg = '🎥 Camera Error\n\n';
        
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMsg += '❌ Camera permission denied.\n\n';
            errorMsg += 'To fix:\n';
            errorMsg += '1. Click the 🔒 or ⓘ icon in your browser address bar\n';
            errorMsg += '2. Allow camera access for this site\n';
            errorMsg += '3. Refresh the page (F5)\n';
            errorMsg += '4. Click "Live Camera" again\n\n';
            errorMsg += 'कैमरा परमिशन नहीं मिली। ब्राउज़र सेटिंग में Allow करें।';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            errorMsg += '❌ No camera found on this device.\n\n';
            errorMsg += 'Please use "Upload Photo" option instead.\n\n';
            errorMsg += 'इस डिवाइस में कैमरा नहीं मिला। "Upload Photo" इस्तेमाल करें।';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            errorMsg += '❌ Camera is already in use.\n\n';
            errorMsg += 'To fix:\n';
            errorMsg += '1. Close other apps using the camera\n';
            errorMsg += '2. Close other browser tabs with camera\n';
            errorMsg += '3. Refresh this page (F5)\n\n';
            errorMsg += 'कैमरा किसी और ऐप में इस्तेमाल हो रहा है।';
        } else if (error.message && error.message.includes('not supported')) {
            errorMsg += '❌ Camera not supported in this browser.\n\n';
            errorMsg += 'Please:\n';
            errorMsg += '1. Use Chrome, Edge, or Firefox browser\n';
            errorMsg += '2. Or use "Upload Photo" option\n\n';
            errorMsg += '"Upload Photo" ऑप्शन इस्तेमाल करें।';
        } else {
            errorMsg += '❌ ' + error.message + '\n\n';
            errorMsg += 'Troubleshooting:\n';
            errorMsg += '1. Refresh page (F5)\n';
            errorMsg += '2. Check browser camera permissions\n';
            errorMsg += '3. Try another browser (Chrome/Edge/Firefox)\n';
            errorMsg += '4. Or use "Upload Photo" option\n\n';
            errorMsg += 'या फिर "Upload Photo" ऑप्शन इस्तेमाल करें।';
        }
        
        alert(errorMsg);
        
        // Switch back to upload mode
        document.getElementById('cameraMode').style.display = 'none';
        document.getElementById('uploadMode').style.display = 'block';
        document.getElementById('uploadOption').classList.add('active');
        document.getElementById('cameraOption').classList.remove('active');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        const cameraVideo = document.getElementById('cameraVideo');
        if (cameraVideo) {
            cameraVideo.srcObject = null;
        }
        console.log('🛑 Camera stopped');
    }
}

// Capture photo from camera
const captureBtn = document.getElementById('captureBtn');
if (captureBtn) {
    captureBtn.addEventListener('click', () => {
        console.log('📸 Capture button clicked');
        const video = document.getElementById('cameraVideo');
        const canvas = document.getElementById('cameraCanvas');
        
        if (!video || !canvas) {
            console.error('❌ Video or canvas element not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to video size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob and upload
        canvas.toBlob(async (blob) => {
            const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
            selectedFile = file;
            
            // Show preview
            previewImage.src = canvas.toDataURL('image/jpeg');
            previewBox.style.display = 'block';
            analyzeBtn.style.display = 'block';
            
            // Stop camera after capture
            stopCamera();
            
            // Switch to preview mode
            document.getElementById('cameraMode').style.display = 'none';
            document.getElementById('uploadMode').style.display = 'block';
            uploadBox.style.display = 'none';
            
            console.log('✅ Photo captured from camera');
        }, 'image/jpeg', 0.95);
    });
}

// Stop camera button
const stopCameraBtn = document.getElementById('stopCameraBtn');
if (stopCameraBtn) {
    stopCameraBtn.addEventListener('click', () => {
        console.log('⏹️ Stop camera button clicked');
        stopCamera();
        document.getElementById('cameraMode').style.display = 'none';
        document.getElementById('cameraOption').classList.remove('active');
    });
}

// ============================================================================
// FILE UPLOAD HANDLING
// ============================================================================
const uploadBox = document.getElementById('uploadBox');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const previewBox = document.getElementById('previewBox');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultContainer = document.getElementById('resultContainer');

uploadBox.addEventListener('click', () => {
    fileInput.click();
});

uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#667eea';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    selectedFile = file;
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.style.display = 'none';
        previewBox.style.display = 'block';
        analyzeBtn.style.display = 'block';
        resultContainer.style.display = 'none';
    };
    
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewBox.style.display = 'none';
    analyzeBtn.style.display = 'none';
    resultContainer.style.display = 'none';
});

// Analyze Image
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    
    const loadingOverlay = document.getElementById('loadingOverlay');
    loadingOverlay.style.display = 'flex';
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResult(data);
            if (isLoggedIn) {
                loadHistory();
            }
        } else {
            alert('Error: ' + data.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during analysis');
    } finally {
        loadingOverlay.style.display = 'none';
    }
});

function displayResult(data) {
    const resultIcon = document.getElementById('resultIcon');
    const resultBadge = document.getElementById('resultBadge');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const resultAdvice = document.getElementById('resultAdvice');
    const gradcamContainer = document.getElementById('gradcamContainer');
    const gradcamImage = document.getElementById('gradcamImage');
    
    const hasCataract = data.result === 'Cataract Detected';
    
    // Update icon
    resultIcon.innerHTML = hasCataract ? 
        '<i class="fas fa-exclamation-circle"></i>' : 
        '<i class="fas fa-check-circle"></i>';
    resultIcon.className = hasCataract ? 'result-icon danger' : 'result-icon success';
    
    // Update badge
    resultBadge.textContent = data.result;
    resultBadge.className = hasCataract ? 'result-badge danger' : 'result-badge success';
    
    // Update confidence
    confidenceValue.textContent = data.confidence + '%';
    confidenceFill.style.width = data.confidence + '%';
    
    // Display Grad-CAM visualization if available
    if (data.gradcam_image) {
        gradcamImage.src = 'data:image/png;base64,' + data.gradcam_image;
        gradcamContainer.style.display = 'block';
    } else {
        gradcamContainer.style.display = 'none';
    }
    
    // Update advice
    if (hasCataract) {
        resultAdvice.innerHTML = `
            <h4>⚠️ Important Next Steps:</h4>
            <ul>
                <li>Consult an ophthalmologist immediately for proper examination</li>
                <li>This is a screening result, not a medical diagnosis</li>
                <li>Early treatment can prevent vision loss</li>
                <li>Book an appointment at nearest eye hospital</li>
            </ul>
            <p><strong>मोतियाबिंद की पुष्टि के लिए नेत्र चिकित्सक से परामर्श करें</strong></p>
        `;
    } else {
        resultAdvice.innerHTML = `
            <h4>✅ Good News!</h4>
            <ul>
                <li>No signs of cataract detected in this screening</li>
                <li>Continue regular eye checkups every 6-12 months</li>
                <li>Protect your eyes from UV radiation with sunglasses</li>
                <li>Maintain a healthy diet rich in vitamins</li>
            </ul>
            <p><strong>नियमित जांच और आंखों की देखभाल जारी रखें</strong></p>
        `;
    }
    
    previewBox.style.display = 'none';
    analyzeBtn.style.display = 'none';
    resultContainer.style.display = 'block';
    
    // Scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// New Test Button
document.getElementById('newTestBtn').addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewBox.style.display = 'none';
    analyzeBtn.style.display = 'none';
    resultContainer.style.display = 'none';
    
    // Scroll back to upload
    uploadBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
});

// Chatbot Handling
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const quickBtns = document.querySelectorAll('.quick-btn');

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

quickBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        chatInput.value = btn.dataset.question;
        sendMessage();
    });
});

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Add user message
    addMessage(message, 'user');
    chatInput.value = '';
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addMessage(data.response, 'bot');
        } else {
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
    }
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    // Convert markdown-style formatting to HTML
    let formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    
    content.innerHTML = formattedText;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Authentication Handling
const loginBtn = document.getElementById('loginBtn');
const loginModal = document.getElementById('loginModal');
const closeLoginModal = document.getElementById('closeLoginModal');
const loginTab = document.getElementById('loginTab');
const registerTab = document.getElementById('registerTab');
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const logoutBtn = document.getElementById('logoutBtn');

loginBtn.addEventListener('click', () => {
    loginModal.style.display = 'block';
});

closeLoginModal.addEventListener('click', () => {
    loginModal.style.display = 'none';
});

window.addEventListener('click', (e) => {
    if (e.target === loginModal) {
        loginModal.style.display = 'none';
    }
});

loginTab.addEventListener('click', () => {
    loginTab.classList.add('active');
    registerTab.classList.remove('active');
    loginForm.style.display = 'block';
    registerForm.style.display = 'none';
    document.getElementById('modalTitle').textContent = 'Login to Your Account';
});

registerTab.addEventListener('click', () => {
    registerTab.classList.add('active');
    loginTab.classList.remove('active');
    registerForm.style.display = 'block';
    loginForm.style.display = 'none';
    document.getElementById('modalTitle').textContent = 'Create New Account';
});

loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;
    
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showFormMessage('Login successful! Welcome back.', 'success');
            setTimeout(() => {
                loginModal.style.display = 'none';
                checkLoginStatus();
                loadHistory();
            }, 1000);
        } else {
            showFormMessage(data.message, 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showFormMessage('An error occurred. Please try again.', 'error');
    }
});

registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('registerUsername').value;
    const email = document.getElementById('registerEmail').value;
    const phone = document.getElementById('registerPhone').value;
    const password = document.getElementById('registerPassword').value;
    const confirmPassword = document.getElementById('registerConfirmPassword').value;
    
    if (password !== confirmPassword) {
        showFormMessage('Passwords do not match', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, phone, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showFormMessage('Registration successful! Welcome.', 'success');
            setTimeout(() => {
                loginModal.style.display = 'none';
                checkLoginStatus();
                loadHistory();
            }, 1000);
        } else {
            showFormMessage(data.message, 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showFormMessage('An error occurred. Please try again.', 'error');
    }
});

logoutBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/logout', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            checkLoginStatus();
            document.getElementById('historyList').style.display = 'none';
            document.getElementById('emptyHistory').style.display = 'none';
            document.getElementById('historyLoginPrompt').style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
    }
});

function showFormMessage(message, type) {
    const formMessage = document.getElementById('formMessage');
    formMessage.textContent = message;
    formMessage.className = `form-message ${type}`;
    formMessage.style.display = 'block';
    
    setTimeout(() => {
        formMessage.style.display = 'none';
    }, 5000);
}

// History Handling
async function loadHistory() {
    if (!isLoggedIn) {
        document.getElementById('historyLoginPrompt').style.display = 'block';
        document.getElementById('historyList').style.display = 'none';
        document.getElementById('emptyHistory').style.display = 'none';
        return;
    }
    
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('historyLoginPrompt').style.display = 'none';
            
            if (data.history.length === 0) {
                document.getElementById('emptyHistory').style.display = 'block';
                document.getElementById('historyList').style.display = 'none';
            } else {
                document.getElementById('emptyHistory').style.display = 'none';
                document.getElementById('historyList').style.display = 'block';
                displayHistory(data.history);
            }
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function displayHistory(history) {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';
    
    history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const hasCataract = item.result === 'Cataract Detected';
        const badgeClass = hasCataract ? 'danger' : 'success';
        
        historyItem.innerHTML = `
            <div class="history-info">
                <h4>${item.result}</h4>
                <p>📅 ${item.timestamp}</p>
                <p>🎯 Confidence: ${item.confidence}%</p>
            </div>
            <div class="history-actions">
                <div class="history-badge ${badgeClass}">
                    ${hasCataract ? '⚠️' : '✅'}
                </div>
                <button class="download-btn" onclick="downloadReport(${item.id})" title="Download PDF Report">
                    <i class="fas fa-download"></i> PDF
                </button>
            </div>
        `;
        
        historyList.appendChild(historyItem);
    });
}

// Download PDF Report
function downloadReport(predictionId) {
    window.location.href = `/api/download-report/${predictionId}`;
    showMessage('Downloading report...', 'success');
}