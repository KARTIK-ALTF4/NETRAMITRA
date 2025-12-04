// Tab Switching
const loginTab = document.getElementById('login-tab');
const registerTab = document.getElementById('register-tab');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');

loginTab.addEventListener('click', () => {
    loginTab.classList.add('active');
    registerTab.classList.remove('active');
    loginForm.style.display = 'block';
    registerForm.style.display = 'none';
});

registerTab.addEventListener('click', () => {
    registerTab.classList.add('active');
    loginTab.classList.remove('active');
    registerForm.style.display = 'block';
    loginForm.style.display = 'none';
});

// Login Form Submission
const loginSubmitBtn = document.getElementById('login-submit');
if (loginSubmitBtn) {
    loginSubmitBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value;
        const messageDiv = document.getElementById('login-message');
    
        // Validation
        if (!username || !password) {
            showMessage(messageDiv, 'Please fill in all fields', 'error');
            return;
        }
        
        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            
            const data = await response.json();
            
            if (data.success) {
                showMessage(messageDiv, 'Login successful! Redirecting...', 'success');
                setTimeout(() => {
                    window.location.href = '/dashboard';
                }, 1000);
            } else {
                showMessage(messageDiv, data.error || 'Login failed', 'error');
            }
        } catch (error) {
            showMessage(messageDiv, 'Network error. Please try again.', 'error');
            console.error('Login error:', error);
        }
    });
}

// Register Form Submission
const registerSubmitBtn = document.getElementById('register-submit');
if (registerSubmitBtn) {
    registerSubmitBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('register-username').value.trim();
        const email = document.getElementById('register-email').value.trim();
        const phone = document.getElementById('register-phone').value.trim();
        const password = document.getElementById('register-password').value;
        const confirmPassword = document.getElementById('register-confirm-password').value;
        const messageDiv = document.getElementById('register-message');
        
        // Validation
        if (!username || !email || !password || !confirmPassword) {
            showMessage(messageDiv, 'Please fill in all required fields', 'error');
            return;
        }
        
        if (password !== confirmPassword) {
            showMessage(messageDiv, 'Passwords do not match', 'error');
            return;
        }
        
        if (password.length < 6) {
            showMessage(messageDiv, 'Password must be at least 6 characters', 'error');
            return;
        }
        
        // Email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            showMessage(messageDiv, 'Please enter a valid email', 'error');
            return;
        }
        
        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, phone, password })
            });
            
            const data = await response.json();
            
            if (data.success) {
                showMessage(messageDiv, 'Registration successful! Redirecting...', 'success');
                setTimeout(() => {
                    window.location.href = '/dashboard';
                }, 1500);
            } else {
                showMessage(messageDiv, data.error || 'Registration failed', 'error');
            }
        } catch (error) {
            showMessage(messageDiv, 'Network error. Please try again.', 'error');
            console.error('Registration error:', error);
        }
    });
}

// Enter key submission
document.querySelectorAll('#login-form input').forEach(input => {
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            document.getElementById('login-submit').click();
        }
    });
});

document.querySelectorAll('#register-form input').forEach(input => {
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            document.getElementById('register-submit').click();
        }
    });
});

// Helper function to show messages
function showMessage(element, message, type) {
    element.textContent = message;
    element.className = `form-message ${type}`;
    element.style.display = 'block';
    
    setTimeout(() => {
        element.style.display = 'none';
    }, 5000);
}
