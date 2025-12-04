"""
Run Flask app with HTTPS for phone camera access
"""
from app import app

if __name__ == '__main__':
    # Run with adhoc SSL (self-signed certificate)
    # This allows camera to work on phones via local network
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        ssl_context='adhoc'  # Creates temporary self-signed certificate
    )
