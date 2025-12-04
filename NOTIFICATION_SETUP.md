# Email and SMS Notification Setup Guide

## Overview
The application now sends automatic email and SMS notifications when cataract detection is performed. Follow these steps to configure the credentials.

---

## 📧 Email Configuration (Gmail)

### Step 1: Enable 2-Step Verification
1. Go to your Google Account: https://myaccount.google.com/
2. Click **Security** in the left sidebar
3. Under "Signing in to Google", enable **2-Step Verification**
4. Follow the setup wizard

### Step 2: Generate App Password
1. Go to: https://myaccount.google.com/apppasswords
2. Select **App**: Choose "Mail" or "Other (Custom name)"
3. Select **Device**: Choose "Windows Computer" or "Other (Custom name)"
4. Click **Generate**
5. **Copy the 16-character password** (e.g., `abcd efgh ijkl mnop`)

### Step 3: Update app.py
Open `app.py` and find these lines (around line 247):

```python
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = 'your_email@gmail.com'  # CHANGE THIS
EMAIL_PASSWORD = 'your_app_password'  # CHANGE THIS
```

Replace with your actual credentials:
```python
EMAIL_USER = 'kartikproject@gmail.com'  # Your Gmail address
EMAIL_PASSWORD = 'abcdefghijklmnop'     # The 16-char app password (no spaces)
```

---

## 📱 SMS Configuration (Twilio)

### Step 1: Create Twilio Account
1. Go to: https://www.twilio.com/try-twilio
2. Sign up for a **free trial account**
3. Verify your phone number during registration

### Step 2: Get Credentials
1. After logging in, go to **Twilio Console Dashboard**: https://console.twilio.com/
2. Find your credentials:
   - **Account SID** (e.g., `ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)
   - **Auth Token** (click "Show" to reveal)
3. Get a **Twilio Phone Number**:
   - Go to **Phone Numbers** → **Manage** → **Buy a number**
   - Or use the free trial number provided
   - Format: `+1234567890` (with country code)

### Step 3: Verify Recipient Numbers (Free Trial)
⚠️ **Important**: Twilio trial accounts can only send SMS to **verified phone numbers**.

1. Go to **Phone Numbers** → **Manage** → **Verified Caller IDs**
2. Click **Add a new number**
3. Enter the phone number that will receive SMS (your test number)
4. Verify via SMS code

### Step 4: Update app.py
Open `app.py` and find these lines (around line 252):

```python
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE = '+1234567890'  # Your Twilio number
```

Replace with your actual credentials:
```python
TWILIO_ACCOUNT_SID = 'ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'  # From console
TWILIO_AUTH_TOKEN = 'your_auth_token_here'                 # From console
TWILIO_PHONE = '+15551234567'                               # Your Twilio number
```

---

## 🧪 Testing

### Test Email Only (without running app)
Create a test file `test_email.py`:

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = 'kartikproject@gmail.com'  # Your email
EMAIL_PASSWORD = 'your_app_password'     # Your app password

msg = MIMEMultipart()
msg['From'] = EMAIL_USER
msg['To'] = 'test@example.com'
msg['Subject'] = 'Test Email'
msg.attach(MIMEText('This is a test email', 'plain'))

try:
    server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASSWORD)
    server.send_message(msg)
    server.quit()
    print('✅ Email sent successfully!')
except Exception as e:
    print(f'❌ Email failed: {e}')
```

Run: `python test_email.py`

### Test SMS Only
Create `test_sms.py`:

```python
from twilio.rest import Client

TWILIO_ACCOUNT_SID = 'ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE = '+15551234567'
TO_PHONE = '+919876543210'  # Verified number for trial accounts

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

try:
    message = client.messages.create(
        body='Test SMS from Cataract Detection App',
        from_=TWILIO_PHONE,
        to=TO_PHONE
    )
    print(f'✅ SMS sent! SID: {message.sid}')
except Exception as e:
    print(f'❌ SMS failed: {e}')
```

Run: `python test_sms.py`

### Test Full App
1. Start Flask: `python app.py`
2. Register a new user with your phone number
3. Upload an eye image
4. Check email inbox and phone for notifications

---

## 📝 Notification Format

### Email
- **Subject**: Cataract Detection Result - [Result]
- **Body**: HTML formatted with:
  - Patient name
  - Detection result (color-coded)
  - Confidence percentage
  - Recommendations
  - Professional disclaimer

### SMS
- **Format**: 
  ```
  ⚠️ Cataract Detection: CATARACT DETECTED (85.3%)
  Immediate consultation recommended.
  Report: [timestamp]
  ```

---

## 🔒 Security Best Practices

1. **Never commit credentials to Git**:
   ```bash
   # Add to .gitignore
   echo "app.py" >> .gitignore  # If you store credentials in code
   ```

2. **Use environment variables** (Recommended):
   ```python
   import os
   EMAIL_USER = os.getenv('EMAIL_USER', 'default@gmail.com')
   EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'default_password')
   ```

3. **Rotate credentials regularly** (every 3-6 months)

4. **Monitor usage**:
   - Gmail: Check "Last account activity" 
   - Twilio: Monitor usage in console dashboard

---

## ⚠️ Troubleshooting

### Email Issues
- **"Username and Password not accepted"**: Check app password (no spaces), ensure 2FA enabled
- **"SMTP Authentication Error"**: Verify EMAIL_USER matches the account that generated app password
- **"Connection refused"**: Check firewall/antivirus blocking port 587

### SMS Issues
- **"Unable to create record"**: Verify phone number format includes country code (e.g., `+91` for India)
- **"Unverified number"**: For trial accounts, recipient must be verified in Twilio console
- **"Insufficient funds"**: Trial accounts have limited credits; upgrade for production use

### General
- **No notifications received**: Check Flask console for error messages
- **Slow notifications**: Notifications run in background thread; allow 5-10 seconds for delivery

---

## 💰 Cost Information

### Email (Gmail)
- **Free**: Unlimited emails from personal Gmail accounts
- **Limits**: 500 recipients per day for free accounts

### SMS (Twilio)
- **Trial**: $15 free credit (≈500 SMS in India)
- **Production**: $0.0079 per SMS in India, $0.0075 in USA
- **Upgrade**: https://www.twilio.com/pricing

---

## 📞 Support

- **Twilio Docs**: https://www.twilio.com/docs/sms
- **Gmail SMTP**: https://support.google.com/mail/answer/7126229
- **Project Issues**: Contact project maintainer

---

## ✅ Quick Checklist

- [ ] Gmail 2-Step Verification enabled
- [ ] Gmail App Password generated
- [ ] Updated EMAIL_USER in app.py
- [ ] Updated EMAIL_PASSWORD in app.py
- [ ] Twilio account created
- [ ] Account SID copied from console
- [ ] Auth Token copied from console
- [ ] Twilio phone number obtained
- [ ] Updated TWILIO_ACCOUNT_SID in app.py
- [ ] Updated TWILIO_AUTH_TOKEN in app.py
- [ ] Updated TWILIO_PHONE in app.py
- [ ] Test phone numbers verified (for trial)
- [ ] Tested email sending
- [ ] Tested SMS sending
- [ ] Full app test completed

---

**Last Updated**: December 2024
