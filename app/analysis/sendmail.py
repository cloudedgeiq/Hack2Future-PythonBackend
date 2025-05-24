import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify
import os

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Email configuration
EMAIL_ADDRESS = os.getenv("GOOGLE_APP_EMAIL")
EMAIL_PASSWORD = os.getenv("GOOGLE_APP_PASSWORD")

def send_email(subject, body, to):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))
    print("Preparing to send email...")

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to, msg.as_string())
            print("Email sent successfully")
    except Exception as e:
        print("Error sending email:", e)


