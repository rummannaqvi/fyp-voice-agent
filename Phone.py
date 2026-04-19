import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

# Set up Twilio Client
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)

# IMPORTANT: Make sure this matches your currently running ngrok URL!
NGROK_URL = "https://aee6-116-90-118-246.ngrok-free.app"

def initiate_outbound_call(customer_number: str):
    """Triggers Twilio to call a customer and connect them to your AI."""
    try:
        call = client.calls.create(
            to=customer_number, 
            from_=os.getenv("TWILIO_PHONE_NUMBER"), 
            url=f"{NGROK_URL}/incoming-call", 
            method="POST",

            record = True,  # Enable call recording
            recording_status_callback=f"{NGROK_URL}/recording-webhook",  # Callback for when recording is complete
            recording_status_callback_event=["completed"]  # Only trigger callback when recording is finished
        )
        print(f"Dialing {customer_number}... Call SID: {call.sid}")
    except Exception as e:
        print(f"Error making call: {e}")

if __name__ == "__main__":
    # ---> PUT YOUR VERIFIED CELL PHONE NUMBER HERE <---
    # Make sure to include the country code (e.g., +92...)
    target_number = "+18139212306" 
    
    initiate_outbound_call(target_number)