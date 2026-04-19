import os
import json
import base64
from fastapi import FastAPI, Response, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect

from datetime import datetime
from llm import generate_response, reset_memory, conversation_history # Note: we now import conversation_history directly
from db import save_call_data, get_all_calls, get_transcript, update_recording_url
from tts import generate_speech_stream
from asr import setup_deepgram_connection  # Import the ear!

load_dotenv()

app = FastAPI(title="FYP Voice Agent API")

API_KEYS = {
    "OpenAI": os.getenv("OPENAI_API_KEY"),
    "Deepgram": os.getenv("DEEPGRAM_API_KEY"),
    "ElevenLabs": os.getenv("ELEVENLABS_API_KEY"),
    "Twilio SID": os.getenv("TWILIO_ACCOUNT_SID"),
}

@app.get("/")
async def root():
    return {"status": "ok", "message": "FYP Voice Agent is live!"}

@app.get("/health")
async def health_check():
    missing_keys = [name for name, key in API_KEYS.items() if not key]
    if missing_keys:
        return {"status": "error", "missing_keys": missing_keys}
    return {"status": "ok", "message": "All API keys loaded successfully!"}

@app.get("/test-chat")
async def test_chat(message: str):
    reply = await generate_response(message)
    return {"user_message": message, "agent_reply": reply}

@app.get("/test-speak")
async def test_speak(message: str):
    reply_text = await generate_response(message)
    audio_data = generate_speech(reply_text)
    if not audio_data:
        return {"error": "Voice generation failed."}
    return Response(content=audio_data, media_type="audio/mpeg")

@app.post("/incoming-call")
async def incoming_call(request: Request):
    response = VoiceResponse()
    response.say("This call is being recorded for quality assurance purposes.")
    
    connect = Connect()
    host = request.headers.get("host")
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    
    return HTMLResponse(content=str(response), media_type="application/xml")



# --- WEB DASHBOARD ROUTES ---

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serves the frontend HTML dashboard."""
    try:
        with open("dashboard.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Dashboard file not found. Create dashboard.html!</h1>")

@app.get("/api/calls")
async def api_get_calls():
    """Returns all call logs as JSON."""
    calls = get_all_calls()
    # Format datetimes to strings so they can be sent as JSON
    for call in calls:
        call['start_time'] = call['start_time'].isoformat() if call['start_time'] else None
        call['end_time'] = call['end_time'].isoformat() if call['end_time'] else None
    return JSONResponse(content=calls)

@app.get("/api/calls/{call_id}/transcript")
async def api_get_transcript(call_id: str):
    """Returns the transcript for a specific call as JSON."""
    transcript = get_transcript(call_id)
    for t in transcript:
        t['timestamp'] = t['timestamp'].isoformat() if t['timestamp'] else None
    return JSONResponse(content=transcript)

@app.post("/recording-webhook")
async def recording_webhook(request: Request):
    """Twilio sends a POST request here when the call recording is ready."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    recording_url = form_data.get("RecordingUrl")
    
    if call_sid and recording_url:
        # Twilio sends a raw URL. Appending .mp3 gives us the actual audio file
        mp3_url = f"{recording_url}.mp3"
        print(f"📥 Received recording from Twilio: {mp3_url}")
        update_recording_url(call_sid, mp3_url)
        
    return {"status": "success"}

# --- UPDATED: THE REAL AUDIO WEBSOCKET ---
# --- UPDATED: THE FULL CONVERSATION WEBSOCKET ---
# --- UPDATED: WEBSOCKET WITH CONTENT INTERRUPTION ---
# --- UPDATED: STATE-MANAGED PRODUCTION WEBSOCKET ---
# --- UPDATED: STATE-MANAGED PRODUCTION WEBSOCKET ---
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("📞 Incoming call connected! Twilio stream is open.")
    
    stream_sid = None
    call_sid = None
    ai_is_speaking = False  # Is Twilio currently playing our audio?
    is_processing = False   # Is the LLM currently thinking?
    call_start_time = datetime.now()  # Track when the call started for data logging
    
# 1. Handle complete sentences (AI responds)
    async def handle_transcript(sentence: str):
        nonlocal ai_is_speaking, is_processing
        
        if is_processing or ai_is_speaking:
            return
            
        is_processing = True
        print(f"\n🗣️ Customer said: {sentence}")
        print("🧠 Passing to LLM...")
        
        reply_text = await generate_response(sentence)
        print(f"🤖 AI says: {reply_text}")
        
        # --- NEW STREAMING LOGIC ---
        ai_is_speaking = True # Set this true BEFORE streaming so we don't hear echoes
        print("🔊 Streaming audio directly to Twilio...")
        
        if stream_sid:
            # Send chunks to the phone the millisecond they are generated
            for audio_chunk in generate_speech_stream(reply_text):
                audio_payload = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": audio_payload
                    }
                }))
                
            # Once all chunks are sent, drop the mark to know when the audio finishes playing
            await websocket.send_text(json.dumps({
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {"name": "ai_finished"}
            }))
        # ---------------------------
            
        is_processing = False
            
    # 2. Handle Interruptions (Barge-in)
    async def handle_interruption():
        nonlocal ai_is_speaking
        # Only clear the buffer if the AI is ACTUALLY talking
        if ai_is_speaking and stream_sid:
            print("🛑 User interrupted! Instantly clearing Twilio audio buffer.")
            await websocket.send_text(json.dumps({
                "event": "clear",
                "streamSid": stream_sid
            }))
            # Instantly unlock so the AI can listen to the interruption
            ai_is_speaking = False 
        
    # 3. Start the Deepgram connection
    dg_connection = await setup_deepgram_connection(handle_transcript, handle_interruption)
    if not dg_connection:
        await websocket.close()
        return

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                call_sid = data['start']['callSid']
                print(f"🔗 Call Stream ID captured: {stream_sid}")
            
            elif data['event'] == 'media':
                # RULE 2: ECHO CANCELLATION. If the AI is currently speaking, we do NOT send the user's audio to Deepgram to prevent it from hearing its own voice and causing a feedback loop.
                # Only stream audio to Deepgram if the AI is NOT speaking.
                if not ai_is_speaking:
                    audio_payload = data['media']['payload']
                    audio_bytes = base64.b64decode(audio_payload)
                    await dg_connection.send(audio_bytes)
            
            elif data['event'] == 'mark':
                # RULE 3: TURN-TAKING. Twilio hit our mark, meaning the AI finished naturally.
                if data['mark']['name'] == 'ai_finished':
                    ai_is_speaking = False
                    print("✅ AI finished speaking naturally. Ready to listen.")
                    
            elif data['event'] == 'stop':
                print("Call ended by user.")
                break

    except Exception as e:
        print(f"Call disconnected: {e}")
    finally:
        if stream_sid:
            print("Saving call data to Postgres...")
            save_call_data(stream_sid, call_sid, call_start_time, conversation_history)
            
        await dg_connection.finish()
        reset_memory()