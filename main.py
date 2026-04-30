import os
import json
import base64
import traceback
import httpx
from fastapi import FastAPI, Response, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient
from langchain_core.messages import AIMessage

from datetime import datetime
from llm import generate_response, reset_memory, conversation_history
from db import save_call_data, get_all_calls, get_transcript, update_recording_url
from tts import generate_speech_stream
from asr import setup_deepgram_connection

load_dotenv()

app = FastAPI(title="FYP Voice Agent API")

API_KEYS = {
    "Deepgram":       os.getenv("DEEPGRAM_API_KEY"),
    "ElevenLabs":     os.getenv("ELEVENLABS_API_KEY"),
    "Twilio SID":     os.getenv("TWILIO_ACCOUNT_SID"),
    "Vertex Project": os.getenv("VERTEX_PROJECT_ID"),
}

# ── Opening greeting — must match STATE 1 in the system prompt ──
OPENING_LINE = "Hey, this is Sam Cooper with Parametric Estimates. How are you doing today?"


# ─────────────────────────────────────────
#  BASIC ROUTES
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
#  TWILIO WEBHOOK — called when Twilio connects
# ─────────────────────────────────────────

@app.post("/incoming-call.xml")
async def incoming_call(request: Request):
    """
    Twilio calls this URL first. We return TwiML that opens a media stream
    back to our WebSocket. The AI speaks the opening line once the
    WebSocket is established — no response.say() needed here.
    """
    form_data = await request.form()
    direction = form_data.get("Direction", "")
    
    # If outbound, the customer is 'To'. If inbound, they are 'From'.
    customer_number = form_data.get("From") if direction == "inbound" else form_data.get("To", "Unknown")
    response = VoiceResponse()
    connect  = Connect()
    host     = request.headers.get("host")
    stream = connect.stream(url=f"wss://{host}/media-stream")
    stream.parameter(name="customer_number", value=customer_number)
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


# ─────────────────────────────────────────
#  DASHBOARD ROUTES
# ─────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        with open("dashboard.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found!</h1>")


@app.get("/api/calls")
async def api_get_calls():
    calls = get_all_calls()
    for call in calls:
        call['start_time'] = call['start_time'].isoformat() if call['start_time'] else None
        call['end_time']   = call['end_time'].isoformat()   if call['end_time']   else None
    return JSONResponse(content=calls)


@app.get("/api/calls/{call_id}/transcript")
async def api_get_transcript(call_id: str):
    transcript = get_transcript(call_id)
    for t in transcript:
        t['timestamp'] = t['timestamp'].isoformat() if t['timestamp'] else None
    return JSONResponse(content=transcript)


@app.post("/recording-webhook")
async def recording_webhook(request: Request):
    """Twilio posts here when the call recording MP3 is ready."""
    form_data     = await request.form()
    call_sid      = form_data.get("CallSid")
    recording_url = form_data.get("RecordingUrl")

    if call_sid and recording_url:
        mp3_url = f"{recording_url}.mp3"
        print(f"📥 Received recording from Twilio: {mp3_url}")
        update_recording_url(call_sid, mp3_url)

    return {"status": "success"}


# ─────────────────────────────────────────
#  TRIGGER CALL ENDPOINT
# ─────────────────────────────────────────

@app.post("/api/trigger-call")
async def trigger_call(request: Request):
    body   = await request.json()
    number = body.get("number", "").strip()
    if not number:
        return JSONResponse({"error": "Phone number is required"}, status_code=400)

    try:
        twilio_client = TwilioClient(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        ngrok_url = os.getenv("NGROK_URL")

        print(f"📞 Dialing: {number}")
        print(f"🔗 NGROK_URL: {ngrok_url}")
        print(f"📱 FROM: {os.getenv('TWILIO_PHONE_NUMBER')}")
        print(f"🔑 TWILIO_SID: {os.getenv('TWILIO_ACCOUNT_SID')}")

        call = twilio_client.calls.create(
            to=number,
            from_=os.getenv("TWILIO_PHONE_NUMBER"),
            url=f"{ngrok_url}/incoming-call.xml",
            method="POST",
            record=True,
            recording_status_callback=f"{ngrok_url}/recording-webhook",
            recording_status_callback_event=["completed"]
        )
        return JSONResponse({"status": "dialing", "call_sid": call.sid})

    except Exception as e:
        print(f"❌ Trigger call error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────
#  DEBUG ENDPOINT (remove before production)
# ─────────────────────────────────────────

@app.get("/debug-env")
async def debug_env():
    return {
        "NGROK_URL":           os.getenv("NGROK_URL"),
        "TWILIO_PHONE_NUMBER": os.getenv("TWILIO_PHONE_NUMBER"),
        "TWILIO_ACCOUNT_SID":  os.getenv("TWILIO_ACCOUNT_SID"),
        "TWILIO_AUTH_TOKEN":   "SET" if os.getenv("TWILIO_AUTH_TOKEN") else "MISSING",
        "VERTEX_PROJECT_ID":   os.getenv("VERTEX_PROJECT_ID"),
        "DEEPGRAM_API_KEY":    "SET" if os.getenv("DEEPGRAM_API_KEY") else "MISSING",
        "ELEVENLABS_API_KEY":  "SET" if os.getenv("ELEVENLABS_API_KEY") else "MISSING",
    }



@app.get("/api/calls/{call_id}/recording")
async def proxy_recording(call_id: str):
    """Proxies the Twilio MP3 through our server so browser can play without auth popup."""
    calls = get_all_calls()
    call  = next((c for c in calls if c['call_id'] == call_id), None)
    if not call or not call.get('recording_url'):
        return JSONResponse({"error": "No recording found"}, status_code=404)

    auth = (os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    async with httpx.AsyncClient() as client:
        r = await client.get(call['recording_url'], auth=auth, follow_redirects=True)

    return Response(
        content=r.content,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"inline; filename={call_id}.mp3"}
    )


# ─────────────────────────────────────────
#  MAIN WEBSOCKET — full conversation loop
# ─────────────────────────────────────────

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("📞 Incoming call connected! Twilio stream is open.")

    stream_sid      = None
    call_sid        = None
    customer_number = "Unknown"  # Will be overwritten when stream starts
    ai_is_speaking  = False   # Is Twilio currently playing our audio?
    is_processing   = False   # Is the LLM currently thinking?
    websocket_open  = True    # Guards against post-close sends
    call_start_time = datetime.now()

    # ── Helper: stream text → audio → Twilio ──────────────────────────
    async def speak(text: str):
        """Convert text to speech and stream every chunk to Twilio instantly."""
        nonlocal ai_is_speaking
        if not stream_sid or not websocket_open:
            return

        ai_is_speaking = True
        print(f"🔊 AI speaking: {text}")

        try:
            for audio_chunk in generate_speech_stream(text):
                if not websocket_open:
                    break
                audio_payload = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": audio_payload}
                }))

            # Mark lets Twilio tell us when it has finished playing all audio
            if websocket_open:
                await websocket.send_text(json.dumps({
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "ai_finished"}
                }))

        except Exception as e:
            print(f"⚠️  speak() error (websocket likely closed): {e}")
            ai_is_speaking = False

    # ── 1. Handle complete customer sentences ─────────────────────────
    async def handle_transcript(sentence: str):
        nonlocal ai_is_speaking, is_processing

        # Do nothing if websocket is closed or we're already busy
        if not websocket_open:
            return
        if is_processing or ai_is_speaking:
            return

        is_processing = True
        print(f"\n🗣️  Customer said: {sentence}")
        print("🧠 Passing to LLM...")

        try:
            reply_text = await generate_response(sentence)
            print(f"🤖 AI reply: {reply_text}")
            await speak(reply_text)
        except Exception as e:
            print(f"⚠️  handle_transcript() error: {e}")
        finally:
            is_processing = False

    # ── 2. Handle barge-in (customer talks while AI is speaking) ──────
    async def handle_interruption():
        nonlocal ai_is_speaking
        if not websocket_open:
            return
        if ai_is_speaking and stream_sid:
            print("🛑 Barge-in detected! Clearing Twilio audio buffer.")
            try:
                await websocket.send_text(json.dumps({
                    "event": "clear",
                    "streamSid": stream_sid
                }))
            except Exception:
                pass
            ai_is_speaking = False

    # ── 3. Connect to Deepgram ────────────────────────────────────────
    dg_connection = await setup_deepgram_connection(handle_transcript, handle_interruption)
    if not dg_connection:
        await websocket.close()
        return

    # ── 4. Main message loop ──────────────────────────────────────────
    try:
        while True:
            message = await websocket.receive_text()
            data    = json.loads(message)

            # ── START: stream is ready, send opening greeting ──
            if data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                call_sid   = data['start']['callSid']
                
                # Catch the customer number passed from the webhook
                custom_params = data['start'].get('customParameters', {})
                customer_number = custom_params.get('customer_number', 'Unknown')
                
                print(f"🔗 Stream ready: {stream_sid} for {customer_number}")

                # Add opening line to history BEFORE speaking so the LLM
                # knows STATE 1 is already done and won't repeat it
                conversation_history.append(AIMessage(content=OPENING_LINE))

                # Speak the opening greeting
                await speak(OPENING_LINE)

            # ── MEDIA: pipe customer audio to Deepgram (only when AI is silent) ──
            elif data['event'] == 'media':
                if not ai_is_speaking:
                    audio_bytes = base64.b64decode(data['media']['payload'])
                    await dg_connection.send(audio_bytes)

            # ── MARK: Twilio finished playing our audio chunk ──
            elif data['event'] == 'mark':
                if data['mark']['name'] == 'ai_finished':
                    ai_is_speaking = False
                    print("✅ AI finished speaking. Listening...")

            # ── STOP: remote party hung up ──
            elif data['event'] == 'stop':
                print("📵 Call ended by remote party.")
                break

    except Exception as e:
        print(f"⚠️  WebSocket loop error: {e}")

    finally:
        # CRITICAL: set websocket_open = False FIRST so any in-flight
        # Deepgram callbacks don't attempt to send on the closed socket
        websocket_open = False

        print("💾 Saving call to database...")
        if stream_sid:
            # Pass the caught customer_number directly to the DB save function
            save_call_data(stream_sid, call_sid, call_start_time, conversation_history, customer_number)

        try:
            await dg_connection.finish()
        except Exception:
            pass

        reset_memory()
        print("🔄 Memory reset. Ready for next call.")