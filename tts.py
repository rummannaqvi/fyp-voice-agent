import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def generate_speech_stream(text: str):
    """Yields audio chunks as soon as ElevenLabs generates them."""
    try:
        # Generate the audio stream
        audio_stream = client.generate(
            text=text,
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            model="eleven_turbo_v2",
            output_format="ulaw_8000",
            stream=True  # <--- CRITICAL: Forces ElevenLabs to stream!
        )
        
        # Yield the chunks instantly instead of waiting for the whole file
        for chunk in audio_stream:
            if chunk:
                yield chunk
                
    except Exception as e:
        print(f"ElevenLabs TTS Error: {e}")