import os
import asyncio
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

load_dotenv()

config = DeepgramClientOptions(options={"keepalive": "true"})
deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)

# NOTICE: We added a new parameter -> on_interruption_callback
async def setup_deepgram_connection(on_transcript_callback, on_interruption_callback):
    try:
        dg_connection = deepgram.listen.asynclive.v("1")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            
            # If the user finished their sentence, send to LLM
            if result.is_final:
                print(f"🗣️ Customer said: {sentence}")
                await on_transcript_callback(sentence)
            else:
                # If it's NOT final, it means the user just started making noise!
                # We trigger the interruption callback instantly.
                await on_interruption_callback()

        async def on_error(self, error, **kwargs):
            print(f"Deepgram Error: {error}")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            encoding="mulaw", 
            sample_rate=8000, 
            endpointing=500,  
            smart_format=True,
            interim_results=True, # <--- CRITICAL FOR BARGE-IN: Tells Deepgram to send partial audio
        )
        
        if await dg_connection.start(options) is False:
            print("Failed to connect to Deepgram")
            return None
            
        print("✅ Deepgram Ear is online and listening!")
        return dg_connection

    except Exception as e:
        print(f"Could not open Deepgram connection: {e}")
        return None