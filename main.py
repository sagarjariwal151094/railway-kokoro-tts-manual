import os
import io
import base64
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from kokoro import KPipeline
import soundfile as sf

app = FastAPI()

# 1. API Key Authentication Layer
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
SECRET_TOKEN = os.environ.get("API_KEY", "your_fallback_token")

def get_api_key(header_value: str = Depends(api_key_header)):
    if not header_value:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")
    # Clean up "Bearer <token>" formatting if sent that way
    token = header_value.replace("Bearer ", "").strip()
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API Token")
    return token

# 2. Initialize the Kokoro Pipeline (Downloads weights dynamically on cold start)
# 'a' stands for American English; adjust to 'b' for British if desired
pipeline = KPipeline(lang_code='a')

class TTSRequest(BaseModel):
    input: str
    voice: str = "af_bella"
    speed: float = 1.0

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest, token: str = Depends(get_api_key)):
    try:
        # Generate generator stream from Kokoro pipeline
        generator = pipeline(request.input, voice=request.voice, speed=request.speed, split_pattern=r'\n+')
        
        # Combine fragments into a single audio structure
        all_audio = []
        for _, _, audio in generator:
            all_audio.append(audio)
            
        if not all_audio:
            raise HTTPException(status_code=400, detail="Could not synthesize text sequence.")
            
        # Concatenate audio chunks safely
        import numpy as np
        final_audio = np.concatenate(all_audio)

        # Write audio track data structure directly to an in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, 24000, format='WAV') # Native Kokoro output is 24kHz
        buffer.seek(0)
        
        # Return binary response stream
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
