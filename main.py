import os
import io
import sys

# 1. Force the writable temporary volume for huggingface & model data download layers
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["XDG_CACHE_HOME"] = "/tmp/cache"

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import soundfile as sf
import numpy as np

# Lazy initialize pipeline to allow the environment variables to set firmly first
pipeline = None

app = FastAPI(title="Railway Manual Kokoro API Engine")

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
SECRET_TOKEN = os.environ.get("API_KEY", "your_fallback_token")

def get_api_key(header_value: str = Depends(api_key_header)):
    if not header_value:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")
    token = header_value.replace("Bearer ", "").strip()
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API Token")
    return token

def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            from kokoro import KPipeline
            # Downloads ~350MB model weights file into /tmp/ dynamically on first call
            pipeline = KPipeline(lang_code='a')
        except Exception as e:
            raise RuntimeError(f"Failed initializing internal KPipeline backend: {str(e)}")
    return pipeline

class TTSRequest(BaseModel):
    input: str
    voice: str = "af_bella"
    speed: float = 1.0

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "kokoro-tts"}

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest, token: str = Depends(get_api_key)):
    try:
        tts_pipeline = get_pipeline()
        generator = tts_pipeline(request.input, voice=request.voice, speed=request.speed, split_pattern=r'\n+')
        
        all_audio = []
        for _, _, audio in generator:
            if audio is not None:
                all_audio.append(audio)
            
        if not all_audio:
            raise HTTPException(status_code=400, detail="Synthesizer processing returned empty data output stream.")
            
        final_audio = np.concatenate(all_audio)

        # Write WAV stream natively into memory
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, 24000, format='WAV')
        buffer.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
