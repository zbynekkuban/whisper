from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import uuid
from transcriber import Transcriber

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Transcriber instance (lazy loading handled in class)
transcriber_instances = {}

def get_transcriber(model_size):
    if model_size not in transcriber_instances:
        transcriber_instances[model_size] = Transcriber(model_size=model_size)
    return transcriber_instances[model_size]

@app.post("/transcribe")
async def transcribe_video(
    file: UploadFile = File(...),
    model_size: str = Form("tiny")
):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    video_filename = f"{file_id}_{file.filename}"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    audio_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    # Extract Audio
    transcriber = get_transcriber(model_size)
    if not transcriber.extract_audio(video_path, audio_path):
        raise HTTPException(status_code=500, detail="Audio extraction failed")

    # Transcribe
    try:
        segments, info = transcriber.transcribe(audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # Diarize
    # segments are already dicts here
    segments = transcriber.diarize(audio_path, segments)

    # Cleanup (optional, maybe keep for debugging or user wants to download?)
    # For now, let's keep them.

    return {
        "language": info.language,
        "duration": info.duration,
        "segments": segments,
        "video_url": f"/uploads/{video_filename}" 
    }

@app.post("/diarize")
async def diarize_audio(
    file: UploadFile = File(...),
    segments: str = Form(...)
):
    import json
    # Save uploaded audio
    file_id = str(uuid.uuid4())
    audio_path = os.path.join(UPLOAD_DIR, f"{file_id}_diarize.wav")

    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    try:
        segments_data = json.loads(segments)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid segments JSON: {e}")

    # Use any transcriber instance for diarization (it's stateless regarding model)
    transcriber = get_transcriber("tiny") 
    
    # Diarize
    segments_data = transcriber.diarize(audio_path, segments_data)

    # Cleanup
    # os.remove(audio_path) # Optional

    return segments_data

@app.get("/models")
def get_models():
    return ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3"]

# Serve uploads
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Serve frontend (must be last)
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
