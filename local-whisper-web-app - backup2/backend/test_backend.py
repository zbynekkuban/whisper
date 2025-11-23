import requests
import os
from moviepy import ColorClip, AudioFileClip
import numpy as np
from scipy.io import wavfile

# Create a dummy video with audio
def create_dummy_video(filename="test_video.mp4"):
    # Create a 2-second video
    # Audio: 440Hz sine wave
    duration = 2
    fps = 24
    sr = 44100
    
    t = np.linspace(0, duration, int(sr * duration))
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    wavfile.write("temp_audio.wav", sr, (audio_data * 32767).astype(np.int16))
    
    # Video: Red screen
    clip = ColorClip(size=(320, 240), color=(255, 0, 0), duration=duration)
    # Add audio
    # Note: moviepy 2.0+ might handle audio differently, but let's try standard way
    # If this fails, we will just upload a text file disguised as video to see if it fails at extraction step, 
    # but we want to test success path.
    # Let's try to just create a wav file and rename it to mp4? No, ffmpeg might complain.
    # Let's use ffmpeg directly to generate a video.
    os.system(f"ffmpeg -f lavfi -i color=c=red:s=320x240:d=2 -f lavfi -i sine=f=440:d=2 -c:v libx264 -c:a aac -shortest {filename} -y")

def test_transcription():
    video_file = "test_video.mp4"
    create_dummy_video(video_file)
    
    url = "http://127.0.0.1:8000/transcribe"
    files = {'file': open(video_file, 'rb')}
    data = {'model_size': 'tiny'}
    
    print(f"Uploading {video_file} to {url}...")
    try:
        response = requests.post(url, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:", response.json())
        else:
            print("Error:", response.text)
    except Exception as e:
        print(f"Request failed: {e}")
    finally:
        files['file'].close()
        # Cleanup
        if os.path.exists(video_file):
            os.remove(video_file)
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

if __name__ == "__main__":
    test_transcription()
