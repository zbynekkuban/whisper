import os
import logging
from faster_whisper import WhisperModel
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import librosa
from moviepy import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load_model(self):
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def extract_audio(self, video_path, audio_path):
        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            return True
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False

    def transcribe(self, audio_path):
        self.load_model()
        logger.info(f"Transcribing {audio_path}...")
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        
        result_segments = []
        for segment in segments:
            result_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": "Unknown" # Placeholder, will be updated by diarization
            })
        
        return result_segments, info

    def diarize(self, audio_path, segments, num_speakers=2):
        logger.info("Starting simple diarization (clustering)...")
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features for each segment
            segment_features = []
            valid_segments_indices = []

            for i, seg in enumerate(segments):
                # Handle both object (server-side) and dict (client-side) segments
                if isinstance(seg, dict):
                    start = seg["start"]
                    end = seg["end"]
                else:
                    start = seg.start
                    end = seg.end

                start_sample = int(start * sr)
                end_sample = int(end * sr)
                
                if end_sample > len(y):
                    end_sample = len(y)
                
                if end_sample - start_sample < 512: # Skip very short segments
                    continue

                segment_audio = y[start_sample:end_sample]
                
                # Extract MFCCs
                mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                segment_features.append(mfcc_mean)
                valid_segments_indices.append(i)

            if not segment_features:
                logger.warning("No valid segments for diarization.")
                return segments

            # Cluster
            X = np.array(segment_features)
            # If we don't know num_speakers, we could use a threshold, but for now let's assume 2 or use distance_threshold
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=50).fit(X)
            labels = clustering.labels_
            
            # Assign labels back to segments
            for idx, label in zip(valid_segments_indices, labels):
                if isinstance(segments[idx], dict):
                    segments[idx]["speaker"] = f"Speaker {label + 1}"
                else:
                    # If it's an object, we might need to set it differently or convert to dict first
                    # For consistency, let's assume we want to modify it in place if possible, 
                    # but if it's a namedtuple or similar from faster-whisper, we can't.
                    # However, in the transcribe method we already converted to dicts.
                    # So this check is mostly for safety if called from elsewhere.
                    pass 
            
            # Fill in missing speakers (too short segments) with previous speaker
            current_speaker = "Speaker 1"
            for seg in segments:
                if isinstance(seg, dict):
                    if seg.get("speaker") == "Unknown":
                        seg["speaker"] = current_speaker
                    else:
                        current_speaker = seg["speaker"]

            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return segments
