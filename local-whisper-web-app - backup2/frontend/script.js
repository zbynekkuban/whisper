import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configuration
env.allowLocalModels = false;
env.useBrowserCache = true;

// DOM Elements
const fileInput = document.getElementById('video-upload');
const dropZone = document.getElementById('drop-zone');
const videoPlayer = document.getElementById('main-video');
const uploadPrompt = document.getElementById('upload-prompt');
const backendSelect = document.getElementById('backend-select');
const modelSelect = document.getElementById('model-select');
const transcribeBtn = document.getElementById('transcribe-btn');
const saveBtn = document.getElementById('save-btn');
const transcriptOutput = document.getElementById('transcript-output');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');

let currentTranscript = [];
let transcriber = null; // WebGPU pipeline instance

// Models list
const SERVER_MODELS = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3"];
const WEBGPU_MODELS = [
    "Xenova/whisper-tiny", "Xenova/whisper-tiny.en",
    "Xenova/whisper-base", "Xenova/whisper-base.en",
    "Xenova/whisper-small", "Xenova/whisper-small.en",
    "Xenova/whisper-medium", "Xenova/whisper-medium.en",
    "Xenova/whisper-large-v2", "Xenova/whisper-large-v3",
    "Xenova/distil-whisper-small.en", "Xenova/distil-whisper-medium.en", "Xenova/distil-whisper-large-v2"
];

// Initialize
function updateModelList() {
    const backend = backendSelect.value;
    modelSelect.innerHTML = '';
    const models = backend === 'server' ? SERVER_MODELS : WEBGPU_MODELS;

    models.forEach(m => {
        const option = document.createElement('option');
        option.value = m;
        option.textContent = m;
        modelSelect.appendChild(option);
    });
}

backendSelect.addEventListener('change', updateModelList);
updateModelList();

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('video/')) {
        alert("Please upload a video file.");
        return;
    }

    const url = URL.createObjectURL(file);
    videoPlayer.src = url;
    videoPlayer.style.display = 'block';
    uploadPrompt.style.display = 'none';
    transcribeBtn.disabled = false;

    // Store file for transcription
    fileInput.files = createFileList(file);
}

// Helper to set file input files programmatically
function createFileList(file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    return dt.files;
}

// Transcribe
transcribeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    transcribeBtn.disabled = true;
    saveBtn.disabled = true;
    transcriptOutput.innerHTML = '';

    const backend = backendSelect.value;

    if (backend === 'server') {
        await transcribeServer(file);
    } else {
        await transcribeWebGPU(file);
    }
});

async function transcribeServer(file) {
    loadingOverlay.classList.remove('hidden');
    loadingText.textContent = "Uploading & Transcribing...";

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_size', modelSelect.value);

    try {
        const response = await fetch('http://127.0.0.1:8000/transcribe', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Transcription failed');

        const data = await response.json();
        currentTranscript = data.segments;
        renderTranscript(data.segments);
        saveBtn.disabled = false;
    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        loadingOverlay.classList.add('hidden');
        transcribeBtn.disabled = false;
    }
}

async function transcribeWebGPU(file) {
    progressContainer.classList.remove('hidden');
    loadingText.textContent = "Loading Model (this may take a while first time)...";

    try {
        // 1. Extract Audio
        const audioData = await extractAudio(file);

        // 2. Load Model & Transcribe
        const modelName = modelSelect.value;
        if (!transcriber || transcriber.modelName !== modelName) {
            transcriber = await pipeline('automatic-speech-recognition', modelName, {
                progress_callback: (data) => {
                    if (data.status === 'progress') {
                        const percent = Math.round(data.progress * 100);
                        progressFill.style.width = `${percent}%`;
                        progressText.textContent = `${percent}%`;
                    }
                }
            });
            transcriber.modelName = modelName;
        }

        loadingText.textContent = "Transcribing...";
        progressFill.style.width = '0%';
        progressText.textContent = 'Running...';

        const output = await transcriber(audioData, {
            chunk_length_s: 30,
            stride_length_s: 5,
            return_timestamps: true
        });

        // Convert output to our format
        let segments = output.chunks.map(chunk => ({
            start: chunk.timestamp[0],
            end: chunk.timestamp[1] || chunk.timestamp[0] + 1, // Handle null end
            text: chunk.text,
            speaker: "Unknown"
        }));

        // 3. Diarize (Hybrid)
        loadingText.textContent = "Diarizing (Server)...";
        // We need to send the audio file and segments to server
        // Since we already have the file object, we can send it directly
        // But we need to send segments as JSON string

        const formData = new FormData();
        formData.append('file', file);
        formData.append('segments', JSON.stringify(segments));

        const response = await fetch('http://127.0.0.1:8000/diarize', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            segments = await response.json();
        } else {
            console.warn("Diarization failed, showing raw transcript");
        }

        currentTranscript = segments;
        renderTranscript(segments);
        saveBtn.disabled = false;

    } catch (e) {
        console.error(e);
        alert(`WebGPU Error: ${e.message}`);
    } finally {
        progressContainer.classList.add('hidden');
        loadingOverlay.classList.add('hidden');
        transcribeBtn.disabled = false;
    }
}

// Audio Extraction Helper
async function extractAudio(file) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer.getChannelData(0); // Mono
}

function formatTime(seconds) {
    if (seconds === null || seconds === undefined) return "00:00";
    const date = new Date(seconds * 1000);
    const hh = date.getUTCHours();
    const mm = date.getUTCMinutes();
    const ss = date.getUTCSeconds();
    if (hh) {
        return `${hh}:${mm.toString().padStart(2, '0')}:${ss.toString().padStart(2, '0')}`;
    }
    return `${mm}:${ss.toString().padStart(2, '0')}`;
}

function renderTranscript(segments) {
    transcriptOutput.innerHTML = '';

    segments.forEach(segment => {
        const div = document.createElement('div');
        div.className = 'transcript-segment';
        div.onclick = () => {
            videoPlayer.currentTime = segment.start;
            videoPlayer.play();
        };

        const speakerClass = `speaker-${segment.speaker.replace(/\s+/g, '.')}`;

        div.innerHTML = `
            <div class="segment-meta">
                <span class="speaker ${speakerClass}">${segment.speaker}</span>
                <span class="timestamp">${formatTime(segment.start)} - ${formatTime(segment.end)}</span>
            </div>
            <div class="segment-text">${segment.text}</div>
        `;
        transcriptOutput.appendChild(div);
    });
}

saveBtn.addEventListener('click', () => {
    if (currentTranscript.length === 0) return;

    let textContent = "";
    currentTranscript.forEach(seg => {
        textContent += `[${formatTime(seg.start)} - ${formatTime(seg.end)}] ${seg.speaker}: ${seg.text}\n`;
    });

    const blob = new Blob([textContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'transcript.txt';
    a.click();
    URL.revokeObjectURL(url);
});
