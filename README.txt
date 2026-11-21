# Reel Studio AI — Module 1 (Windows)

Local-first video ingestion, transcription, transcript correction and silence analysis.

## Requirements
- Windows 10/11
- Python 3.11 (64-bit)
- Internet for first-time package/model downloads
- 8 GB RAM minimum; 16 GB recommended
- NVIDIA GPU is optional. CPU mode is the compatibility fallback.

## Install and run
1. Install Python 3.11 (64-bit) from https://www.python.org/downloads/
   During setup, enable **Add Python to PATH**.
2. Extract this ZIP to a normal folder, for example `C:\ReelStudioAI`.
3. Double-click `run_windows.bat`.
4. Wait while dependencies install. The first run may take several minutes.
5. Streamlit should open the app in your browser at `http://localhost:8501`.
6. Upload an MP4/MOV, choose Kokni or Hindi, start with the `small` model and choose `Auto (GPU if available)`.
7. Open **My projects** to review/correct the transcript and export SRT/TXT/JSON.

## GPU note
The app attempts NVIDIA CUDA with FP16 in Auto mode, then falls back to CPU INT8 if GPU initialisation fails. Current faster-whisper GPU builds require compatible NVIDIA CUDA 12 cuBLAS and cuDNN 9 libraries. NVIDIA GPU presence alone does not guarantee those libraries are installed. If GPU mode fails, choose CPU to continue. See:
https://github.com/SYSTRAN/faster-whisper

## Project storage
All project data is stored under `projects\<project_id>\`:
- original uploaded video
- extracted mono 16 kHz WAV
- `project.json`
- `transcript.srt`
- `transcript.txt`

Back up the `projects` folder if you want to preserve work. Do not delete project folders while the app is running.

## Limitations in Module 1
- No automatic cuts or final MP4 rendering yet.
- Silence intervals are approximate suggestions, not automatic edits.
- Kokni is dialectal and code-switched; speech recognition can make mistakes. Review spellings, names and phrases.
- Word-level timestamps are retained from the transcription engine, but the editable table currently edits sentence/segment-level text and timing.

## Troubleshooting
- Python not found: install Python 3.11 64-bit and reopen the terminal.
- GPU error: choose CPU in Processing device.
- Upload too large: copy a smaller test clip or increase Streamlit's server max upload size in a future configuration.
- Browser page not opening: manually visit http://localhost:8501 after the terminal says the app is running.
