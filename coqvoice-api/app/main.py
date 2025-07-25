import base64
import io
import os
import soundfile as sf
import torch
import torchaudio.transforms as T
import asyncio
import json
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import torchaudio

# Import LLaMA-Omni specific components from the cloned repository
from omni_speech.model.builder import load_model as load_llama_omni_model
from omni_speech.common.config import Config

# For VAD
import silero_vad

# For vocoder components from fairseq
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
from fairseq import checkpoint_utils

# Initialize FastAPI app
app = FastAPI(
    title="CoqVoice Real-time API",
    description="API for seamless real-time speech interaction with LLaMA-Omni model via WebSockets, rebranded as CoqVoice.",
    version="1.0.0"
)

# Global variables to hold the loaded models
llama_omni_model = None
vocoder_model = None
vocoder_cfg = None
vad_model = None
vad_utils = None

# Constants for audio processing
TARGET_SAMPLE_RATE = 16000 # LLaMA-Omni's expected sample rate
AUDIO_CHUNK_SIZE_MS = 50 # Process audio in 50ms chunks for VAD
AUDIO_CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * (AUDIO_CHUNK_SIZE_MS / 1000))
SPEECH_PAD_MS = 200 # Milliseconds of silence to append after speech for VAD to detect end of utterance
SPEECH_PAD_SAMPLES = int(TARGET_SAMPLE_RATE * (SPEECH_PAD_MS / 1000))

class LlamaOmniArgs:
    """
    A dummy arguments object to mimic the command-line arguments structure
    expected by `omni_speech.model.builder.load_model`.
    This allows us to configure the model loading programmatically.
    """
    def __init__(self):
        self.model_path = os.environ.get("LLAMA_OMNI_MODEL_NAME", "ictnlp/LLaMA-Omni")
        self.model_name = self.model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing LlamaOmniArgs with device: {self.device}")
        self.s2s = True
        self.num_gpus = 1
        self.max_gpu_memory = None
        self.load_8bit = False
        self.cpu_offloading = False
        self.debug = False
        self.dtype = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"

        self.cfg = Config()
        self.cfg.model = Config()
        self.cfg.model.arch = "llama_omni"
        self.cfg.model.model_type = "llama_omni"
        self.cfg.model.llama_model = self.model_path
        self.cfg.model.whisper_model = "large-v3"
        self.cfg.model.whisper_model_path = os.environ.get("WHISPER_MODEL_DIR")
        self.cfg.model.vocoder_path = os.environ.get("VOCODER_MODEL_PATH")
        self.cfg.model.vocoder_config_path = os.environ.get("VOCODER_CONFIG_PATH")
        self.cfg.model.max_txt_len = 512
        self.cfg.model.max_output_len = 512
        self.cfg.model.use_flash_attention = True
        self.cfg.model.freeze_llm = False
        self.cfg.model.freeze_speech_encoder = False

        self.cfg.processor = Config()
        self.cfg.processor.audio_processor = Config()
        self.cfg.processor.audio_processor.name = "whisper_processor"
        self.cfg.processor.audio_processor.model_type = "large-v3"
        self.cfg.processor.audio_processor.model_path = os.environ.get("WHISPER_MODEL_DIR")
        self.cfg.processor.text_processor = Config()
        self.cfg.processor.text_processor.name = "llama_processor"

# FastAPI startup event: Load all models when the application starts
@app.on_event("startup")
async def load_models():
    global llama_omni_model, vocoder_model, vocoder_cfg, vad_model, vad_utils

    print("Starting CoqVoice model loading process... This may take a while.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Models will be loaded to device: {device}")

    try:
        # 0. Load Silero VAD Model
        print("Loading Silero VAD model...")
        vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False,
                                              onnx=True) # Use ONNX for better performance
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = vad_utils
        vad_model = vad_model.to(device)
        print("Silero VAD model loaded successfully for CoqVoice.")

        # 1. Load the HiFi-GAN Vocoder
        print("Loading HiFi-GAN vocoder for CoqVoice...")
        vocoder_path = os.environ.get("VOCODER_MODEL_PATH")
        vocoder_config_path = os.environ.get("VOCODER_CONFIG_PATH")

        if not os.path.exists(vocoder_path) or not os.path.exists(vocoder_config_path):
             raise FileNotFoundError(
                 f"Vocoder files not found. Check paths: {vocoder_path} and {vocoder_config_path}. "
                 "Ensure they were downloaded correctly in the Dockerfile."
             )

        with open(vocoder_config_path, 'r') as f:
            vocoder_config_dict = json.load(f)
        vocoder_cfg = Config(vocoder_config_dict)

        vocoder_model = CodeHiFiGANVocoder(vocoder_cfg)
        vocoder_model.eval()
        vocoder_model.load_state_dict(
            checkpoint_utils.load_checkpoint_to_cpu(vocoder_path)["model"]
        )
        vocoder_model = vocoder_model.to(device)
        print("HiFi-GAN Vocoder loaded successfully for CoqVoice.")

        # 2. Load the LLaMA-Omni model (used by CoqVoice)
        print(f"Loading LLaMA-Omni model from Hugging Face: {os.environ.get('LLAMA_OMNI_MODEL_NAME')}...")
        args = LlamaOmniArgs()

        # Pass the loaded vocoder and its config to the LLaMA-Omni model's configuration.
        args.cfg.model.vocoder_model = vocoder_model
        args.cfg.model.vocoder_cfg = vocoder_cfg

        llama_omni_model = load_llama_omni_model(args)
        llama_omni_model.eval()
        llama_omni_model = llama_omni_model.to(device)
        print("LLaMA-Omni model loaded successfully for CoqVoice.")

        print("All CoqVoice models initialized and loaded successfully. API is ready.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: Failed to load one or more CoqVoice models: {e}")
        raise RuntimeError(f"Failed to load CoqVoice models during startup: {e}")

# Health check endpoint to verify model readiness
@app.get("/health")
async def health_check():
    if llama_omni_model and vocoder_model:
        return {"status": "ok", "message": "All models loaded and ready."}
    return {"status": "error", "message": "Models not loaded yet or failed to load. Check server logs."}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("CoqVoice WebSocket connection accepted.")

    if not llama_omni_model or not vocoder_model:
        await websocket.send_json({"error": "Models are still loading or failed to load. Please try again later."})
        await websocket.close()
        return

    # Initialize audio buffer and text buffer for the conversation
    # Initialize audio buffer and text buffer for the conversation
    # Use a list to store speech chunks for VAD processing
    speech_chunks = []
    current_prompt = ""
    # Initialize VAD iterator for this connection
    vad_iterator = vad_utils[3](vad_model, threshold=0.7, sampling_rate=TARGET_SAMPLE_RATE) # VADIterator

    try:
        while True:
            # Receive data from the client
            data = await websocket.receive_bytes()

            # Convert bytes to float32 numpy array, then to torch tensor
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            audio_chunk_tensor = torch.from_numpy(audio_chunk).to(llama_omni_model.device)

            # Process audio chunk with VAD
            # Silero VAD expects 16kHz float32 audio
            speech_prob = vad_iterator(audio_chunk_tensor, return_seconds=True)

            if speech_prob is not None:
                if speech_prob['start'] is not None:
                    print(f"Speech start detected at {speech_prob['start']:.2f}s")
                    # Clear buffer and start new speech segment
                    speech_chunks = []
                if speech_prob['end'] is not None:
                    print(f"Speech end detected at {speech_prob['end']:.2f}s")
                    # Append remaining audio in buffer to speech_chunks
                    speech_chunks.append(audio_chunk_tensor)
                    # Process the collected speech
                    if speech_chunks:
                        full_speech_segment = torch.cat(speech_chunks)
                        print(f"Processing speech segment of length: {full_speech_segment.shape[0] / TARGET_SAMPLE_RATE:.2f}s")
                        await process_speech_segment(websocket, full_speech_segment, current_prompt)
                        speech_chunks = [] # Reset for next utterance
                        current_prompt = "" # Reset prompt for new utterance
                else:
                    # If speech is ongoing, append to buffer
                    speech_chunks.append(audio_chunk_tensor)
            else:
                # If no speech detected, and we have buffered speech, it means silence after speech
                if speech_chunks:
                    # Add a small padding of silence to ensure VAD detects end of utterance
                    # This is a simplified approach; a more robust VAD would handle this internally
                    # or use a more sophisticated silence detection.
                    # For now, we'll just process what we have if no speech is detected for a while.
                    pass # VADIterator handles silence internally to some extent

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WebSocket error: {e}")
        await websocket.send_json({"error": f"Server error: {e}"})
    finally:
        await websocket.close()

async def process_speech_segment(websocket: WebSocket, speech_tensor: torch.Tensor, prompt: str):
    """Processes a detected speech segment with LLaMA-Omni and sends response."""
    global current_prompt # Allow modification of global prompt

    samples = {
        "audio": [speech_tensor],
        "text_input": [prompt],
    }

    print(f"Sending speech segment to CoqVoice (LLaMA-Omni). Prompt: '{prompt}'")
    output = llama_omni_model.generate(
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_new_tokens=100,
        min_new_tokens=1,
        length_penalty=1.0,
        repetition_penalty=1.0
    )

    if output and len(output) > 0:
        text_response = output[0].get("text", "")
        audio_features = output[0].get("audio_features")

        current_prompt = text_response # Update prompt for next turn

        speech_base64 = ""
        if audio_features is not None:
            audio_features = audio_features.to(vocoder_model.device)
            generated_speech_waveform = vocoder_model(audio_features).squeeze().cpu().numpy()

            output_audio_buffer = io.BytesIO()
            sf.write(output_audio_buffer, generated_speech_waveform, vocoder_cfg.sample_rate, format='WAV')
            speech_base64 = base64.b64encode(output_audio_buffer.getvalue()).decode('utf-8')

        response_data = {
            "text": text_response,
            "audio_base64": speech_base64
        }
        await websocket.send_json(response_data)
        print(f"Sent response: Text='{text_response[:50]}...', Audio present: {bool(speech_base64)}")
    else:
        print("CoqVoice (LLaMA-Omni) generated no output for this segment.")
        await websocket.send_json({"text": "", "audio_base64": ""}) # Send empty response
