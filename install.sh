#!/bin/bash

# CoqVoice Deployment Wizard with Automated GCS Integration
# This script automates the setup of CoqVoice with FastAPI and Docker on a Linux server,
# including automated GCS bucket, service account, and model weight upload.
# It assumes an Ubuntu/Debian-based system with root/sudo privileges.

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Global Variables ---
PROJECT_DIR="coqvoice-api"
DOCKER_IMAGE_NAME="coqvoice-api"
DOCKER_CONTAINER_NAME="coqvoice-app"
API_PORT=8000
INSTALL_LOG="coqvoice_install.log"
INSTALL_REPORT="coqvoice_install_report.txt"

GCS_ENABLED="no"
GCP_PROJECT_ID=""
GCS_BUCKET_NAME=""
GCS_SERVICE_ACCOUNT_NAME="coqvoice-sa"
GCS_KEY_FILE_NAME="key.json"
GCP_KEY_DIR="$PROJECT_DIR/.gcp" # Directory to store key on host, relative to script run location
GCS_KEY_MOUNT_PATH="/app/.gcp/$GCS_KEY_FILE_NAME" # Path inside container

WHISPER_MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/whisper-large-v3.pt"
VOCODER_MODEL_URL="https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000"
VOCODER_CONFIG_URL="https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json"
TEMP_MODEL_DIR="/tmp/llama_omni_models_download"

# --- Functions ---

log_message() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$INSTALL_LOG"
}

print_header() {
    echo -e "${BLUE}=====================================================${NC}"
    echo -e "${BLUE}  CoqVoice Real-time API Deployment Wizard           ${NC}"
    echo -e "${BLUE}  (Automated Google Cloud Storage Integration)       ${NC}"
    echo -e "${BLUE}=====================================================${NC}"
    echo ""
}

print_step() {
    echo -e "\n${CYAN}--- STEP $(($STEP_COUNTER++)): $1 ---${NC}"
    log_message "STEP $(($STEP_COUNTER-1)): $1"
}

print_success() {
    echo -e "${GREEN}✔ SUCCESS: $1${NC}"
    log_message "SUCCESS: $1"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
    log_message "WARNING: $1"
}

print_error() {
    echo -e "${RED}✖ ERROR: $1${NC}"
    log_message "ERROR: $1"
    EXIT_STATUS=1
}

ask_yes_no() {
    while true; do
        read -p "$(echo -e "${YELLOW}$1 (y/n): ${NC}")" yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo -e "${RED}Please answer yes or no.${NC}";;
        esac
    done
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed."
        return 1
    fi
    return 0
}

check_sudo() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script needs to be run with root privileges (sudo)."
        print_error "Please run: ${YELLOW}sudo ./$(basename "$0")${NC}"
        exit 1
    fi
    print_success "Sudo privileges confirmed."
}

install_gcloud_cli() {
    print_step "Checking and installing Google Cloud SDK (gcloud CLI)..."
    if check_command "gcloud"; then
        print_success "gcloud CLI is already installed."
        return 0
    fi

    print_warning "gcloud CLI not found. Attempting to install."
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
    sudo apt-get update && sudo apt-get install -y google-cloud-sdk
    if [ $? -eq 0 ]; then
        print_success "Google Cloud SDK installed successfully."
    else
        print_error "Failed to install Google Cloud SDK. Please install it manually and re-run."
        return 1
    fi
    return 0
}

gcloud_login() {
    print_step "Authenticating with Google Cloud"
    echo -e "${MAGENTA}You need to log in to your Google Cloud account.${NC}"
    echo -e "${MAGENTA}A browser window will open, or you'll get a URL to copy.${NC}"
    echo -e "${MAGENTA}After authentication, paste the verification code back into the terminal.${NC}"
    echo ""
    if gcloud auth login --no-launch-browser --enable-gdrive-access; then
        print_success "Successfully logged in to gcloud."
        return 0
    else
        print_error "Failed to log in to gcloud. Please ensure your account has necessary permissions and try again."
        return 1
    fi
}

check_docker() {
    print_step "Checking Docker installation..."
    if ! check_command "docker"; then
        print_warning "Docker not found. Attempting to install Docker."
        install_docker
    else
        print_success "Docker is installed."
    fi

    if ! docker info &> /dev/null; then
        print_warning "Docker daemon might not be running or current user ($USER) is not in 'docker' group."
        print_warning "Attempting to add current user to 'docker' group. You might need to log out and back in for changes to take effect."
        sudo usermod -aG docker "$USER"
        print_warning "Added $USER to 'docker' group. Please log out and back in, then re-run this script to continue."
        print_error "Docker is not fully accessible. Exiting."
        exit 1
    else
        print_success "Docker is running and accessible."
    fi
}

install_docker() {
    print_step "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    if [ $? -eq 0 ]; then
        print_success "Docker installed successfully."
    else
        print_error "Failed to install Docker."
        return 1
    fi
    return 0
}

check_nvidia_docker() {
    print_step "Checking NVIDIA Container Toolkit (nvidia-docker2) installation..."
    if ! dpkg -s nvidia-container-toolkit &> /dev/null; then
        print_warning "NVIDIA Container Toolkit not found. Attempting to install."
        install_nvidia_docker
    else
        print_success "NVIDIA Container Toolkit is installed."
    fi

    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_error "Docker cannot access NVIDIA GPUs. Ensure NVIDIA drivers are installed and nvidia-container-toolkit is configured correctly."
        print_error "You might need to reboot your system after installing drivers or toolkit."
        EXIT_STATUS=1
        return 1
    else
        print_success "Docker can access NVIDIA GPUs."
    fi
    return 0
}

install_nvidia_docker() {
    print_step "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-container-toolkit/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-container-toolkit/$distribution/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    if [ $? -eq 0 ]; then
        print_success "NVIDIA Container Toolkit installed and Docker restarted."
    else
        print_error "Failed to install NVIDIA Container Toolkit."
        return 1
    fi
    return 0
}

display_hardware_info() {
    print_step "Displaying Hardware Information (for reference)"
    echo -e "${YELLOW}--- CPU Info ---${NC}"
    lscpu | grep "Model name\|CPU(s)\|MHz"
    echo -e "${YELLOW}--- Memory Info ---${NC}"
    free -h
    echo -e "${YELLOW}--- GPU Info (if available) ---${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        echo "NVIDIA GPU drivers (nvidia-smi) not found. GPU acceleration might not be available."
    fi
    echo -e "${YELLOW}----------------------------------${NC}"
    print_warning "CoqVoice (LLaMA-Omni 8B model) requires significant GPU VRAM (>= 16GB recommended for optimal performance)."
    print_warning "Ensure your cloud instance has adequate resources."
}

configure_gcs_automated() {
    print_step "Automated Google Cloud Storage (GCS) Integration Setup"
    echo ""
    if ask_yes_no "Do you want to use automated GCS integration for model weights?"; then
        GCS_ENABLED="yes"
        echo -e "${MAGENTA}This will automatically create a GCS bucket (if it doesn't exist),${NC}"
        echo -e "${MAGENTA}create a Service Account, and upload necessary model weights to your bucket.${NC}"
        echo -e "${MAGENTA}Your logged-in Google account needs permissions for these operations.${NC}"
        echo ""

        read -p "$(echo -e "${YELLOW}Enter your Google Cloud Project ID (e.g., my-gcp-project-12345): ${NC}")" GCP_PROJECT_ID
        if [[ -z "$GCP_PROJECT_ID" ]]; then
            print_error "Google Cloud Project ID cannot be empty. Disabling GCS integration."
            GCS_ENABLED="no"
            return 1
        fi
        gcloud config set project "$GCP_PROJECT_ID" >> "$INSTALL_LOG" 2>&1
        print_success "Set gcloud project to $GCP_PROJECT_ID."

        read -p "$(echo -e "${YELLOW}Enter your desired GCS Bucket Name for models (e.g., llama-omni-models-unique-id): ${NC}")" GCS_BUCKET_NAME
        if [[ -z "$GCS_BUCKET_NAME" ]]; then
            print_error "GCS Bucket Name cannot be empty. Disabling GCS integration."
            GCS_ENABLED="no"
            return 1
        fi

        # Create GCS bucket if it doesn't exist
        print_step "Creating GCS bucket '$GCS_BUCKET_NAME' (if it doesn't exist)..."
        if ! gsutil ls "gs://$GCS_BUCKET_NAME" >> "$INSTALL_LOG" 2>&1; then
            gsutil mb "gs://$GCS_BUCKET_NAME" >> "$INSTALL_LOG" 2>&1
            if [ $? -eq 0 ]; then
                print_success "GCS bucket '$GCS_BUCKET_NAME' created."
            else
                print_error "Failed to create GCS bucket '$GCS_BUCKET_NAME'. Check permissions."
                GCS_ENABLED="no"
                return 1
            fi
        else
            print_success "GCS bucket '$GCS_BUCKET_NAME' already exists."
        fi

        # Create or get Service Account
        print_step "Creating/Verifying GCS Service Account '$GCS_SERVICE_ACCOUNT_NAME'..."
        SERVICE_ACCOUNT_EMAIL="$GCS_SERVICE_ACCOUNT_NAME@$GCP_PROJECT_ID.iam.gserviceaccount.com"
        if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" --project "$GCP_PROJECT_ID" >> "$INSTALL_LOG" 2>&1; then
            gcloud iam service-accounts create "$GCS_SERVICE_ACCOUNT_NAME" \
                --display-name="CoqVoice Service Account" \
                --project "$GCP_PROJECT_ID" >> "$INSTALL_LOG" 2>&1
            if [ $? -eq 0 ]; then
                print_success "Service Account '$SERVICE_ACCOUNT_EMAIL' created."
            else
                print_error "Failed to create Service Account. Check permissions."
                GCS_ENABLED="no"
                return 1
            fi
        else
            print_success "Service Account '$SERVICE_ACCOUNT_EMAIL' already exists."
        fi

        # Grant permissions to Service Account on the bucket
        print_step "Granting 'Storage Object Admin' role to Service Account on bucket..."
        gsutil iam ch "serviceAccount:$SERVICE_ACCOUNT_EMAIL:objectAdmin" "gs://$GCS_BUCKET_NAME" >> "$INSTALL_LOG" 2>&1
        if [ $? -eq 0 ]; then
            print_success "Granted 'Storage Object Admin' to '$SERVICE_ACCOUNT_EMAIL' on '$GCS_BUCKET_NAME'."
        else
            print_error "Failed to grant permissions to Service Account. Check your IAM permissions."
            GCS_ENABLED="no"
            return 1
        fi

        # Generate and download Service Account key
        print_step "Generating JSON key for Service Account..."
        mkdir -p "$GCP_KEY_DIR" || { print_error "Failed to create directory $GCP_KEY_DIR"; GCS_ENABLED="no"; return 1; }
        gcloud iam service-accounts keys create "$GCP_KEY_DIR/$GCS_KEY_FILE_NAME" \
            --iam-account="$SERVICE_ACCOUNT_EMAIL" \
            --project "$GCP_PROJECT_ID" >> "$INSTALL_LOG" 2>&1
        if [ $? -eq 0 ]; then
            chmod 600 "$GCP_KEY_DIR/$GCS_KEY_FILE_NAME" # Restrict permissions
            print_success "Service Account key generated and saved to '$GCP_KEY_DIR/$GCS_KEY_FILE_NAME'."
        else
            print_error "Failed to generate Service Account key. Check permissions."
            GCS_ENABLED="no"
            return 1
        fi

        # Download models locally then upload to GCS
        print_step "Downloading models locally and uploading to GCS bucket..."
        mkdir -p "$TEMP_MODEL_DIR" || { print_error "Failed to create temporary model directory."; GCS_ENABLED="no"; return 1; }

        # Whisper Model
        echo "Downloading Whisper model from $WHISPER_MODEL_URL..." | tee -a "$INSTALL_LOG"
        wget -q -O "$TEMP_MODEL_DIR/whisper-large-v3.pt" "$WHISPER_MODEL_URL" >> "$INSTALL_LOG" 2>&1
        if [ $? -eq 0 ]; then
            print_success "Whisper model downloaded."
            echo "Uploading Whisper model to gs://$GCS_BUCKET_NAME/models/speech_encoder/whisper-large-v3.pt..." | tee -a "$INSTALL_LOG"
            gsutil cp "$TEMP_MODEL_DIR/whisper-large-v3.pt" "gs://$GCS_BUCKET_NAME/models/speech_encoder/whisper-large-v3.pt" >> "$INSTALL_LOG" 2>&1
            if [ $? -eq 0 ]; then print_success "Whisper model uploaded to GCS."; else print_error "Failed to upload Whisper model to GCS."; GCS_ENABLED="no"; return 1; fi
        else
            print_error "Failed to download Whisper model from public source. Check URL and network."
            GCS_ENABLED="no"
            return 1
        fi

        # Vocoder Model
        echo "Downloading Vocoder model from $VOCODER_MODEL_URL..." | tee -a "$INSTALL_LOG"
        wget -q -O "$TEMP_MODEL_DIR/g_00500000" "$VOCODER_MODEL_URL" >> "$INSTALL_LOG" 2>&1
        if [ $? -eq 0 ]; then
            print_success "Vocoder model downloaded."
            echo "Uploading Vocoder model to gs://$GCS_BUCKET_NAME/models/vocoder/g_00500000..." | tee -a "$INSTALL_LOG"
            gsutil cp "$TEMP_MODEL_DIR/g_00500000" "gs://$GCS_BUCKET_NAME/models/vocoder/g_00500000" >> "$INSTALL_LOG" 2>&1
            if [ $? -eq 0 ]; then print_success "Vocoder model uploaded to GCS."; else print_error "Failed to upload Vocoder model to GCS."; GCS_ENABLED="no"; return 1; fi
        else
            print_error "Failed to download Vocoder model from public source. Check URL and network."
            GCS_ENABLED="no"
            return 1
        fi

        # Vocoder Config
        echo "Downloading Vocoder config from $VOCODER_CONFIG_URL..." | tee -a "$INSTALL_LOG"
        wget -q -O "$TEMP_MODEL_DIR/config.json" "$VOCODER_CONFIG_URL" >> "$INSTALL_LOG" 2>&1
        if [ $? -eq 0 ]; then
            print_success "Vocoder config downloaded."
            echo "Uploading Vocoder config to gs://$GCS_BUCKET_NAME/models/vocoder/config.json..." | tee -a "$INSTALL_LOG"
            gsutil cp "$TEMP_MODEL_DIR/config.json" "gs://$GCS_BUCKET_NAME/models/vocoder/config.json" >> "$INSTALL_LOG" 2>&1
            if [ $? -eq 0 ]; then print_success "Vocoder config uploaded to GCS."; else print_error "Failed to upload Vocoder config to GCS."; GCS_ENABLED="no"; return 1; fi
        else
            print_error "Failed to download Vocoder config from public source. Check URL and network."
            GCS_ENABLED="no"
            return 1
        fi

        # Clean up temporary models
        print_step "Cleaning up temporary local model downloads..."
        rm -rf "$TEMP_MODEL_DIR" >> "$INSTALL_LOG" 2>&1
        print_success "Temporary models cleaned up."

    else
            print_warning "Automated GCS integration skipped. Models will be downloaded by the Dockerfile from public sources if available. (Note: LLaMA-Omni models are still referred to by their original names.)"
    fi
    return 0
}


create_project_files() {
    print_step "Creating project directory and files..."
    mkdir -p "$PROJECT_DIR/app"
    if [ $? -ne 0 ]; then print_error "Failed to create project directory."; return 1; fi

    # Create Dockerfile
    cat <<EOF > "$PROJECT_DIR/Dockerfile"
# Use a CUDA-enabled base image for GPU support.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3.10-dev \\
    python3-pip \\
    git \\
    wget \\
    libsndfile1 \\
    ffmpeg \\
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

RUN git clone https://github.com/ictnlp/LLaMA-Omni .

# Install core Python packages including google-cloud-storage
RUN pip install pip==24.0
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers==4.30.2 accelerate==0.20.3 huggingface_hub==0.16.4
RUN pip install -e .
RUN pip install fairseq # Install fairseq directly now, as the git clone is already done for LLaMA-Omni
RUN pip install flash-attn --no-build-isolation
RUN pip install fastapi uvicorn[standard] python-multipart soundfile pydub websockets google-cloud-storage

# Create directories for models (where they will be downloaded to)
RUN mkdir -p models/speech_encoder vocoder

# Copy application code and the new download script
COPY app/ /app/app/
COPY index.html /app/index.html # Copy the web client HTML

# Set environment variables for LLaMA-Omni and vocoder paths
ENV WHISPER_MODEL_DIR="/app/models/speech_encoder"
ENV VOCODER_MODEL_PATH="/app/vocoder/g_00500000"
ENV VOCODER_CONFIG_PATH="/app/vocoder/config.json"
ENV LLAMA_OMNI_MODEL_NAME="ictnlp/LLaMA-Omni"

# Expose port 8000
EXPOSE 8000

# Entrypoint to first download models from GCS (if GCS_BUCKET_NAME is set)
# then start the Uvicorn server.
# The GCS_BUCKET_NAME and GOOGLE_APPLICATION_CREDENTIALS env vars
# will be passed during 'docker run' by the install script.
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["python /app/app/download_models.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
EOF
    if [ $? -ne 0 ]; then print_error "Failed to create Dockerfile."; return 1; fi

    # Create app/main.py
    cat <<EOF > "$PROJECT_DIR/app/main.py"
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

# Import LLaMA-Omni specific components from the cloned repository
from omni_speech.model.builder import load_model as load_llama_omni_model
from omni_speech.common.config import Config

# For vocoder components from fairseq
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
from fairseq import checkpoint_utils

# Initialize FastAPI app
app = FastAPI(
    title="LLaMA-Omni Real-time API",
    description="API for seamless real-time speech interaction with LLaMA-Omni model via WebSockets.",
    version="1.0.0"
)

# Global variables to hold the loaded models
llama_omni_model = None
vocoder_model = None
vocoder_cfg = None

# Constants for audio processing
TARGET_SAMPLE_RATE = 16000 # LLaMA-Omni's expected sample rate
AUDIO_CHUNK_SIZE_MS = 1000 # Process audio in 1-second chunks for streaming
AUDIO_CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * (AUDIO_CHUNK_SIZE_MS / 1000))

class LlamaOmniArgs:
    """
    A dummy arguments object to mimic the command-line arguments structure
    expected by \`omni_speech.model.builder.load_model\`.
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
        # Point Whisper to the local path where it will be downloaded by download_models.py
        self.cfg.model.whisper_model_path = os.environ.get("WHISPER_MODEL_DIR")
        # Point Vocoder to the local path where it will be downloaded by download_models.py
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
    global llama_omni_model, vocoder_model, vocoder_cfg

    print("Starting model loading process... This may take a while.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Models will be loaded to device: {device}")

    try:
        # 1. Load the HiFi-GAN Vocoder
        print("Loading HiFi-GAN vocoder...")
        vocoder_path = os.environ.get("VOCODER_MODEL_PATH")
        vocoder_config_path = os.environ.get("VOCODER_CONFIG_PATH")

        if not os.path.exists(vocoder_path) or not os.path.exists(vocoder_config_path):
             raise FileNotFoundError(
                 f"Vocoder files not found locally. Check paths: {vocoder_path} and {vocoder_config_path}. "
                 "Ensure they were downloaded correctly by download_models.py or are present from GCS."
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
        print("HiFi-GAN Vocoder loaded successfully.")

        # 2. Load the LLaMA-Omni model
        # This function internally handles loading the base LLM (Llama-3.1-8B-Omni)
        # and the Whisper speech encoder from Hugging Face, or from local path if specified.
        print(f"Loading LLaMA-Omni model from Hugging Face: {os.environ.get('LLAMA_OMNI_MODEL_NAME')}...")
        args = LlamaOmniArgs()

        # Pass the loaded vocoder and its config to the LLaMA-Omni model's configuration.
        args.cfg.model.vocoder_model = vocoder_model
        args.cfg.model.vocoder_cfg = vocoder_cfg

        llama_omni_model = load_llama_omni_model(args)
        llama_omni_model.eval()
        llama_omni_model = llama_omni_model.to(device)
        print("LLaMA-Omni model loaded successfully.")

        print("All models initialized and loaded successfully. API is ready.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: Failed to load one or more models: {e}")
        raise RuntimeError(f"Failed to load models during startup: {e}")

# Health check endpoint to verify model readiness
@app.get("/health")
async def health_check():
    if llama_omni_model and vocoder_model:
        return {"status": "ok", "message": "All models loaded and ready."}
    return {"status": "error", "message": "Models not loaded yet or failed to load. Check server logs."}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")

    if not llama_omni_model or not vocoder_model:
        await websocket.send_json({"error": "Models are still loading or failed to load. Please try again later."})
        await websocket.close()
        return

    audio_buffer = torch.empty(0, dtype=torch.float32, device=llama_omni_model.device)
    current_prompt = ""
    # Resampler is initialized here, but actual resampling logic needs to be robust for varied input sample rates
    # For this example, we assume client sends 16kHz or close, and server handles minor discrepancies.
    # For production, consider robust resampling if client input sample rate varies significantly.
    resampler = T.Resample(orig_freq=TARGET_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE).to(llama_omni_model.device)

    try:
        while True:
            data = await websocket.receive_bytes()

            audio_chunk = np.frombuffer(data, dtype=np.float32)
            audio_chunk_tensor = torch.from_numpy(audio_chunk).to(llama_omni_model.device)

            audio_buffer = torch.cat((audio_buffer, audio_chunk_tensor))

            while audio_buffer.shape[0] >= AUDIO_CHUNK_SAMPLES:
                process_chunk = audio_buffer[:AUDIO_CHUNK_SAMPLES]
                audio_buffer = audio_buffer[AUDIO_CHUNK_SAMPLES:]

                samples = {
                    "audio": [process_chunk],
                    "text_input": [current_prompt],
                }

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

                    current_prompt = text_response

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

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WebSocket error: {e}")
        await websocket.send_json({"error": f"Server error: {e}"})
    finally:
        await websocket.close()
EOF
    if [ $? -ne 0 ]; then print_error "Failed to create app/main.py."; return 1; fi

    # Create app/download_models.py
    cat <<EOF > "$PROJECT_DIR/app/download_models.py"
import os
import sys
from google.cloud import storage

def download_model_from_gcs(bucket_name, gcs_path, local_path):
    """Downloads a single model file from GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        # Create local directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading {gcs_path} from bucket {bucket_name} to {local_path}...")
        blob.download_to_filename(local_path)
        print(f"Successfully downloaded {gcs_path}")
        return True
    except Exception as e:
        print(f"Error downloading {gcs_path}: {e}", file=sys.stderr)
        return False

def main():
    GCS_BUCKET = os.environ.get("GCS_BUCKET_NAME")
    GCP_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not GCS_BUCKET or not GCP_CREDENTIALS_PATH:
        print("GCS_BUCKET_NAME or GOOGLE_APPLICATION_CREDENTIALS environment variables not set. Skipping GCS model download.", file=sys.stderr)
        print("Models will be downloaded from public Hugging Face/Fairseq sources by LLaMA-Omni's builder if not found locally.", file=sys.stderr)
        return

    # Define your model mappings: GCS path to local path within the container
    # These paths should mirror where your main.py expects the models
    models_to_download = {
        "models/vocoder/g_00500000": "/app/vocoder/g_00500000",
        "models/vocoder/config.json": "/app/vocoder/config.json",
        "models/speech_encoder/whisper-large-v3.pt": "/app/models/speech_encoder/whisper-large-v3.pt",
        # Add other large model files here if you move them to GCS
        # Note: The main LLaMA-Omni model (ictnlp/LLaMA-Omni) is typically
        # downloaded by its internal 'load_model' from Hugging Face.
        # If you want to host it on GCS, you'd need to modify LLaMA-Omni's source.
    }

    all_downloads_successful = True
    for gcs_path, local_path in models_to_download.items():
        if not download_model_from_gcs(GCS_BUCKET, gcs_path, local_path):
            all_downloads_successful = False
            # Continue to try other downloads, but mark overall failure
            print(f"Failed to download {gcs_path}. Continuing with other files...", file=sys.stderr)

    if not all_downloads_successful:
        print("WARNING: One or more models failed to download from GCS. Please check your GCS bucket, object paths, and service account permissions.", file=sys.stderr)
        # We don't exit here immediately, as LLaMA-Omni might still try to download from HF.
        # But it's a strong warning.
    else:
        print("All specified models successfully downloaded from GCS.")

if __name__ == "__main__":
    main()
EOF
    if [ $? -ne 0 ]; then print_error "Failed to create app/download_models.py."; return 1; fi

    # Create index.html
    cat <<EOF > "$PROJECT_DIR/index.html"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLaMA-Omni Real-time Chat</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        .chat-container {
            background-color: #ffffff;
            border-radius: 1rem; /* rounded-xl */
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); /* shadow-lg */
            width: 100%;
            max-width: 768px; /* md:max-w-2xl */
            overflow: hidden;
            display: flex;
            flex-direction: column;
            min-height: 600px;
        }
        .chat-messages {
            flex-grow: 1;
            padding: 1.5rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .message-bubble {
            padding: 0.75rem 1rem;
            border-radius: 0.75rem; /* rounded-lg */
            max-width: 80%;
        }
        .user-message {
            background-color: #3b82f6; /* blue-500 */
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 0.25rem; /* rounded-br-sm */
        }
        .ai-message {
            background-color: #e5e7eb; /* gray-200 */
            color: #374151; /* gray-700 */
            align-self: flex-start;
            border-bottom-left-radius: 0.25rem; /* rounded-bl-sm */
        }
        .controls {
            padding: 1.5rem;
            border-top: 1px solid #e5e7eb; /* border-gray-200 */
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .record-button {
            padding: 0.75rem 1.5rem;
            border-radius: 9999px; /* rounded-full */
            font-weight: 600; /* font-semibold */
            transition: background-color 0.2s ease-in-out;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            cursor: pointer;
        }
        .record-button.idle {
            background-color: #10b981; /* green-500 */
            color: white;
        }
        .record-button.recording {
            background-color: #ef4444; /* red-500 */
            color: white;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .status-message {
            text-align: center;
            color: #6b7280; /* gray-500 */
            font-size: 0.875rem; /* text-sm */
        }
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #fff;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: none; /* Hidden by default */
        }
        .record-button.recording .spinner {
            display: block;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .server-status {
            font-size: 0.875rem;
            text-align: center;
            margin-top: 10px;
            padding: 5px;
            border-radius: 0.5rem;
        }
        .server-status.connected {
            background-color: #d1fae5; /* green-100 */
            color: #065f46; /* green-800 */
        }
        .server-status.disconnected {
            background-color: #fee2e2; /* red-100 */
            color: #991b1b; /* red-800 */
        }
        .server-status.connecting {
            background-color: #fffbeb; /* yellow-100 */
            color: #92400e; /* yellow-800 */
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">

    <div class="chat-container">
        <h1 class="text-3xl font-bold text-center p-6 bg-blue-600 text-white rounded-t-xl">
            LLaMA-Omni Real-time Chat
        </h1>
        <div id="serverStatus" class="server-status connecting">Connecting to server...</div>

        <div id="chatMessages" class="chat-messages">
            <div class="ai-message message-bubble">
                Hello! I'm LLaMA-Omni. How can I help you today?
            </div>
        </div>

        <div class="controls">
            <button id="recordButton" class="record-button idle">
                <span id="recordIcon">🎤</span>
                <span id="recordText">Start Recording</span>
                <div class="spinner" id="recordSpinner"></div>
            </button>
            <p id="statusMessage" class="status-message">Click to start recording.</p>
        </div>
    </div>

    <script>
        const recordButton = document.getElementById('recordButton');
        const recordText = document.getElementById('recordText');
        const recordIcon = document.getElementById('recordIcon');
        const recordSpinner = document.getElementById('recordSpinner');
        const statusMessage = document.getElementById('statusMessage');
        const chatMessages = document.getElementById('chatMessages');
        const serverStatusDiv = document.getElementById('serverStatus');

        let mediaRecorder;
        let audioChunks = [];
        let websocket;
        let isRecording = false;
        let audioContext;
        let audioWorkletNode;
        let sourceNode;

        // Configuration for the WebSocket server
        // IMPORTANT: Replace with your cloud instance's IP address or domain
        const WS_URL = "ws://YOUR_CLOUD_INSTANCE_IP:8000/ws"; // e.g., "ws://192.168.1.100:8000/ws" or "wss://yourdomain.com/ws"

        // Function to update server status
        function updateServerStatus(status, message) {
            serverStatusDiv.className = \`server-status \${status}\`;
            serverStatusDiv.textContent = message;
        }

        // Initialize WebSocket connection
        function connectWebSocket() {
            updateServerStatus('connecting', 'Connecting to server...');
            websocket = new WebSocket(WS_URL);

            websocket.onopen = () => {
                console.log("WebSocket connected.");
                updateServerStatus('connected', 'Connected to LLaMA-Omni server.');
                recordButton.disabled = false; // Enable button once connected
            };

            websocket.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                if (data.error) {
                    console.error("Server error:", data.error);
                    statusMessage.textContent = \`Server error: \${data.error}\`;
                    return;
                }

                if (data.text) {
                    addMessageToChat('ai', data.text);
                }

                if (data.audio_base64) {
                    const audioBlob = base64ToBlob(data.audio_base64, 'audio/wav');
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = new Audio(audioUrl);
                    audio.play();
                    audio.onended = () => URL.revokeObjectURL(audioUrl); // Clean up
                }
            };

            websocket.onclose = (event) => {
                console.log("WebSocket disconnected:", event);
                updateServerStatus('disconnected', 'Disconnected from server. Retrying in 5s...');
                recordButton.disabled = true; // Disable button on disconnect
                // Attempt to reconnect after a delay
                setTimeout(connectWebSocket, 5000);
            };

            websocket.onerror = (error) => {
                console.error("WebSocket error:", error);
                updateServerStatus('disconnected', 'WebSocket error. Retrying in 5s...');
                recordButton.disabled = true;
                websocket.close(); // Force close to trigger onclose and reconnect
            };
        }

        // Start connection on page load
        connectWebSocket();

        // Base64 to Blob utility
        function base64ToBlob(base64, mimeType) {
            const byteCharacters = atob(base64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            return new Blob([byteArray], { type: mimeType });
        }

        // Add message to chat display
        function addMessageToChat(sender, text) {
            const messageDiv = document.createElement('div');
            messageDiv.className = \`message-bubble \${sender}-message\`;
            messageDiv.textContent = text;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom
        }

        // Audio Worklet Processor for resampling and sending audio
        const audioWorkletCode = \`
            class AudioProcessor extends AudioWorkletProcessor {
                constructor() {
                    super();
                    this.sampleRate = 16000; // Target sample rate for the server
                    this.buffer = [];
                    this.bufferSize = this.sampleRate * 0.1; // 100ms chunks for sending

                    // Resampling logic (simplified for demonstration, full resampler is complex)
                    // For production, consider a dedicated resampler library or more robust logic.
                    // This assumes input sample rate is close to target or handles simple decimation/interpolation.
                }

                process(inputs, outputs, parameters) {
                    const input = inputs[0]; // First input (microphone)
                    if (input.length === 0) return true; // No audio data

                    const inputChannel = input[0]; // Mono audio

                    // Append input audio to buffer
                    for (let i = 0; i < inputChannel.length; i++) {
                        this.buffer.push(inputChannel[i]);
                    }

                    // Send chunks if buffer is large enough
                    while (this.buffer.length >= this.bufferSize) {
                        const chunk = new Float32Array(this.buffer.splice(0, this.bufferSize));
                        this.port.postMessage(chunk);
                    }

                    return true; // Keep processor alive
                }
            }
            registerProcessor('audio-processor', AudioProcessor);
        \`;

        async function startRecording() {
            if (isRecording) return;

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 }); // Set context sample rate to 16kHz
                sourceNode = audioContext.createMediaStreamSource(stream);

                await audioContext.audioWorklet.addModule(URL.createObjectURL(new Blob([audioWorkletCode], { type: 'application/javascript' })));
                audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');

                audioWorkletNode.port.onmessage = (event) => {
                    // Send raw float32 array as binary data over WebSocket
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(event.data.buffer);
                    }
                };

                sourceNode.connect(audioWorkletNode);
                audioWorkletNode.connect(audioContext.destination); // Connect to speakers to hear yourself (optional)

                isRecording = true;
                recordButton.classList.remove('idle');
                recordButton.classList.add('recording');
                recordText.textContent = 'Recording...';
                recordIcon.textContent = '🔴';
                statusMessage.textContent = 'Speak now. Audio is streaming to the model.';
                recordButton.disabled = false; // Ensure it's enabled for stopping
            } catch (err) {
                console.error('Error accessing microphone:', err);
                statusMessage.textContent = 'Error: Could not access microphone. Please allow access.';
                recordButton.disabled = true; // Disable if mic access fails
            }
        }

        function stopRecording() {
            if (!isRecording) return;

            isRecording = false;
            if (sourceNode) {
                sourceNode.disconnect();
                sourceNode = null;
            }
            if (audioWorkletNode) {
                audioWorkletNode.disconnect();
                audioWorkletNode = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }

            recordButton.classList.remove('recording');
            recordButton.classList.add('idle');
            recordText.textContent = 'Start Recording';
            recordIcon.textContent = '🎤';
            recordSpinner.style.display = 'none'; // Hide spinner
            statusMessage.textContent = 'Recording stopped. Click to start again.';
            recordButton.disabled = false;
        }

        recordButton.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });

        // Initial state
        recordButton.disabled = true; // Disable until WebSocket connects
    </script>
</body>
</html>
EOF
    if [ $? -ne 0 ]; then print_error "Failed to create index.html."; return 1; fi

    print_success "Project files created successfully in '$PROJECT_DIR/'."
    return 0
}

build_docker_image() {
    print_step "Building Docker image '$DOCKER_IMAGE_NAME'..."
    cd "$PROJECT_DIR" || { print_error "Failed to change directory to $PROJECT_DIR"; return 1; }
    echo "Building Docker image. This will download base images and install dependencies. This may take a while..." | tee -a "../$INSTALL_LOG"
    docker build -t "$DOCKER_IMAGE_NAME":latest . 2>&1 | tee -a "../$INSTALL_LOG"
    BUILD_STATUS=$?
    cd .. # Go back to original directory
    if [ $BUILD_STATUS -eq 0 ]; then
        print_success "Docker image '$DOCKER_IMAGE_NAME' built successfully."
        return 0
    else
        print_error "Failed to build Docker image '$DOCKER_IMAGE_NAME'."
        return 1
    fi
}

run_docker_container() {
    print_step "Running Docker container '$DOCKER_CONTAINER_NAME'..."
    # Stop and remove existing container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "$DOCKER_CONTAINER_NAME"; then
        print_warning "Existing container '$DOCKER_CONTAINER_NAME' found. Stopping and removing it."
        docker stop "$DOCKER_CONTAINER_NAME" 2>&1 | tee -a "$INSTALL_LOG"
        docker rm "$DOCKER_CONTAINER_NAME" 2>&1 | tee -a "$INSTALL_LOG"
    fi

    # Construct the docker run command dynamically
    DOCKER_RUN_CMD="docker run -d --name "$DOCKER_CONTAINER_NAME" --gpus all -p "$API_PORT":"$API_PORT""

    if [[ "$GCS_ENABLED" == "yes" ]]; then
        # Mount the GCS key file into the container
        DOCKER_RUN_CMD+=" -v $(pwd)/$GCP_KEY_DIR/$GCS_KEY_FILE_NAME:$GCS_KEY_MOUNT_PATH:ro"
        # Set environment variables for GCS authentication and bucket name
        DOCKER_RUN_CMD+=" -e GOOGLE_APPLICATION_CREDENTIALS=\"$GCS_KEY_MOUNT_PATH\""
        DOCKER_RUN_CMD+=" -e GCS_BUCKET_NAME=\"$GCS_BUCKET_NAME\""
    fi

    DOCKER_RUN_CMD+=" $DOCKER_IMAGE_NAME:latest"

    echo "Executing Docker command: $DOCKER_RUN_CMD" | tee -a "$INSTALL_LOG"
    eval "$DOCKER_RUN_CMD" >> "$INSTALL_LOG" 2>&1
    RUN_STATUS=$?
    if [ $RUN_STATUS -eq 0 ]; then
        print_success "Docker container '$DOCKER_CONTAINER_NAME' started successfully on port $API_PORT."
        print_warning "Model loading (including GCS downloads if configured) will take several minutes. You can monitor progress with: ${YELLOW}docker logs -f $DOCKER_CONTAINER_NAME${NC}"
        return 0
    else
        print_error "Failed to run Docker container '$DOCKER_CONTAINER_NAME'."
        print_error "Check '$INSTALL_LOG' for details. You might need to stop and remove previous containers with 'docker rm -f $DOCKER_CONTAINER_NAME'."
        return 1
    fi
}

# --- Main Script Execution ---

# Clear previous logs and reports
rm -f "$INSTALL_LOG" "$INSTALL_REPORT"

STEP_COUNTER=1
EXIT_STATUS=0 # 0 for success, 1 for failure

print_header
log_message "LLaMA-Omni Deployment Wizard started."

# Trap errors to generate report
trap 'generate_install_report' EXIT

# 1. Check for sudo privileges
check_sudo

# 2. Install gcloud CLI
install_gcloud_cli
[ $EXIT_STATUS -ne 0 ] && exit $EXIT_STATUS

# 3. Authenticate with Google Cloud
gcloud_login
[ $EXIT_STATUS -ne 0 ] && exit $EXIT_STATUS

# 4. Check and install Docker
check_docker
[ $EXIT_STATUS -ne 0 ] && exit $EXIT_STATUS

# 5. Check and install NVIDIA Container Toolkit (and GPU access)
check_nvidia_docker
[ $EXIT_STATUS -ne 0 ] && exit $EXIT_STATUS

# 6. Display Hardware Info
display_hardware_info

# 7. Configure Automated GCS (User interaction)
configure_gcs_automated
# Note: configure_gcs_automated sets GCS_ENABLED and EXIT_STATUS internally.
[ $EXIT_STATUS -ne 0 ] && exit $EXIT_STATUS # Exit if GCS setup failed

# 8. Create Project Files (Dockerfile, app/main.py, app/download_models.py, index.html)
create_project_files
[ $EXIT_STATUS -ne 0 ] && exit $EXIT_STATUS

# 9. Build Docker Image
build_docker_image
[ $EXIT_STATUS -ne 0 ] && exit $EXIT_STATUS

# 10. Run Docker Container
run_docker_container
[ $EXIT_STATUS -ne 0 ] && exit $EXIT_STATUS

# 11. Firewall Configuration Reminder
print_step "Firewall Configuration"
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo 'N/A')
print_warning "Your server's public IP address is likely: ${YELLOW}$PUBLIC_IP${NC}"
print_warning "Ensure port ${YELLOW}$API_PORT${NC} is open in your cloud provider's firewall/security groups."
print_warning "For UFW (Ubuntu Firewall), you can run: ${YELLOW}sudo ufw allow $API_PORT/tcp${NC}"

# 12. Final Instructions
print_step "Deployment Complete!"
echo -e "${GREEN}CoqVoice API is now running in a Docker container.${NC}"
echo -e "${GREEN}Access the web client by opening your browser to:${NC}"
echo -e "${MAGENTA}http://$PUBLIC_IP:$API_PORT/index.html${NC}"
echo -e "${YELLOW}Remember to replace 'YOUR_CLOUD_INSTANCE_IP' in index.html with '$PUBLIC_IP' if you haven't already!${NC}"
echo -e "${YELLOW}Wait for the model to finish loading inside the container (check logs) before interacting.${NC}"

# --- Installation Report Generation ---
generate_install_report() {
    echo -e "\n${BLUE}=====================================================${NC}" | tee -a "$INSTALL_REPORT"
    echo -e "${BLUE}  CoqVoice Deployment Report                         ${NC}" | tee -a "$INSTALL_REPORT"
    echo -e "${BLUE}=====================================================${NC}" | tee -a "$INSTALL_REPORT"
    echo "Date: $(date)" | tee -a "$INSTALL_REPORT"
    echo "Server IP: $(curl -s ifconfig.me 2>/dev/null || echo 'N/A')" | tee -a "$INSTALL_REPORT"
    echo "Project Directory: $(pwd)/$PROJECT_DIR" | tee -a "$INSTALL_REPORT"
    echo "Docker Image: $DOCKER_IMAGE_NAME:latest" | tee -a "$INSTALL_REPORT"
    echo "Docker Container: $DOCKER_CONTAINER_NAME" | tee -a "$INSTALL_REPORT"
    echo "API Port: $API_PORT" | tee -a "$INSTALL_REPORT"
    echo "GCS Integration: $GCS_ENABLED" | tee -a "$INSTALL_REPORT"
    if [[ "$GCS_ENABLED" == "yes" ]]; then
        echo "GCP Project ID: $GCP_PROJECT_ID" | tee -a "$INSTALL_REPORT"
        echo "GCS Bucket: $GCS_BUCKET_NAME" | tee -a "$INSTALL_REPORT"
        echo "GCS Service Account: $GCS_SERVICE_ACCOUNT_NAME" | tee -a "$INSTALL_REPORT"
        echo "GCS Key Mounted At: $GCS_KEY_MOUNT_PATH (inside container)" | tee -a "$INSTALL_REPORT"
    fi
    echo "" | tee -a "$INSTALL_REPORT"

    if [ $EXIT_STATUS -eq 0 ]; then
        echo -e "${GREEN}Overall Status: SUCCESS${NC}" | tee -a "$INSTALL_REPORT"
        echo "LLaMA-Omni API and web client files are deployed." | tee -a "$INSTALL_REPORT"
        echo "Access at: http://<YOUR_CLOUD_INSTANCE_IP>:$API_PORT/index.html" | tee -a "$INSTALL_REPORT"
        echo "Remember to update WS_URL in index.html with your actual IP." | tee -a "$INSTALL_REPORT"
    else
        echo -e "${RED}Overall Status: FAILED${NC}" | tee -a "$INSTALL_REPORT"
        echo "Some steps failed during the installation. Please review the log file for details." | tee -a "$INSTALL_REPORT"
    fi
    echo "" | tee -a "$INSTALL_REPORT"
    echo "Detailed logs can be found in: $INSTALL_LOG" | tee -a "$INSTALL_REPORT"
    echo -e "${BLUE}=====================================================${NC}" | tee -a "$INSTALL_REPORT"
    echo -e "\nInstallation report saved to ${YELLOW}$INSTALL_REPORT${NC}"
    echo -e "Full log saved to ${YELLOW}$INSTALL_LOG${NC}"
}
