# Use a CUDA-enabled base image for GPU support for CoqVoice.
# This specific version (11.8.0-cudnn8) is chosen for compatibility with PyTorch 2.0.1.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables for non-buffered Python output and non-interactive Debian frontend.
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies:
# - python3.10, python3.10-dev: Python interpreter and development headers.
# - python3-pip: Python package installer.
# - git: For cloning repositories.
# - wget: For downloading files.
# - libsndfile1: Required by soundfile for audio processing.
# - ffmpeg: Essential for various audio/video operations, often a dependency for audio libraries.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache to reduce image size

# Set Python 3.10 as the default Python and pip.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory inside the container.
WORKDIR /app

# Clone the LLaMA-Omni repository (used by CoqVoice) into the working directory.
RUN git clone https://github.com/ictnlp/LLaMA-Omni .

# Install core Python packages:
# - pip: Upgrade pip to a recent stable version.
# - torch, torchvision, torchaudio: Install PyTorch with CUDA 11.8 support.
#   The specific versions are chosen to align with common practices for LLaMA-Omni and Flash Attention.
# - transformers, accelerate, huggingface_hub: Essential for loading models from Hugging Face.
# - -e .: Install LLaMA-Omni's own Python package dependencies from its setup.py.
RUN pip install pip==24.0
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers==4.30.2 accelerate==0.20.3 huggingface_hub==0.16.4
RUN pip install -e .

# Install fairseq, which is a dependency for the vocoder and other components.
# It's cloned and installed in editable mode.
RUN git clone https://github.com/pytorch/fairseq /app/fairseq
RUN pip install -e /app/fairseq --no-build-isolation

# Install flash-attention for optimized attention mechanisms (requires GPU).
# This must be installed after PyTorch.
RUN pip install flash-attn --no-build-isolation

# Install FastAPI and other libraries for the web application:
# - fastapi: The web framework.
# - uvicorn[standard]: ASGI server to run FastAPI, with standard dependencies (like httptools, websockets).
# - python-multipart: For handling form data (e.g., file uploads).
# - soundfile: For reading and writing audio files.
# - pydub: A simple audio manipulation library (useful for basic audio ops).
# - websockets: Explicitly ensure websockets is available for streaming.
RUN pip install fastapi uvicorn[standard] python-multipart soundfile pydub websockets silero-vad onnxruntime

# Create directories to store downloaded models.
RUN mkdir -p models/speech_encoder vocoder

# Download the unit-based HiFi-GAN vocoder model and its configuration.
# These are directly downloaded using wget into the 'vocoder' directory.
RUN wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -P vocoder/
RUN wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -P vocoder/

# Copy your FastAPI application code into the container.
# The 'app' directory in your local project will be copied to '/app/app' in the container.
COPY ./app /app/app

# Set environment variables that your CoqVoice FastAPI application will use to locate models.
# LLAMA_OMNI_MODEL_NAME is the Hugging Face model ID for the underlying LLaMA-Omni model.
ENV WHISPER_MODEL_DIR="/app/models/speech_encoder"
ENV VOCODER_MODEL_PATH="/app/vocoder/g_00500000"
ENV VOCODER_CONFIG_PATH="/app/vocoder/config.json"
ENV LLAMA_OMNI_MODEL_NAME="ictnlp/LLaMA-Omni"

# Expose port 8000, which is where the FastAPI application will listen.
EXPOSE 8000

# Define the command to run the FastAPI application using Uvicorn.
# It will listen on all network interfaces (0.0.0.0) on port 8000.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
