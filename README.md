# CoqVoice Real-time Voice Pipeline - Beaverhand

## Overview
This repository contains the CoqVoice Real-time Voice Pipeline, designed for seamless, real-time speech interaction. It leverages the LLaMA-Omni model for speech-to-speech (S2S) and speech-to-text (S2T) capabilities, integrated with a FastAPI backend and a web-based client. The system now includes Voice Activity Detection (VAD) for improved efficiency and responsiveness.

## Features
*   **Real-time Speech-to-Speech (S2S)**: Convert spoken input directly into spoken responses.
*   **Real-time Speech-to-Text (S2T)**: Transcribe spoken input into text.
*   **CoqVoice Integration**: Utilizes the powerful LLaMA-Omni model for advanced language understanding and generation, rebranded as CoqVoice.
*   **FastAPI Backend**: A high-performance Python API built with FastAPI for handling WebSocket connections and model inference.
*   **Dockerized Deployment**: Easy and consistent deployment using Docker.
*   **Automated GCS Integration**: Optional automated setup for storing and serving model weights from Google Cloud Storage.
*   **Voice Activity Detection (VAD)**: Server-side VAD using Silero VAD to intelligently detect speech and process only relevant audio segments, improving efficiency.

## Installation Guide

This guide provides instructions for deploying the LLaMA-Omni Real-time API on a Linux server (Ubuntu/Debian recommended) using the provided `install.sh` script.

### Prerequisites
*   **Operating System**: Ubuntu/Debian-based Linux distribution.
*   **Sudo Privileges**: The installation script requires `sudo` access.
*   **Google Cloud SDK (gcloud CLI)**: Required for GCS integration. The script will attempt to install it if not found.
*   **Docker**: Required for containerizing the application. The script will attempt to install it if not found.
*   **NVIDIA GPU & Drivers**: Essential for GPU acceleration. The script will check for NVIDIA drivers and attempt to install NVIDIA Container Toolkit. A GPU with >= 16GB VRAM is recommended for optimal performance with the LLaMA-Omni 8B model.
*   **Internet Connectivity**: For downloading models and dependencies.

### Step-by-Step Installation

1.  **Clone the Repository**:
    First, clone this repository to your server:
    ```bash
    git clone https://github.com/your-repo/CoqVoice.git # Replace with your actual repo URL
    cd CoqVoice
    ```

2.  **Run the Installation Wizard**:
    Execute the `install.sh` script with `sudo` privileges:
    ```bash
    sudo ./install.sh
    ```
    The script will guide you through the installation process:
    *   It will check for and install necessary tools (gcloud CLI, Docker, NVIDIA Container Toolkit).
    *   It will prompt you to log in to your Google Cloud account (`gcloud auth login`).
    *   **Automated GCS Integration**: You will be asked if you want to enable automated GCS integration for model weights.
        *   If you choose `yes`, you'll need to provide your Google Cloud Project ID and a desired GCS Bucket Name. The script will then automatically create the bucket (if it doesn't exist), set up a service account, grant permissions, generate a key, and upload the Whisper and Vocoder models to your GCS bucket.
        *   If you choose `no`, models will be downloaded from public Hugging Face/Fairseq sources by the Docker container at runtime.
    *   The script will then create the necessary application files (`Dockerfile`, `app/main.py`, `app/download_models.py`, `index.html`), build the Docker image, and run the Docker container.

3.  **Monitor Model Loading**:
    The LLaMA-Omni model and other components are large and will take several minutes to download and load inside the Docker container. You can monitor the progress by checking the Docker logs:
    ```bash
    docker logs -f coqvoice-app
    ```
    Wait until you see messages indicating that "All models initialized and loaded successfully. API is ready."

4.  **Configure Firewall (Important!)**:
    Ensure that port `8000` (the API port) is open in your server's firewall or cloud provider's security groups. For Ubuntu systems using UFW, you can run:
    ```bash
    sudo ufw allow 8000/tcp
    ```

5.  **Access the Web Client**:
    Once the Docker container is running and models are loaded, you can access the web client.
    *   Find your server's public IP address (e.g., using `curl ifconfig.me`).
    *   **Crucially, update the `WS_URL` variable in `coqvoice-api/index.html`** to point to your server's actual IP address or domain.
        ```javascript
        // In coqvoice-api/index.html
        const WS_URL = "ws://YOUR_CLOUD_INSTANCE_IP:8000/ws"; // e.g., "ws://192.168.1.100:8000/ws"
        ```
    *   Open your web browser and navigate to:
        ```
        http://YOUR_CLOUD_INSTANCE_IP:8000/index.html
        ```
    You should see the CoqVoice Real-time Chat interface.

## API Documentation

The CoqVoice Real-time API exposes a single WebSocket endpoint for real-time speech interaction.

**Endpoint:** `ws://YOUR_CLOUD_INSTANCE_IP:8000/ws`

### Request Format (Client to Server)
Clients should establish a WebSocket connection and continuously stream raw audio data.
*   **Audio Format**: `Float32Array` (32-bit floating-point PCM)
*   **Sample Rate**: `16000 Hz` (16kHz)
*   **Channels**: Mono
*   **Chunk Size**: Audio should be sent in small chunks, ideally 50ms (800 samples) for optimal real-time VAD processing.

### Response Format (Server to Client)
The server sends JSON messages containing the transcribed text and synthesized speech.
*   **Content Type**: `application/json`
*   **Example Response**:
    ```json
    {
      "text": "Hello, how can I help you today?",
      "audio_base64": "data:audio/wav;base64,..." // Base64 encoded WAV audio
    }
    ```
*   `text`: The transcribed text from your speech input or the LLaMA-Omni model's generated text response.
*   `audio_base64`: A base64-encoded WAV audio string of the LLaMA-Omni model's synthesized speech response. This field might be empty if no speech is generated.

## Client Examples

### 1. JavaScript (Web Client)

The `coqvoice-api/index.html` file provides a fully functional web client. It demonstrates how to capture microphone audio, stream it via WebSockets, and play back the AI's audio responses.

**Key JavaScript Concepts:**
*   **WebSocket Connection**:
    ```javascript
    let websocket;
    const WS_URL = "ws://YOUR_CLOUD_INSTANCE_IP:8000/ws"; // IMPORTANT: Update this!
    function connectWebSocket() {
        websocket = new WebSocket(WS_URL);
        websocket.onopen = () => console.log("WebSocket connected.");
        websocket.onmessage = async (event) => { /* handle response */ };
        websocket.onclose = (event) => console.log("WebSocket disconnected.");
        websocket.onerror = (error) => console.error("WebSocket error:", error);
    }
    connectWebSocket();
    ```
*   **Microphone Access & Audio Streaming (using AudioWorklet)**:
    The client uses `navigator.mediaDevices.getUserMedia` to get microphone access and an `AudioWorkletNode` to process and send audio chunks.
    ```javascript
    const SAMPLE_RATE = 16000;
    const CHUNK_SIZE_MS = 50;
    const CHUNK_SAMPLES = SAMPLE_RATE * (CHUNK_SIZE_MS / 1000);

    const audioWorkletCode = `
        class AudioProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
                this.sampleRate = ${SAMPLE_RATE};
                this.chunkSizeSamples = ${CHUNK_SAMPLES};
                this.audioBuffer = [];
            }
            process(inputs, outputs, parameters) {
                const input = inputs[0];
                if (input.length === 0) return true;
                const inputChannel = input[0];
                for (let i = 0; i < inputChannel.length; i++) {
                    this.audioBuffer.push(inputChannel[i]);
                }
                while (this.audioBuffer.length >= this.chunkSizeSamples) {
                    const chunk = new Float32Array(this.audioBuffer.splice(0, this.chunkSizeSamples));
                    this.port.postMessage(chunk);
                }
                return true;
            }
        }
        registerProcessor('audio-processor', AudioProcessor);
    `;

    async function startRecording() {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
        sourceNode = audioContext.createMediaStreamSource(stream);
        await audioContext.audioWorklet.addModule(URL.createObjectURL(new Blob([audioWorkletCode], { type: 'application/javascript' })));
        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');
        audioWorkletNode.port.onmessage = (event) => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(event.data.buffer); // Send raw Float32Array buffer
            }
        };
        sourceNode.connect(audioWorkletNode);
        audioWorkletNode.connect(audioContext.destination); // Optional: to hear yourself
    }
    ```
*   **Playing AI Audio Responses**:
    ```javascript
    function base64ToBlob(base64, mimeType) { /* ... implementation from index.html ... */ }

    // Inside websocket.onmessage:
    if (data.audio_base64) {
        const audioBlob = base64ToBlob(data.audio_base64, 'audio/wav');
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
        audio.onended = () => URL.revokeObjectURL(audioUrl);
    }
    ```

### 2. Python Client

You can create a Python client to interact with the API using `websockets` and `sounddevice`.

**Prerequisites (Python):**
```bash
pip install websockets sounddevice numpy soundfile
```

**Example Python Client (`python_client.py`):**
```python
import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json
import base64
import io
import soundfile as sf

# Configuration
WS_URL = "ws://YOUR_CLOUD_INSTANCE_IP:8000/ws" # IMPORTANT: Update this!
SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 50
CHUNK_SAMPLES = int(SAMPLE_RATE * (CHUNK_SIZE_MS / 1000))

async def send_audio_stream():
    print(f"Connecting to WebSocket at {WS_URL}")
    async with websockets.connect(WS_URL) as websocket:
        print("WebSocket connected. Start speaking...")

        def callback(indata, frames, time, status):
            if status:
                print(status)
            # Convert int16 (default for sounddevice) to float32
            audio_chunk = indata.astype(np.float32).tobytes()
            asyncio.run_coroutine_threadsafe(
                websocket.send(audio_chunk), asyncio.get_event_loop()
            )

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES,
                               dtype='int16', channels=1, callback=callback):
            while True:
                try:
                    response_json = await websocket.recv()
                    response_data = json.loads(response_json)

                    if response_data.get("error"):
                        print(f"Server error: {response_data['error']}")
                        continue

                    text_response = response_data.get("text", "")
                    audio_base64 = response_data.get("audio_base64", "")

                    if text_response:
                        print(f"AI: {text_response}")

                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        audio_buffer = io.BytesIO(audio_bytes)
                        # Play audio (requires pydub or similar for direct playback, or save to file)
                        # For simplicity, we'll just save and print info here.
                        # In a real app, you'd use a playback library.
                        try:
                            with sf.SoundFile(audio_buffer, 'r') as f:
                                audio_data = f.read(dtype='float32')
                                sd.play(audio_data, f.samplerate)
                                sd.wait()
                        except Exception as e:
                            print(f"Error playing audio: {e}")

                except websockets.exceptions.ConnectionClosedOK:
                    print("WebSocket connection closed normally.")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"WebSocket connection closed with error: {e}")
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

if __name__ == "__main__":
    try:
        asyncio.run(send_audio_stream())
    except KeyboardInterrupt:
        print("Client stopped.")
```

### 3. Flutter Client (Conceptual Outline)

Building a Flutter client involves using packages for WebSocket communication, microphone access, and audio playback.

**Key Flutter Packages:**
*   `web_socket_channel`: For WebSocket communication.
*   `record`: For microphone recording.
*   `audioplayers` or `just_audio`: For playing audio.
*   `path_provider`: For temporary file storage (if needed for audio playback).

**Conceptual Steps:**

1.  **Add Dependencies**:
    In your `pubspec.yaml`:
    ```yaml
    dependencies:
      flutter:
        sdk: flutter
      web_socket_channel: ^2.x.x
      record: ^x.x.x # Or another audio recording package
      audioplayers: ^x.x.x # Or just_audio
      permission_handler: ^x.x.x # For microphone permissions
    ```

2.  **Request Permissions**:
    Request microphone permission using `permission_handler`.

3.  **WebSocket Connection**:
    ```dart
    import 'package:web_socket_channel/web_socket_channel.dart';

    final wsUrl = Uri.parse('ws://YOUR_CLOUD_INSTANCE_IP:8000/ws'); // IMPORTANT: Update this!
    late WebSocketChannel channel;

    void connectWebSocket() {
      channel = WebSocketChannel.connect(wsUrl);
      channel.stream.listen((message) {
        // Handle incoming JSON messages (text and base64 audio)
        final data = jsonDecode(message);
        if (data['text'] != null) {
          // Update UI with text
        }
        if (data['audio_base64'] != null) {
          // Decode base64 audio and play
          final audioBytes = base64Decode(data['audio_base64']);
          // Use AudioPlayer to play bytes
        }
      }, onError: (error) {
        print('WebSocket Error: $error');
      }, onDone: () {
        print('WebSocket Disconnected');
        // Implement reconnect logic
      });
    }
    ```

4.  **Microphone Recording and Streaming**:
    Use the `record` package to record audio and stream it. You'll need to convert the audio to `Float32List` if the package doesn't provide it directly, and then send it as bytes over the WebSocket.

    ```dart
    import 'package:record/record.dart';
    import 'dart:typed_data';

    final audioRecorder = Record();

    Future<void> startRecordingAndStreaming() async {
      if (await audioRecorder.hasPermission()) {
        await audioRecorder.start(
          path: null, // Stream directly, don't save to file
          encoder: AudioEncoder.pcm16bit, // Or pcmFloat32 if supported
          samplingRate: 16000,
          numChannels: 1,
        );

        audioRecorder.onStateChanged((state) {
          if (state == RecordState.record) {
            audioRecorder.onAmplitudeChanged((amp) {
              // Optional: visual feedback
            });
            audioRecorder.onSounData.listen((data) {
              // 'data' is Uint8List, convert to Float32List if necessary
              // For 16-bit PCM, convert to Float32
              final float32Data = Int16List.fromList(data).buffer.asFloat32List();
              channel.sink.add(float32Data.buffer.asUint8List()); // Send as bytes
            });
          }
        });
      }
    }

    void stopRecording() {
      audioRecorder.stop();
    }
    ```

5.  **Audio Playback**:
    Use `audioplayers` or `just_audio` to play the received base64-encoded audio.

    ```dart
    import 'package:audioplayers/audioplayers.dart';

    final audioPlayer = AudioPlayer();

    Future<void> playBase64Audio(String base64String) async {
      final audioBytes = base64Decode(base64String);
      await audioPlayer.playBytes(audioBytes);
    }
    ```

This updated `README.md` provides a comprehensive guide for installation and client development across multiple platforms.
