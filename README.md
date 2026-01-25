# ComfyUI-Qwen-TTS

English | [中文版](README_CN.md)

![Nodes Screenshot](example/example.png)

ComfyUI custom nodes for speech synthesis, voice cloning, and voice design, based on the open-source **Qwen3-TTS** project by the Alibaba Qwen team.

## 📋 Changelog

- **2026-01-24**: Added generation parameters (top_p, top_k, temperature, repetition_penalty) to all TTS nodes ([update.md](doc/update.md))
- **2026-01-23**: Dependency compatibility & Mac (MPS) support, New nodes: VoiceClonePromptNode, DialogueInferenceNode ([update.md](doc/update.md))

## Key Features

- 🎵 **Speech Synthesis**: High-quality text-to-speech conversion.
- 🎭 **Voice Cloning**: Zero-shot voice cloning from short reference audio.
- 🎨 **Voice Design**: Create custom voice characteristics based on natural language descriptions.
- 🚀 **Efficient Inference**: Supports both 12Hz and 25Hz speech tokenizer architectures.
- 🎯 **Multilingual**: Native support for 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian).
- ⚡ **Integrated Loading**: No separate loader nodes required; model loading is managed on-demand with global caching.
- ⏱️ **Ultra-Low Latency**: Supports high-fidelity speech reconstruction with low-latency streaming.

## Nodes List

### 1. Qwen3-TTS Voice Design (`VoiceDesignNode`)

Generate unique voices based on text descriptions.

- **Inputs**:
  - `text`: Target text to synthesize.
  - `instruct`: Description of the voice (e.g., "A gentle female voice with a high pitch").
  - `model_choice`: Currently locked to **1.7B** for VoiceDesign features.
- **Capabilities**: Best for creating "imaginary" voices or specific character archetypes.

### 2. Qwen3-TTS Voice Clone (`VoiceCloneNode`)

Clone a voice from a reference audio clip.

- **Inputs**:
  - `ref_audio`: A short (5-15s) audio clip to clone.
  - `ref_text`: Text spoken in the `ref_audio` (helps improve quality).
  - `target_text`: The new text you want the cloned voice to say.
  - `model_choice`: Choose between **0.6B** (fast) or **1.7B** (high quality).

### 3. Qwen3-TTS Custom Voice (`CustomVoiceNode`)

Standard TTS using preset speakers.

- **Inputs**:
  - `text`: Target text.
  - `speaker`: Selection from preset voices (Aiden, Eric, Serena, etc.).
  - `instruct`: Optional style instructions.

### 4. Qwen3-TTS Voice Clone Prompt (`VoiceClonePromptNode`) [New]

Extract and reuse voice features from reference audio.

- **Capabilities**: Extract a "prompt item" once and use it multiple times across different `VoiceCloneNode` instances for faster and more consistent generation.

### 5. Qwen3-TTS Multi-role Dialogue (`DialogueInferenceNode`) [New]

Synthesize complex dialogues with multiple speakers.

- **Capabilities**: Handles multi-role speech synthesis in a single node, ideal for audiobook narration or roleplay scenarios.

## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchaudio transformers librosa accelerate
```

## Model Downloading (Offline Use)

By default, the nodes will attempt to download models from Hugging Face automatically. For offline usage or to avoid download timeouts, you can manually download the models.

### Option 1: Automatic Download Script (Recommended)

You can simply run the provided python script to download all necessary models to the correct location:

```bash
python download_models.py
```

_Note: This script will try to detect your ComfyUI installation. You can also pass arguments like `--small` for 0.6B models._

### Option 2: Manual Download

1. Create a folder `qwen-tts` inside your ComfyUI `models` directory:

   ```
   ComfyUI/models/qwen-tts/
   ```

2. Download the models from Hugging Face and place them in subfolders. The folder names **must match exactly**:

| Model Type               | Folder Name (Create this inside `qwen-tts`) | Hugging Face Repo                                                   |
| ------------------------ | ------------------------------------------- | ------------------------------------------------------------------- |
| **Tokenizer** (Required) | `Qwen3-TTS-Tokenizer-12Hz`                  | [Link](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)        |
| **Voice Clone** (Base)   | `Qwen3-TTS-12Hz-1.7B-Base`                  | [Link](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)        |
| **Voice Design**         | `Qwen3-TTS-12Hz-1.7B-VoiceDesign`           | [Link](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) |
| **Custom Voice**         | `Qwen3-TTS-12Hz-1.7B-CustomVoice`           | [Link](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) |

> **Note**: For the 0.6B versions, check the [Qwen3 Collection](https://huggingface.co/collections/Qwen/qwen3-tts).

## Tips for Best Results

- **Cloning**: Use clean, noise-free reference audio (5-15 seconds).
- **VRAM**: Use `bf16` precision to save significant memory with minimal quality loss.
- **Local Models**: Pre-download weights to `models/qwen-tts/` to prioritize local loading and avoid HuggingFace timeouts.

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Official open-source repository by Alibaba Qwen team.

## License

- This project is licensed under the **Apache License 2.0**.
- Model weights are subject to the [Qwen3-TTS License Agreement](https://github.com/QwenLM/Qwen3-TTS#License).
