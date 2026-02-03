import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
import cv2
from PIL import Image
import librosa
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, Wav2Vec2Processor, HubertModel, BertTokenizer, BertModel

import sys
import torchaudio

class FeatureExtractor:
    def __init__(self, device='cuda'):
        # Ensure torchaudio initializes properly BEFORE librosa/soundfile calls if there are conflicts
        try:
             # Basic sanity check
             pass
        except:
             pass

        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Initializing FeatureExtractor on {self.device}...")
        self.init_models()
        print("Models loaded successfully.")
        
    def init_models(self):
        # 1. Text Model: Baichuan-13B-Base
        # Dim: 5120
        print("Loading Text Model (baichuan-inc/Baichuan-13B-Base)...")
        self.tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-13B-Base", trust_remote_code=True)
        from transformers import AutoModelForCausalLM
        self.text_model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Base", trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.text_model.eval()

        # 2. Audio Model: Chinese Hubert Large
        # Dim: 1024
        print("Loading Audio Model (TencentGameMate/chinese-hubert-large)...")
        try:
            self.audio_processor = AutoFeatureExtractor.from_pretrained("TencentGameMate/chinese-hubert-large")
            self.audio_model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-large").to(self.device)
        except Exception as e:
            print(f"Failed to load chinese-hubert-large: {e}")
            print("Fallback to facebook/hubert-large-ls960-ft...")
            self.audio_processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
            self.audio_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(self.device)
        self.audio_model.eval()

        # 3. Video Model: CLIP ViT Large
        # Dim: 768
        print("Loading Video Model (openai/clip-vit-large-patch14)...")
        self.video_processor = AutoFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
        self.video_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.video_model.eval()
        
    def extract_text_feature(self, text):
        if not text:
            return np.zeros((1, 5120))
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs, output_hidden_states=True)
        # Use last hidden state mean for Baichuan
        return outputs.hidden_states[-1].mean(dim=1).float().cpu().numpy()

    def extract_audio_feature(self, audio_path):
        import soundfile as sf
        
        # Load audio using soundfile directly if librosa fails or use librosa with soundfile backend
        try:
            raw_speech, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Librosa load failed: {e}, attempting torchaudio...")
            wav, sr = torchaudio.load(audio_path)
            if sr != 16000:
                transform = torchaudio.transforms.Resample(sr, 16000)
                wav = transform(wav)
            raw_speech = wav.mean(dim=0).numpy() # Convert stereo to mono if needed
        
        # Handle short audio
        if len(raw_speech) < 1600: # < 0.1s
            raw_speech = np.pad(raw_speech, (0, 1600-len(raw_speech)))

        inputs = self.audio_processor(raw_speech, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
        
        # Average pooling over time dimension -> (1, 1024)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def extract_video_feature(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Uniform sampling 16 frames
        indices = np.linspace(0, frame_count - 1, 16, dtype=int)
        
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in indices:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            current_frame += 1
        cap.release()
        
        if len(frames) == 0:
            return np.zeros((1, 768))
            
        # If we missed some frames due to read errors, pad
        while len(frames) < 16:
            frames.append(frames[-1])
            
        # CLIP inputs
        inputs = self.video_processor(images=frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Use get_image_features for CLIPModel to avoid calling the text branch
            # This returns the projected 512-dim features
            outputs = self.video_model.get_image_features(**inputs)
            
        # outputs: (16, 768) -> Mean pooling -> (1, 768)
        return outputs.mean(dim=0, keepdim=True).cpu().numpy()
