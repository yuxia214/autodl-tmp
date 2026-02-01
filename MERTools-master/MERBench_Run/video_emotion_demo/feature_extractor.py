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
        # 1. Text Model: BERT-base-Chinese (Replacing Baichuan 13B)
        # Dim: 768
        print("Loading Text Model (bert-base-chinese)...")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.text_model = BertModel.from_pretrained("bert-base-chinese").to(self.device)
        self.text_model.eval()
        
        # 2. Audio Model: Chinese Hubert Base (Replacing Chinese Hubert Large)
        # Dim: 768
        print("Loading Audio Model (TencentGameMate/chinese-hubert-base)...")
        try:
            # We don't need the tokenizer for feature extraction, only the feature extractor part of the processor
            # So we load AutoFeatureExtractor instead of Wav2Vec2Processor
            self.audio_processor = AutoFeatureExtractor.from_pretrained("TencentGameMate/chinese-hubert-base")
            self.audio_model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-base").to(self.device)
        except Exception as e:
            print(f"Failed to load chinese-hubert-base: {e}")
            print("Fallback to facebook/hubert-base-ls960...")
            self.audio_processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            self.audio_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device)
        self.audio_model.eval()

        # 3. Video Model: CLIP ViT Base (Replacing CLIP ViT Large)
        # Dim: 512
        print("Loading Video Model (openai/clip-vit-base-patch32)...")
        self.video_processor = AutoFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        self.video_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.video_model.eval()
        
    def extract_text_feature(self, text):
        if not text:
            return np.zeros((1, 768))
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        # Use pooler output for sentence representation
        return outputs.pooler_output.cpu().numpy()

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
        
        # Average pooling over time dimension -> (1, 768)
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
            return np.zeros((1, 512))
            
        # If we missed some frames due to read errors, pad
        while len(frames) < 16:
            frames.append(frames[-1])
            
        # CLIP inputs
        inputs = self.video_processor(images=frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Use get_image_features for CLIPModel to avoid calling the text branch
            # This returns the projected 512-dim features
            outputs = self.video_model.get_image_features(**inputs)
            
        # outputs: (16, 512) -> Mean pooling -> (1, 512)
        return outputs.mean(dim=0, keepdim=True).cpu().numpy()
