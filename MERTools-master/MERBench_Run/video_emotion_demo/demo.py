import os
import sys
import argparse
import subprocess
import torch
import numpy as np

# Add parent directory to path to import toolkit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import FeatureExtractor
from toolkit.models.attention_robust_v2 import AttentionRobustV2

def extract_audio_from_video(video_path, audio_path):
    command = f"ffmpeg -y -i {video_path} -ac 1 -ar 16000 -vn {audio_path} -loglevel quiet"
    subprocess.call(command, shell=True)

class DemoArgs:
    def __init__(self):
        # Feature dimensions matching checkpoint (Baichuan-13B + hubert-large + clip-large)
        self.text_dim = 5120
        self.audio_dim = 1024
        self.video_dim = 768
        
        # Model params
        self.output_dim1 = 6  # Emotion classes
        self.output_dim2 = 1  # Valence (if needed)
        self.dropout = 0.3
        self.hidden_dim = 128
        self.grad_clip = 1.0
        
        # V2 params
        self.use_vae = True
        self.kl_weight = 0.01
        self.recon_weight = 0.1
        self.cross_kl_weight = 0.01
        self.use_proxy_attention = True
        self.fusion_temperature = 1.0
        self.num_attention_heads = 4
        
        self.modality_dropout = 0.0
        self.use_modality_dropout = False
        self.modality_dropout_warmup = 0
        
        self.feat_type = 'utt' # Utterance level

def main():
    parser = argparse.ArgumentParser(description='Video Emotion Recognition Demo')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--text', type=str, default=None, help='Text content of the video (optional)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return

    # 1. Initialize Feature Extractor
    print("\n>>> Step 1: Initializing Feature Extractor...")
    extractor = FeatureExtractor()
    
    # 2. Extract Features
    print("\n>>> Step 2: Extracting Features...")
    
    # Audio
    audio_path = "temp_audio.wav"
    extract_audio_from_video(args.video_path, audio_path)
    audio_feat = extractor.extract_audio_feature(audio_path)
    os.remove(audio_path)
    print(f"Audio feature shape: {audio_feat.shape}")
    
    # Video
    video_feat = extractor.extract_video_feature(args.video_path)
    print(f"Video feature shape: {video_feat.shape}")
    
    # Text
    text_content = args.text
    if text_content is None:
        print("Warning: No text provided. Using empty text feature.")
        text_content = ""
    text_feat = extractor.extract_text_feature(text_content)
    print(f"Text feature shape: {text_feat.shape}")
    
    # 3. Load Model
    print("\n>>> Step 3: Loading Model...")
    model_args = DemoArgs()
    model = AttentionRobustV2(model_args).cuda()
    model.eval()
    
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}...")
        try:
            state_dict = torch.load(args.checkpoint)
            # Remove 'model.' prefix from keys if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v  # Remove 'model.' prefix
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using random initialization (Predictions will be random!)")
    else:
        print("Warning: No checkpoint provided. Using random initialization.")
        print("Note: Ideally you should train the model with these new feature dimensions first.")

    # 4. Inference
    print("\n>>> Step 4: Inference...")
    
    # Prepare batch
    batch = {
        'audios': torch.tensor(audio_feat).float().cuda(),
        'texts': torch.tensor(text_feat).float().cuda(),
        'videos': torch.tensor(video_feat).float().cuda()
    }
    
    with torch.no_grad():
        _, emos_out, vals_out, _ = model(batch)
        probs = torch.softmax(emos_out, dim=1)
        
    emotion_labels = ['Neutral', 'Angry', 'Happy', 'Sad', 'Worry', 'Surprise'] # MER2023 order typically
    
    print("\n====== Prediction Results ======")
    print(f"Input Video: {args.video_path}")
    print(f"Input Text: \"{text_content}\"")
    print("-" * 30)
    
    pred_idx = torch.argmax(probs).item()
    print(f"Predicted Emotion: {emotion_labels[pred_idx]}")
    print(f"Confidence: {probs[0][pred_idx]:.4f}")
    
    print("\nAll Probabilities:")
    for i, label in enumerate(emotion_labels):
        print(f"{label}: {probs[0][i]:.4f}")

if __name__ == '__main__':
    main()
