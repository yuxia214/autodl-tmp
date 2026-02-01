import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import torch
import gradio as gr
import numpy as np
import cv2
import time

# Add parent directory to path to import toolkit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import FeatureExtractor
from toolkit.models.attention_robust_v2 import AttentionRobustV2
from demo import DemoArgs, extract_audio_from_video

# --- Global Components ---
MODELS_LOADED = False
extractor = None
model = None
emotion_labels = ['Neutral', 'Angry', 'Happy', 'Sad', 'Worry', 'Surprise']

def load_components():
    global extractor, model, MODELS_LOADED
    if MODELS_LOADED:
        return

    print("Loading Extractor...")
    extractor = FeatureExtractor()
    
    print("Loading Model...")
    model_args = DemoArgs()
    # model_args.checkpoint = "..." 
    model = AttentionRobustV2(model_args).cuda()
    model.eval()
    
    MODELS_LOADED = True

def predict_emotion(features):
    batch = {
        'audios': torch.tensor(features['audio']).float().cuda(),
        'texts': torch.tensor(features['text']).float().cuda(),
        'videos': torch.tensor(features['video']).float().cuda()
    }
    
    with torch.no_grad():
        _, emos_out, _, _ = model(batch)
        probs = torch.softmax(emos_out, dim=1).cpu().numpy()[0]
        
    return {label: float(prob) for label, prob in zip(emotion_labels, probs)}

def process_video_stream(video_path, text_input):
    """
    Simulates real-time processing by yielding results chunk by chunk.
    """
    if not video_path:
        yield None, "Please upload a video.", None
        return

    if not MODELS_LOADED:
        yield None, "Loading Models... (This happens once)", None
        load_components()

    try:
        yield None, "Preprocessing Audio...", None
        # 1. Extract Audio for the whole file first (easier for slicing)
        audio_path = video_path + ".wav"
        extract_audio_from_video(video_path, audio_path)
        
        # Load raw audio for slicing
        import librosa
        # Use simple reload since we need raw data arrays
        try:
            full_audio, sr = librosa.load(audio_path, sr=16000)
        except:
             # Fallback just in case
            import torchaudio
            wav, sr = torchaudio.load(audio_path)
            if sr != 16000:
                transform = torchaudio.transforms.Resample(sr, 16000)
                wav = transform(wav)
            full_audio = wav.mean(dim=0).numpy()
            sr = 16000
            
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # 2. Text Feature (Use global text for context, ideally ASR would update this)
        text_content = text_input if text_input else ""
        # Cache text feature since it doesn't change per frame in this demo
        text_feat = extractor.extract_text_feature(text_content)

        # 3. Stream Video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process in windows of 1 second
        window_size_sec = 1.0
        window_size_frames = int(fps * window_size_sec)
        
        frame_buffer = []
        current_frame_idx = 0
        
        # Current predictions
        current_probs = None
        current_msg = "Starting..."

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
            current_frame_idx += 1
            
            # Use skipping to reduce display load (e.g. show every 2nd frame)
            # But yielding every frame gives smoothest "video" feel, limited by network
            if current_frame_idx % 2 == 0:
                 # Resize for faster display
                 display_frame = cv2.resize(frame_rgb, (640, 360))
                 yield display_frame, current_msg, current_probs

            # If buffer full, process
            if len(frame_buffer) >= window_size_frames:
                # --- Feature Extraction for this window ---
                yield display_frame, "Analyzing Window...", current_probs
                
                # Video indices
                indices = np.linspace(0, len(frame_buffer) - 1, 16, dtype=int)
                sampled_frames = [frame_buffer[i] for i in indices]
                
                inputs = extractor.video_processor(images=sampled_frames, return_tensors="pt").to(extractor.device)
                with torch.no_grad():
                    vid_out = extractor.video_model.get_image_features(**inputs)
                video_feat_window = vid_out.mean(dim=0, keepdim=True).cpu().numpy()
                
                # Audio indices
                start_sample = int((current_frame_idx - len(frame_buffer)) / fps * sr)
                end_sample = int(current_frame_idx / fps * sr)
                
                if end_sample > len(full_audio):
                    end_sample = len(full_audio)
                
                audio_chunk = full_audio[start_sample:end_sample]
                if len(audio_chunk) < 1600: 
                    audio_chunk = np.pad(audio_chunk, (0, max(0, 1600 - len(audio_chunk))))
                
                a_inputs = extractor.audio_processor(audio_chunk, sampling_rate=16000, return_tensors="pt").to(extractor.device)
                with torch.no_grad():
                    a_out = extractor.audio_model(**a_inputs)
                audio_feat_window = a_out.last_hidden_state.mean(dim=1).cpu().numpy()

                # --- Predict ---
                features = {
                    'text': text_feat,
                    'audio': audio_feat_window,
                    'video': video_feat_window
                }
                
                current_probs = predict_emotion(features)
                top_emo = max(current_probs, key=current_probs.get)
                
                current_sec = current_frame_idx / fps
                time_str = time.strftime('%H:%M:%S', time.gmtime(current_sec))
                current_msg = f"Time: {time_str}\nEmotion: {top_emo} ({current_probs[top_emo]:.2%})"
                
                # Yield update immediately after inference
                yield display_frame, current_msg, current_probs
                
                frame_buffer = []

        cap.release()
        yield None, f"Analysis Finished.\nLast segment: {current_msg}", current_probs

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"Error: {str(e)}", None

# --- Interface ---

with gr.Blocks(title="AttentionRobustV2 Real-time Demo") as app:
    gr.Markdown("# Multimodal Video Emotion Recognition (Streaming)")
    
    with gr.Row():
        # Left Column: Visuals
        with gr.Column(scale=2):
            output_image = gr.Image(label="Live Analysis Stream", type="numpy")   
            video_input = gr.Video(label="Source Video", format="mp4")
            
        # Right Column: Controls & Results
        with gr.Column(scale=1):
            output_label = gr.Label(label="Real-time Emotion", num_top_classes=6)
            output_msg = gr.Textbox(label="Status")
            text_input = gr.Textbox(label="Context (Text)", placeholder="Enter subtitles here...")
            submit_btn = gr.Button("▶ Start Analysis", variant="primary")
    
    submit_btn.click(
        fn=process_video_stream,
        inputs=[video_input, text_input],
        outputs=[output_image, output_msg, output_label]
    )


if __name__ == "__main__":
    print("Starting server on port 6006...")
    print("If you are using VS Code: Go to the 'Ports' tab (next to Terminal), add port 6006, and click the Globe icon to open.")
    app.queue().launch(server_name="0.0.0.0", server_port=6006, share=False)
