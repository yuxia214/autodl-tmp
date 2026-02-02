import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import torch
import gradio as gr
import numpy as np
import cv2
import time
import threading
import queue
from collections import deque

# Add parent directory to path to import toolkit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import FeatureExtractor
from toolkit.models.attention_robust_v2 import AttentionRobustV2
from demo import DemoArgs, extract_audio_from_video

# --- Global Components ---
MODELS_LOADED = False
extractor = None
model = None
emotion_labels = ['Neutral', 'Angry', 'Happy', 'Sad', 'Worried', 'Surprise']
EMOTION_COLORS = {
    'Neutral': '#6B7280',
    'Angry': '#EF4444',
    'Happy': '#F59E0B',
    'Sad': '#3B82F6',
    'Worried': '#8B5CF6',
    'Surprise': '#10B981'
}
EMOTION_EMOJIS = {
    'Neutral': '😐',
    'Angry': '😠',
    'Happy': '😊',
    'Sad': '😢',
    'Worried': '😟',
    'Surprise': '😲'
}

# --- Custom CSS for Beautiful UI ---
CUSTOM_CSS = """
/* 全局样式 */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    min-height: 100vh;
    padding: 20px !important;
}

/* 标题样式 */
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.3rem !important;
    letter-spacing: -0.5px;
}

.subtitle {
    text-align: center;
    color: #6B7280;
    font-size: 1rem;
    margin-bottom: 1rem;
}

/* 关键：让图片完整显示不被裁剪 */
.gr-image img {
    object-fit: contain !important;
    max-height: 100% !important;
    width: auto !important;
    margin: auto !important;
}

/* 情绪结果面板 */
.emotion-panel {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}

/* 按钮样式 */
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

/* Tab样式 */
.tabs {
    border-radius: 16px !important;
    overflow: hidden;
}

.tab-nav {
    background: #f1f5f9 !important;
    padding: 8px !important;
    border-radius: 12px !important;
    margin-bottom: 16px !important;
}

.tab-nav button {
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

/* 输入框样式 */
.gr-textbox {
    border-radius: 10px !important;
    border: 2px solid #e2e8f0 !important;
    transition: border-color 0.2s ease !important;
}

.gr-textbox:focus-within {
    border-color: #667eea !important;
}

/* 状态指示器 */
.status-box {
    background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
    border-radius: 10px;
    padding: 12px 16px;
    font-weight: 500;
    color: #3730a3;
}

/* 提示卡片 */
.tip-card {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-radius: 12px;
    padding: 16px;
    border-left: 4px solid #f59e0b;
}

.warning-card {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border-radius: 12px;
    padding: 16px;
    border-left: 4px solid #ef4444;
}

/* 实时指示动画 */
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(0.95); }
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ef4444;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.live-badge::before {
    content: '';
    width: 8px;
    height: 8px;
    background: white;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .main-title { font-size: 1.6rem !important; }
    .gradio-container { padding: 10px !important; }
    .main-content { padding: 16px; }
}

/* 隐藏不需要的元素 */
.gr-form { gap: 12px !important; }
footer { display: none !important; }
"""

def load_components():
    """加载模型组件（只执行一次）"""
    global extractor, model, MODELS_LOADED
    if MODELS_LOADED:
        return True, "✅ 模型已就绪"

    try:
        print("Loading Extractor...")
        extractor = FeatureExtractor()
        
        print("Loading Model...")
        model_args = DemoArgs()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AttentionRobustV2(model_args)
        if device == 'cuda':
            model = model.cuda()
        model.eval()
        
        MODELS_LOADED = True
        return True, "✅ 模型加载成功"
    except Exception as e:
        return False, f"❌ 模型加载失败: {str(e)}"

def predict_emotion(features):
    """预测情绪"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = {
        'audios': torch.tensor(features['audio']).float().to(device),
        'texts': torch.tensor(features['text']).float().to(device),
        'videos': torch.tensor(features['video']).float().to(device)
    }
    
    with torch.no_grad():
        _, emos_out, _, _ = model(batch)
        probs = torch.softmax(emos_out, dim=1).cpu().numpy()[0]
        
    return {label: float(prob) for label, prob in zip(emotion_labels, probs)}

def create_emotion_display(probs_dict):
    """创建美观的情绪显示HTML"""
    if probs_dict is None:
        return "<div style='text-align:center;color:#9CA3AF;padding:40px;'>等待分析...</div>"
    
    top_emotion = max(probs_dict, key=probs_dict.get)
    top_prob = probs_dict[top_emotion]
    emoji = EMOTION_EMOJIS.get(top_emotion, '🎭')
    color = EMOTION_COLORS.get(top_emotion, '#6B7280')
    
    html = f"""
    <div style="text-align:center;padding:20px;">
        <div style="font-size:4rem;margin-bottom:10px;">{emoji}</div>
        <div style="font-size:1.8rem;font-weight:700;color:{color};">{top_emotion}</div>
        <div style="font-size:1.2rem;color:#6B7280;margin-bottom:20px;">置信度: {top_prob:.1%}</div>
        <div style="text-align:left;max-width:300px;margin:auto;">
    """
    
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    for label, prob in sorted_probs:
        bar_color = EMOTION_COLORS.get(label, '#6B7280')
        emoji_small = EMOTION_EMOJIS.get(label, '')
        html += f"""
            <div style="margin:8px 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span>{emoji_small} {label}</span>
                    <span style="color:#6B7280;">{prob:.1%}</span>
                </div>
                <div style="background:#E5E7EB;border-radius:4px;height:8px;overflow:hidden;">
                    <div style="background:{bar_color};height:100%;width:{prob*100}%;border-radius:4px;transition:width 0.3s;"></div>
                </div>
            </div>
        """
    
    html += "</div></div>"
    return html

def extract_frame_features(frame_rgb, audio_chunk, text_feat):
    """从单帧和音频片段提取特征"""
    # Video feature from single frame
    inputs = extractor.video_processor(images=[frame_rgb], return_tensors="pt").to(extractor.device)
    with torch.no_grad():
        vid_out = extractor.video_model.get_image_features(**inputs)
    video_feat = vid_out.cpu().numpy()
    
    # Audio feature
    if audio_chunk is not None and len(audio_chunk) >= 1600:
        a_inputs = extractor.audio_processor(audio_chunk, sampling_rate=16000, return_tensors="pt").to(extractor.device)
        with torch.no_grad():
            a_out = extractor.audio_model(**a_inputs)
        audio_feat = a_out.last_hidden_state.mean(dim=1).cpu().numpy()
    else:
        audio_feat = np.zeros((1, 768))
    
    return {
        'text': text_feat,
        'audio': audio_feat,
        'video': video_feat
    }

# ==================== 视频文件分析 ====================

def process_video_realtime(video_path, text_input, progress=gr.Progress()):
    """实时处理视频文件 - 优化同步"""
    if not video_path:
        yield None, "请先上传视频", None, create_emotion_display(None)
        return

    if not MODELS_LOADED:
        yield None, "正在加载模型...", None, create_emotion_display(None)
        success, msg = load_components()
        if not success:
            yield None, msg, None, create_emotion_display(None)
            return

    try:
        # 1. 预处理音频
        yield None, "正在提取音频...", None, create_emotion_display(None)
        audio_path = video_path + ".wav"
        extract_audio_from_video(video_path, audio_path)

        import librosa
        try:
            full_audio, sr = librosa.load(audio_path, sr=16000)
        except:
            import torchaudio
            wav, sr = torchaudio.load(audio_path)
            if sr != 16000:
                transform = torchaudio.transforms.Resample(sr, 16000)
                wav = transform(wav)
            full_audio = wav.mean(dim=0).numpy()
            sr = 16000

        if os.path.exists(audio_path):
            os.remove(audio_path)

        # 2. 准备文本特征
        text_content = text_input if text_input else ""
        text_feat = extractor.extract_text_feature(text_content)

        # 3. 打开视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算显示尺寸
        max_width = 720
        if width > max_width:
            scale = max_width / width
            display_size = (max_width, int(height * scale))
        else:
            display_size = (width, height)

        # 分析间隔 (每0.5秒分析一次)
        analysis_interval = 0.5
        # 显示帧率 (降低到10fps以减少延迟)
        display_fps = 10
        frame_skip = max(1, int(fps / display_fps))

        current_probs = None
        last_analysis_time = 0
        frame_idx = 0

        total_duration = total_frames / fps
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = frame_idx / fps

            # 跳帧显示以保持流畅
            if frame_idx % frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(frame_rgb, display_size)

            # 在帧上叠加情绪信息
            if current_probs:
                top_emo = max(current_probs, key=current_probs.get)
                conf = current_probs[top_emo]
                # 半透明背景
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (10, 10), (220, 70), (0, 0, 0), -1)
                display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
                cv2.putText(display_frame, f"{top_emo}: {conf:.0%}",
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # 进度信息
            time_str = time.strftime('%M:%S', time.gmtime(current_time))
            total_time_str = time.strftime('%M:%S', time.gmtime(total_duration))
            status_msg = f"播放中: {time_str} / {total_time_str}"

            # 是否需要分析
            if current_time - last_analysis_time >= analysis_interval:
                start_sample = int(last_analysis_time * sr)
                end_sample = int(current_time * sr)
                end_sample = min(end_sample, len(full_audio))

                audio_chunk = full_audio[start_sample:end_sample] if start_sample < end_sample else None
                if audio_chunk is not None and len(audio_chunk) < 1600:
                    audio_chunk = np.pad(audio_chunk, (0, max(0, 1600 - len(audio_chunk))))

                features = extract_frame_features(frame_rgb, audio_chunk, text_feat)
                current_probs = predict_emotion(features)
                last_analysis_time = current_time

            # 简化的同步：基于实际经过时间
            elapsed = time.time() - start_time
            target_time = current_time
            if elapsed < target_time:
                time.sleep(min(target_time - elapsed, 0.1))

            yield display_frame, status_msg, current_probs, create_emotion_display(current_probs)

        cap.release()

        final_msg = f"分析完成 | 总时长: {total_time_str}"
        yield display_frame, final_msg, current_probs, create_emotion_display(current_probs)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"错误: {str(e)}", None, create_emotion_display(None)

# ==================== 摄像头分析 ====================

# 全局变量用于摄像头实时分析
webcam_text_context = ""
last_webcam_analysis = 0
cached_text_feat = None

def process_webcam_stream(frame, text_input):
    """实时处理摄像头视频流"""
    global webcam_text_context, last_webcam_analysis, cached_text_feat

    if frame is None:
        return None, "等待摄像头画面...", create_emotion_display(None)

    if not MODELS_LOADED:
        success, msg = load_components()
        if not success:
            return frame, msg, create_emotion_display(None)

    try:
        current_time = time.time()

        # 转换颜色空间
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = frame if frame.dtype == np.uint8 else (frame * 255).astype(np.uint8)
        else:
            frame_rgb = frame

        # 限制分析频率 (每0.5秒分析一次，避免卡顿)
        if current_time - last_webcam_analysis < 0.5:
            return frame_rgb, "实时分析中...", None  # 返回None表示不更新情绪显示

        last_webcam_analysis = current_time

        # 文本特征（缓存，只在文本变化时重新计算）
        text_content = text_input if text_input else ""
        if text_content != webcam_text_context or cached_text_feat is None:
            webcam_text_context = text_content
            cached_text_feat = extractor.extract_text_feature(text_content)

        # 摄像头没有音频，使用零向量
        audio_feat = np.zeros((1, 768))

        # 视频特征
        inputs = extractor.video_processor(images=[frame_rgb], return_tensors="pt").to(extractor.device)
        with torch.no_grad():
            vid_out = extractor.video_model.get_image_features(**inputs)
        video_feat = vid_out.cpu().numpy()

        features = {
            'text': cached_text_feat,
            'audio': audio_feat,
            'video': video_feat
        }

        probs = predict_emotion(features)
        top_emo = max(probs, key=probs.get)

        # 在画面上叠加情绪信息
        display_frame = frame_rgb.copy()
        h, w = display_frame.shape[:2]

        # 绘制半透明背景
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (220, 70), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)

        # 绘制情绪文字
        cv2.putText(display_frame, f"{top_emo}: {probs[top_emo]:.0%}",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        status = f"实时分析中 | {top_emo} ({probs[top_emo]:.0%})"
        return display_frame, status, create_emotion_display(probs)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return frame, f"分析错误: {str(e)}", create_emotion_display(None)

# ==================== Gradio界面 ====================

def create_interface():
    with gr.Blocks(css=CUSTOM_CSS, title="多模态情绪识别系统", theme=gr.themes.Soft()) as app:

        # 标题区域
        gr.HTML("""
            <div style="text-align:center;padding:15px 0 10px 0;">
                <h1 class="main-title">多模态情绪识别系统</h1>
                <p class="subtitle">基于 AttentionRobustV2 | 视觉 + 音频 + 文本 多模态融合</p>
            </div>
        """)

        # 模型状态栏
        with gr.Row():
            model_status = gr.Textbox(
                value="模型未加载 - 首次分析时自动加载",
                label="",
                interactive=False,
                scale=5,
                elem_classes=["status-box"]
            )
            load_btn = gr.Button("预加载模型", variant="secondary", scale=1, size="sm")

        gr.HTML("<div style='height:10px'></div>")

        # Tab选项卡
        with gr.Tabs() as tabs:

            # ========== Tab 1: 视频文件分析 ==========
            with gr.TabItem("视频文件分析", id=0):
                # 上方：上传和控制
                with gr.Row():
                    video_input = gr.Video(
                        label="上传视频",
                        format="mp4",
                        height=100,
                        scale=2
                    )
                    text_input_video = gr.Textbox(
                        label="文本内容（可选）",
                        placeholder="输入视频中的对话内容...",
                        lines=2,
                        scale=2
                    )
                    with gr.Column(scale=1):
                        analyze_btn = gr.Button(
                            "开始分析",
                            variant="primary",
                            size="lg"
                        )
                        video_status = gr.Textbox(
                            value="等待上传视频",
                            label="",
                            interactive=False,
                            lines=1
                        )

                # 下方：视频画面和情绪结果并排
                with gr.Row():
                    video_display = gr.Image(
                        label="实时分析画面",
                        type="numpy",
                        scale=3
                    )
                    with gr.Column(scale=2):
                        gr.HTML("<h3 style='margin:0 0 12px 0;color:#374151;'>情绪分析结果</h3>")
                        emotion_html = gr.HTML(create_emotion_display(None))
                        emotion_output = gr.Label(
                            label="情绪概率分布",
                            num_top_classes=6,
                            visible=False
                        )

            # ========== Tab 2: 摄像头实时分析 ==========
            with gr.TabItem("摄像头实时分析", id=1):
                # 提示
                gr.HTML("""
                    <div class="warning-card" style="margin-bottom:12px;">
                        <strong>提示：</strong>摄像头需要 HTTPS 连接，请使用 <code>--share</code> 启动。启用摄像头后自动开始实时分析。
                    </div>
                """)

                # 上方：文本输入和状态
                with gr.Row():
                    text_input_webcam = gr.Textbox(
                        label="文本内容（可选）",
                        placeholder="输入当前情境描述...",
                        lines=1,
                        scale=3
                    )
                    webcam_status = gr.Textbox(
                        value="启用摄像头后自动开始分析",
                        label="状态",
                        interactive=False,
                        lines=1,
                        scale=2
                    )

                # 下方：摄像头画面和情绪结果
                with gr.Row():
                    with gr.Column(scale=3):
                        webcam_input = gr.Image(
                            label="摄像头输入（点击启用）",
                            source="webcam",
                            streaming=True,
                            type="numpy"
                        )
                        webcam_output = gr.Image(
                            label="分析结果（带情绪标注）",
                            type="numpy"
                        )

                    with gr.Column(scale=2):
                        gr.HTML("<h3 style='margin:0 0 12px 0;color:#374151;'>实时情绪</h3>")
                        webcam_emotion_html = gr.HTML(create_emotion_display(None))

                        gr.HTML("""
                            <div class="tip-card" style="margin-top:20px;">
                                <strong>使用说明：</strong><br>
                                1. 点击摄像头区域启用<br>
                                2. 自动实时分析情绪<br>
                                3. 结果叠加在下方画面
                            </div>
                        """)

        # 使用说明折叠区
        with gr.Accordion("使用说明", open=False):
            gr.HTML("""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:10px;">
                    <div>
                        <h4 style="color:#667eea;margin-bottom:10px;">功能介绍</h4>
                        <ul style="color:#4b5563;line-height:1.8;">
                            <li><strong>视频文件分析</strong>：上传视频，实时显示情绪分析</li>
                            <li><strong>摄像头实时分析</strong>：像直播一样实时分析情绪</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color:#667eea;margin-bottom:10px;">支持的情绪</h4>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;color:#4b5563;">
                            <span>Neutral 中性</span><span>Angry 愤怒</span>
                            <span>Happy 开心</span><span>Sad 悲伤</span>
                            <span>Worry 担忧</span><span>Surprise 惊讶</span>
                        </div>
                    </div>
                </div>
            """)

        # ========== 事件绑定 ==========

        def preload_model():
            success, msg = load_components()
            return msg

        load_btn.click(
            fn=preload_model,
            outputs=model_status
        )

        analyze_btn.click(
            fn=process_video_realtime,
            inputs=[video_input, text_input_video],
            outputs=[video_display, video_status, emotion_output, emotion_html]
        )

        # 摄像头实时流处理
        webcam_input.stream(
            fn=process_webcam_stream,
            inputs=[webcam_input, text_input_webcam],
            outputs=[webcam_output, webcam_status, webcam_emotion_html]
        )

    return app


# ==================== 主程序 ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true', help='生成公网HTTPS链接（摄像头功能需要）')
    parser.add_argument('--port', type=int, default=6006, help='服务端口')
    args = parser.parse_args()

    # 设置环境变量，避免从CDN加载资源导致卡住
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    print("=" * 60)
    print("多模态情绪识别系统")
    print("=" * 60)
    print(f"本地地址: http://localhost:{args.port}")
    if args.share:
        print("正在生成公网HTTPS链接...")
    else:
        print("提示: 使用 --share 参数可生成HTTPS公网链接（摄像头功能需要）")
    print("=" * 60)

    app = create_interface()
    app.queue(concurrency_count=1).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        favicon_path=None
    )
