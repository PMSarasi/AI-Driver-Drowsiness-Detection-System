import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
from collections import deque
import sys
import os
import threading
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Driver Drowsiness Detection | AI Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fixed Navigation Tabs
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }
    
    /* Fix navigation tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        padding: 8px;
        border-radius: 50px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        padding: 12px 30px;
        border-radius: 40px;
        color: rgba(255,255,255,0.7);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0,198,255,0.1);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
        color: white !important;
        font-weight: 600;
    }
    
    .metric-card { 
        background: rgba(255, 255, 255, 0.05); 
        border-radius: 12px; 
        padding: 15px; 
        backdrop-filter: blur(10px); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        margin-bottom: 12px; 
    }
    
    .main-title { 
        font-size: 1.8rem; 
        font-weight: 700; 
        background: linear-gradient(90deg, #00c6ff, #0072ff); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin: 0; 
    }
    
    .alert-box { 
        background: rgba(255, 50, 50, 0.2); 
        border-left: 5px solid #ff3333; 
        padding: 12px; 
        border-radius: 10px; 
        animation: pulse 1.5s infinite; 
    }
    
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
    
    .metric-value { font-size: 1.8rem; font-weight: 700; color: white; }
    .metric-label { font-size: 0.8rem; color: rgba(255, 255, 255, 0.6); text-transform: uppercase; letter-spacing: 1px; }
    .model-badge { background: linear-gradient(90deg, #00c6ff, #0072ff); border-radius: 20px; padding: 4px 12px; display: inline-block; color: white; font-size: 0.75rem; }
    .status-badge { padding: 4px 12px; border-radius: 8px; color: white; font-weight: bold; display: inline-block; font-size: 0.9rem; }
    
    /* Scrollable right panel */
    .scrollable-panel {
        max-height: calc(100vh - 250px);
        overflow-y: auto;
        padding-right: 10px;
    }
    
    .scrollable-panel::-webkit-scrollbar { width: 6px; }
    .scrollable-panel::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 10px; }
    .scrollable-panel::-webkit-scrollbar-thumb { background: rgba(0,198,255,0.5); border-radius: 10px; }
    
    /* Reduce spacing */
    .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
    
    /* Button styling */
    .stButton > button {
        border-radius: 30px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,198,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
if 'log_data' not in st.session_state:
    st.session_state.log_data = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'risk_history' not in st.session_state:
    st.session_state.risk_history = deque(maxlen=100)
if 'blink_count' not in st.session_state:
    st.session_state.blink_count = 0
if 'yawn_count' not in st.session_state:
    st.session_state.yawn_count = 0
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0
if 'session_start' not in st.session_state:
    st.session_state.session_start = time.time()
if 'eye_was_closed' not in st.session_state:
    st.session_state.eye_was_closed = False
if 'was_yawning' not in st.session_state:
    st.session_state.was_yawning = False
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
if 'latest_results' not in st.session_state:
    st.session_state.latest_results = None
if 'fps_history' not in st.session_state:
    st.session_state.fps_history = deque(maxlen=30)
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = deque(maxlen=30)

# ============================================
# SIDEBAR - COMPACT CONTROL PANEL
# ============================================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    with st.expander("🎯 Fusion Weights", expanded=True):
        cnn_weight_raw = st.slider("CNN", 0.0, 1.0, 0.35, 0.05)
        ear_weight_raw = st.slider("EAR", 0.0, 1.0, 0.35, 0.05)
        mar_weight_raw = st.slider("MAR", 0.0, 1.0, 0.30, 0.05)
        
        total = cnn_weight_raw + ear_weight_raw + mar_weight_raw
        if total > 0:
            cnn_weight = cnn_weight_raw / total
            ear_weight = ear_weight_raw / total
            mar_weight = mar_weight_raw / total
        else:
            cnn_weight = ear_weight = mar_weight = 1.0/3.0
        st.caption(f"CNN({cnn_weight:.0%}) EAR({ear_weight:.0%}) MAR({mar_weight:.0%})")
    
    with st.expander("🚨 Alert Settings", expanded=True):
        alert_threshold = st.slider("Threshold", 30, 90, 60, 5)
        alert_cooldown = st.slider("Cooldown (s)", 1, 10, 3)
    
    with st.expander("⚡ Performance", expanded=True):
        show_landmarks = st.checkbox("Show Landmarks", True)
        if st.session_state.fps_history:
            st.metric("FPS", f"{np.mean(st.session_state.fps_history):.1f}")
        st.metric("Frames", len(st.session_state.risk_history))
    
    with st.expander("🤖 Models", expanded=True):
        st.success("CNN: 96.25%")
        st.success("Eye: 98.97%")
        st.success("Yawn: 99.31%")
    
    st.markdown("---")
    st.caption("🚗 v3.0 | AI Driver Monitor")

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    with st.spinner("🔄 Loading AI Models..."):
        from src.pipeline.face_detector import FaceDetector
        from src.pipeline.model_inference import ModelInference
        from src.fusion.temporal_analyzer import TemporalAnalyzer
        from src.fusion.risk_scorer import RiskScorer
        from src.features.eye_aspect_ratio import calculate_ear, get_eye_landmarks, LEFT_EYE_SIMPLE, RIGHT_EYE_SIMPLE
        from src.features.mouth_aspect_ratio import calculate_mar, get_mouth_landmarks, MOUTH_INDICES
        
        return (FaceDetector(), ModelInference(), TemporalAnalyzer(), RiskScorer(),
                calculate_ear, get_eye_landmarks, LEFT_EYE_SIMPLE, RIGHT_EYE_SIMPLE,
                calculate_mar, get_mouth_landmarks, MOUTH_INDICES)

try:
    (face_detector, model_inference, temporal_analyzer, risk_scorer,
     calculate_ear, get_eye_landmarks, LEFT_EYE_SIMPLE, RIGHT_EYE_SIMPLE,
     calculate_mar, get_mouth_landmarks, MOUTH_INDICES) = load_models()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

risk_scorer.CNN_WEIGHT = cnn_weight
risk_scorer.EAR_WEIGHT = ear_weight
risk_scorer.MAR_WEIGHT = mar_weight

# ============================================
# HELPER FUNCTIONS
# ============================================
def play_alert_sound():
    def _play():
        try:
            import winsound
            winsound.Beep(1000, 300)
        except: pass
    threading.Thread(target=_play, daemon=True).start()

def get_status_color(risk_score):
    if risk_score < 30: return "#00ff88", "🟢 SAFE", (0, 255, 0)
    elif risk_score < 50: return "#ffaa00", "🟡 CAUTION", (0, 255, 255)
    elif risk_score < 70: return "#ff6600", "🟠 WARNING", (0, 165, 255)
    else: return "#ff3333", "🔴 DANGER", (0, 0, 255)

# ============================================
# MAIN TABS - STYLED NAVIGATION
# ============================================
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 LIVE MONITOR", "🧠 HOW IT WORKS", "📄 SESSION REPORT"])

# ============================================
# TAB 1: LIVE MONITOR
# ============================================
with tab1:
    # Header with controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown('<p class="main-title">🚗 Driver Drowsiness AI</p>', unsafe_allow_html=True)
    with col2:
        if st.session_state.camera_active:
            st.markdown('<span class="model-badge" style="background:#00ff88; color:#000;">🔴 LIVE</span>', unsafe_allow_html=True)
    with col3:
        if not st.session_state.camera_active:
            start_btn = st.button("▶️ START", use_container_width=True, type="primary")
        else:
            start_btn = st.button("▶️ START", use_container_width=True, disabled=True)
        if start_btn:
            st.session_state.camera_active = True
            st.session_state.session_start = time.time()
            st.rerun()
    with col4:
        if st.session_state.camera_active:
            if st.button("⏹️ STOP", use_container_width=True):
                st.session_state.camera_active = False
                st.rerun()
        else:
            st.button("⏹️ STOP", use_container_width=True, disabled=True)
    
    st.markdown("---")
    
    # Main layout
    left_col, right_col = st.columns([1.2, 1])
    
    with left_col:
        st.markdown("### 📹 Camera Feed")
        camera_container = st.empty()
        status_bar = st.empty()
    
    with right_col:
        st.markdown("### 📊 Live Metrics")
        with st.container():
            st.markdown('<div class="scrollable-panel">', unsafe_allow_html=True)
            risk_placeholder = st.empty()
            metrics_container = st.empty()
            alert_container = st.empty()
            conf_container = st.empty()
            explanation_container = st.empty()
            chart_container = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

    # Camera loop
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Camera access failed")
            st.session_state.camera_active = False
        else:
            frame_start_time = time.time()
            
            while st.session_state.camera_active:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (640, 480))
                h, w = frame.shape[:2]
                display_frame = frame.copy()
                
                landmarks = face_detector.detect(frame)
                
                if landmarks is not None:
                    left_eye = get_eye_landmarks(landmarks, LEFT_EYE_SIMPLE)
                    right_eye = get_eye_landmarks(landmarks, RIGHT_EYE_SIMPLE)
                    mouth = get_mouth_landmarks(landmarks, MOUTH_INDICES)
                    
                    ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
                    mar = calculate_mar(mouth)
                    
                    face_roi, bbox = face_detector.get_face_roi(frame, landmarks)
                    cnn_prob = model_inference.predict_cnn(face_roi)
                    
                    smooth_ear, smooth_mar, smooth_cnn = temporal_analyzer.update(ear, mar, cnn_prob)
                    fusion_risk, cnn_risk, ear_risk, mar_risk = risk_scorer.calculate_fusion_risk(
                        smooth_cnn, smooth_ear, smooth_mar
                    )
                    
                    st.session_state.risk_history.append(fusion_risk)
                    st.session_state.log_data.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "risk": fusion_risk, "ear": smooth_ear, "mar": smooth_mar, "cnn": smooth_cnn
                    })
                    
                    eye_closed = smooth_ear < 0.2
                    if eye_closed and not st.session_state.eye_was_closed:
                        st.session_state.blink_count += 1
                    st.session_state.eye_was_closed = eye_closed
                    
                    is_yawning = smooth_mar > 0.6
                    if is_yawning and not st.session_state.was_yawning:
                        st.session_state.yawn_count += 1
                    st.session_state.was_yawning = is_yawning
                    
                    eye_closed_duration = temporal_analyzer.detect_eye_closure_duration(1 - smooth_ear)
                    is_drowsy = fusion_risk >= alert_threshold or eye_closed_duration > 2.0
                    
                    if is_drowsy and (time.time() - st.session_state.last_alert_time) > alert_cooldown:
                        st.session_state.alert_count += 1
                        st.session_state.last_alert_time = time.time()
                        play_alert_sound()
                    
                    status_color, status_text, color_bgr = get_status_color(fusion_risk)
                    
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_bgr, 2)
                        cv2.putText(display_frame, f"{status_text} ({fusion_risk:.0f})", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                    
                    if show_landmarks:
                        for pt in left_eye: cv2.circle(display_frame, (int(pt[0]*w), int(pt[1]*h)), 2, (0,255,0), -1)
                        for pt in right_eye: cv2.circle(display_frame, (int(pt[0]*w), int(pt[1]*h)), 2, (0,255,0), -1)
                    
                    st.session_state.latest_results = {
                        'ear': smooth_ear, 'mar': smooth_mar, 'cnn_prob': smooth_cnn,
                        'fusion_risk': fusion_risk, 'cnn_risk': cnn_risk, 'ear_risk': ear_risk, 'mar_risk': mar_risk,
                        'eye_closed': eye_closed, 'is_yawning': is_yawning, 'is_drowsy': is_drowsy,
                        'status_color': status_color, 'status_text': status_text
                    }
                else:
                    st.session_state.latest_results = None
                    cv2.putText(display_frame, "No Face Detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # FPS calculation
                frame_time = time.time() - frame_start_time
                if frame_time > 0:
                    st.session_state.fps_history.append(1.0 / frame_time)
                frame_start_time = time.time()
                
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                camera_container.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Status bar
                avg_fps = np.mean(st.session_state.fps_history) if st.session_state.fps_history else 0
                session_time = int(time.time() - st.session_state.session_start)
                status_bar.caption(f"🎯 FPS: {avg_fps:.1f} | ⏱️ {session_time//60}m {session_time%60}s | 👀 {st.session_state.blink_count} | 😮 {st.session_state.yawn_count}")
                
                if st.session_state.latest_results:
                    r = st.session_state.latest_results
                    
                    with risk_placeholder.container():
                        st.markdown(f"""
                        <div style="background:rgba(0,0,0,0.3); border-radius:12px; padding:15px; text-align:center; border:2px solid {r['status_color']};">
                            <span class="metric-label">Drowsiness Score</span>
                            <div class="metric-value" style="color: {r['status_color']};">{r['fusion_risk']:.0f}/100</div>
                            <div class="status-badge" style="background: {r['status_color']};">{r['status_text']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(r['fusion_risk'] / 100)
                    
                    with metrics_container.container():
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("👁️ Eye", "CLOSED" if r['eye_closed'] else "OPEN", delta=f"EAR: {r['ear']:.2f}")
                        m2.metric("😮 Yawn", "YES" if r['is_yawning'] else "NO", delta=f"MAR: {r['mar']:.2f}")
                        m3.metric("🧠 CNN", f"{r['cnn_prob']:.0%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with alert_container.container():
                        if r['is_drowsy']:
                            st.markdown(f"""
                            <div class="alert-box">
                                <h3 style="color:#ff3333; margin:0;">🚨 DROWSINESS ALERT!</h3>
                                <p style="margin:5px 0 0;">Risk: {r['fusion_risk']:.0f}/100</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif r['fusion_risk'] >= 30:
                            st.warning(f"⚠️ {r['status_text']} - Risk: {r['fusion_risk']:.0f}/100")
                        else:
                            st.success(f"✅ {r['status_text']} - Risk: {r['fusion_risk']:.0f}/100")
                    
                    with conf_container.container():
                        st.markdown("**Model Confidence**")
                        st.progress(r['cnn_prob'], text=f"CNN: {r['cnn_prob']:.0%}")
                        st.progress(r['ear_risk']/100, text=f"Eye: {r['ear_risk']:.0f}")
                        st.progress(r['mar_risk']/100, text=f"Yawn: {r['mar_risk']:.0f}")
                    
                    with explanation_container.container():
                        reasons = []
                        if r['ear'] < 0.2: reasons.append("Eyes closed")
                        if r['mar'] > 0.6: reasons.append("Yawning")
                        if r['cnn_prob'] > 0.5: reasons.append("CNN drowsy")
                        
                        st.info(f"**🧠 Why?** {', '.join(reasons) if reasons else 'Driver alert'} | EAR:{r['ear']:.2f} MAR:{r['mar']:.2f} CNN:{r['cnn_prob']:.0%}")
                    
                    with chart_container.container():
                        if len(st.session_state.risk_history) > 5:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=list(st.session_state.risk_history), mode='lines',
                                line=dict(color='#00c6ff', width=2), fill='tozeroy', fillcolor='rgba(0,198,255,0.1)'
                            ))
                            fig.add_hline(y=alert_threshold, line_dash="dash", line_color="#ff3333")
                            fig.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0),
                                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                             yaxis=dict(range=[0, 100]), showlegend=False)
                            chart_container.plotly_chart(fig, use_container_width=True)
                else:
                    with alert_container.container():
                        st.warning("⚠️ Face not detected")
                
                time.sleep(0.03)
            
            cap.release()
    else:
        camera_container.markdown("""
        <div style="background:rgba(0,0,0,0.3); border-radius:15px; padding:60px; text-align:center;">
            <h3>📷 Camera Off</h3><p>Click START to begin</p>
        </div>
        """, unsafe_allow_html=True)
        risk_placeholder.info("Click ▶️ START")
        metrics_container.info("Ready")
        alert_container.success("System Ready")

# ============================================
# TAB 2: HOW IT WORKS
# ============================================
with tab2:
    st.title("🧠 How the AI Works")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🔄 Detection Pipeline
        
        1. **📷 Camera Input** - Webcam captures at 30 FPS
        2. **👤 Face Detection** - MediaPipe 468 landmarks
        3. **👁️ Eye Analysis** - EAR (Eye Aspect Ratio)
        4. **😮 Mouth Analysis** - MAR (Mouth Aspect Ratio)
        5. **🧠 CNN Analysis** - MobileNetV2 on face
        6. **⏱️ Temporal Smoothing** - Rolling average
        7. **⚖️ Fusion Scoring** - Weighted combination
        8. **🚨 Alert Decision** - Threshold-based warning
        """)
    with col2:
        st.markdown(f"""
        ### 🎯 Fusion Formula
        
        **Risk = {cnn_weight:.0%} × CNN + {ear_weight:.0%} × EAR + {mar_weight:.0%} × MAR**
        
        ### 📊 Risk Levels
        - 🟢 0-30: SAFE
        - 🟡 30-50: CAUTION
        - 🟠 50-70: WARNING
        - 🔴 70-100: DANGER
        
        ### 🤖 Model Accuracy
        - CNN: **96.25%**
        - Eye Detection: **98.97%**
        - Yawn Detection: **99.31%**
        """)

# ============================================
# TAB 3: SESSION REPORT
# ============================================
with tab3:
    st.title("📄 Session Report")
    st.markdown("---")
    
    if st.session_state.log_data:
        df = pd.DataFrame(st.session_state.log_data)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Avg Risk", f"{df['risk'].mean():.1f}")
        col2.metric("📈 Max Risk", f"{df['risk'].max():.0f}")
        col3.metric("🚨 Alerts", st.session_state.alert_count)
        col4.metric("👀 Blinks", st.session_state.blink_count)
        
        st.markdown("---")
        st.line_chart(df["risk"], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download CSV", df.to_csv(index=False), "report.csv", use_container_width=True)
        with col2:
            if st.button("🔄 Reset Session", use_container_width=True):
                for key in ['blink_count', 'yawn_count', 'alert_count']:
                    st.session_state[key] = 0
                st.session_state.risk_history.clear()
                st.session_state.log_data.clear()
                st.rerun()
    else:
        st.warning("⚠️ No data yet. Run the Live Monitor first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.4);">
    🚗 AI Driver Drowsiness Detection | CNN(96%) Eye(99%) Yawn(99%)
</div>
""", unsafe_allow_html=True)
