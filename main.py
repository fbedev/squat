import cv2, pkg_resources
print(">>> CV2 version:", cv2.__version__)
try:
    dist = pkg_resources.get_distribution("opencv-python-headless")
    print(">>> Loaded from:", dist)
except Exception as e:
    print(">>> opencv-python-headless not found!", e)


# app.py
# ---------------------------------------------------------------
# Squat posture live analyzer — Streamlit + OpenCV (cv2) + MediaPipe
# "Grandma Mode" edition: slower, calmer, bigger, clearer.
# ---------------------------------------------------------------
# Features
#   • Live webcam via WebRTC (browser) -> Python backend processing with cv2
#   • MediaPipe Pose landmark tracking
#   • Real‑time angles (knee / hip / ankle) and simple depth check
#   • **Grandma Mode**: high‑contrast BIG UI, slower cadence, extra smoothing,
#     simple messages ("Sit", "Stand tall", "Good"), optional beeps.
#   • Hold‑to‑count (requires staying at bottom/top for N ms to avoid jitter)
#   • FPS cap for stability + lower CPU
#   • Configurable thresholds in the sidebar
#
# Run:
#   1) pip install -r requirements.txt  (see requirements below)
#   2) streamlit run app.py
#   3) Allow camera access in the browser tab that opens
#
# Requirements (put these lines into a requirements.txt):
#   streamlit
#   streamlit-webrtc
#   opencv-python
#   av
#   numpy
#   mediapipe==0.10.14
# ---------------------------------------------------------------

import math
import time
from collections import deque
from dataclasses import dataclass

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Optional: silence MediaPipe verbose logs
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import mediapipe as mp
except Exception as e:
    st.error("Failed to import MediaPipe. Did you install requirements?`pip install mediapipe==0.10.14`")
    raise e

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# --------------------------- Geometry Utils ---------------------------

def to_xy(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)


def angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees for points a, b, c (2D)."""
    BA = a - b
    BC = c - b
    # Normalize
    if np.linalg.norm(BA) < 1e-6 or np.linalg.norm(BC) < 1e-6:
        return float("nan")
    BA = BA / (np.linalg.norm(BA) + 1e-9)
    BC = BC / (np.linalg.norm(BC) + 1e-9)
    cosang = float(np.clip(np.dot(BA, BC), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


# --------------------------- Config Model -----------------------------

@dataclass
class Thresholds:
    knee_down_deg: float = 100.0   # knee angle <= this -> considered deep enough
    knee_up_deg: float = 165.0     # knee angle >= this -> considered back to top
    hip_depth_margin_px: float = 8.0  # hip below knee by at least this many px
    smooth_n: int = 4              # moving average over last N frames
    hold_ms_bottom: int = 400      # must stay at bottom ≥ this
    hold_ms_top: int = 400         # must stay at top ≥ this
    fps_cap: int = 20              # cap FPS for calmer UI


# --------------------------- Streamlit UI -----------------------------

st.set_page_config(page_title="Squat Posture Analyzer", page_icon="🏋️", layout="wide")
st.title("🏋️ Squat Posture Live Analyzer — OpenCV + MediaPipe")
st.caption("Side view recommended · Educational only · Uses your browser webcam (WebRTC)")

with st.sidebar:
    st.header("Settings")
    side = st.radio("Tracking side", ["left", "right"], index=0, help="Pick the leg to analyze (camera should see this side clearly).")
    mirror = st.checkbox("Mirror preview (selfie)", value=True)

    st.subheader("Thresholds")
    knee_down_deg = st.slider("Knee angle at bottom (≤)", 60, 140, 100, 1, help="If knee angle goes below/equals this, we consider you at the bottom.")
    knee_up_deg = st.slider("Knee angle at top (≥)", 140, 180, 165, 1, help="If knee angle goes above/equals this after a bottom, we count 1 rep.")
    hip_margin = st.slider("Hip-below-knee margin (px)", 0, 30, 8, 1, help="Extra pixels hip should be below knee to confirm depth (side view).")
    smooth_n = st.slider("Angle smoothing (frames)", 1, 20, 8, 1, help="Moving average to reduce jitter.")
    hold_ms_bottom = st.slider("Hold at bottom (ms)", 0, 2000, 600, 50)
    hold_ms_top = st.slider("Hold at top (ms)", 0, 2000, 700, 50)
    fps_cap = st.slider("FPS cap", 5, 30, 20, 1)

    st.subheader("Display")
    grandma_mode = st.checkbox("Grandma Mode (big & simple)", value=True)
    show_angles = st.checkbox("Show angles (advanced)", value=not grandma_mode)
    show_lines = st.checkbox("Show helper lines", value=not grandma_mode)
    show_skeleton = st.checkbox("Show skeleton", value=not grandma_mode)
    audible_beep = st.checkbox("Beep at bottom/top", value=True, help="Short beeps when you reach bottom/top.")

    st.markdown("""
    **Tip**: Place camera ~3–4m away at hip height, perpendicular to your side.
    Keep entire body in frame. Wear contrasting clothes.
    """)

# apply presets when Grandma Mode is on (safer, calmer)
if grandma_mode:
    smooth_n = max(smooth_n, 10)
    hold_ms_bottom = max(hold_ms_bottom, 700)
    hold_ms_top = max(hold_ms_top, 800)
    fps_cap = min(fps_cap, 20)

th = Thresholds(
    knee_down_deg=knee_down_deg,
    knee_up_deg=knee_up_deg,
    hip_depth_margin_px=float(hip_margin),
    smooth_n=smooth_n,
    hold_ms_bottom=hold_ms_bottom,
    hold_ms_top=hold_ms_top,
    fps_cap=fps_cap,
)

# --------------------------- Helper: Beeps ----------------------------

def _beep(kind: str = "low"):
    """Very small synthetic beep using OpenCV's waitKey timing + NumPy (host-side).
    It's intentionally minimal; if it fails on some systems, it quietly does nothing.
    """
    try:
        import sounddevice as sd
        sr = 22050
        dur = 0.08 if kind == "low" else 0.12
        f = 650 if kind == "low" else 880
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        wave = 0.2*np.sin(2*np.pi*f*t).astype(np.float32)
        sd.play(wave, sr)
    except Exception:
        pass


# --------------------------- Video Processor --------------------------

class SquatProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose( model_complexity=1,
                                  enable_segmentation=False,
                                  smooth_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5 )
        self.rep_count = 0
        self.state = "top"  # "top" or "bottom"
        self.last_state_change_ms = int(time.time()*1000)
        self.side = "left"
        self.mirror = True
        self.th = th
        self.grandma_mode = True
        self.show_angles = True
        self.show_lines = True
        self.show_skeleton = True
        self.audible_beep = True
        # smoothing buffers
        self.knee_buf = deque(maxlen=self.th.smooth_n)
        self.hip_buf = deque(maxlen=self.th.smooth_n)
        self.ankle_buf = deque(maxlen=self.th.smooth_n)
        # fps capping
        self._last_frame_time = 0.0

    def _smooth(self, buf: deque, val: float) -> float:
        if not (val is None or math.isnan(val)):
            buf.append(val)
        if len(buf) == 0:
            return float("nan")
        return float(np.nanmean(np.array(buf, dtype=np.float32)))

    def _draw_text(self, img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
        # Black outline for readability
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _draw_big_status(self, img, status_text: str, color: tuple):
        h, w, _ = img.shape
        # Big banner
        pad = 20
        cv2.rectangle(img, (pad, h-140), (w-pad, h-20), (0, 0, 0), -1)
        cv2.rectangle(img, (pad+2, h-138), (w-pad-2, h-22), color, 3)
        # Centered text
        scale = 1.3
        size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
        x = (w - size[0]) // 2
        self._draw_text(img, status_text, (x, h - 70), scale=scale, color=color, thick=3)

    def _now_ms(self):
        return int(time.time()*1000)

    def _enough_hold(self, phase: str) -> bool:
        elapsed = self._now_ms() - self.last_state_change_ms
        need = self.th.hold_ms_top if phase == "top" else self.th.hold_ms_bottom
        return elapsed >= need

    def _cap_fps(self):
        # Keep processing calmer & consistent
        target_dt = 1.0 / max(5, self.th.fps_cap)
        now = time.time()
        if self._last_frame_time == 0:
            self._last_frame_time = now
            return
        dt = now - self._last_frame_time
        if dt < target_dt:
            time.sleep(target_dt - dt)
        self._last_frame_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._cap_fps()
        img = frame.to_ndarray(format="bgr24")

        if self.mirror:
            img = cv2.flip(img, 1)

        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        knee_ang = hip_ang = ank_ang = float("nan")
        depth_ok = False
        cues = []

        if res.pose_landmarks:
            if self.show_skeleton and not self.grandma_mode:
                mp_drawing.draw_landmarks(
                    img,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

            lm = res.pose_landmarks.landmark
            if self.side == "left":
                SH, HIP, KNEE, ANK, TOE = (mp_pose.PoseLandmark.LEFT_SHOULDER,
                                           mp_pose.PoseLandmark.LEFT_HIP,
                                           mp_pose.PoseLandmark.LEFT_KNEE,
                                           mp_pose.PoseLandmark.LEFT_ANKLE,
                                           mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
            else:
                SH, HIP, KNEE, ANK, TOE = (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                           mp_pose.PoseLandmark.RIGHT_HIP,
                                           mp_pose.PoseLandmark.RIGHT_KNEE,
                                           mp_pose.PoseLandmark.RIGHT_ANKLE,
                                           mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

            sh_xy = to_xy(lm[SH], w, h)
            hip_xy = to_xy(lm[HIP], w, h)
            knee_xy = to_xy(lm[KNEE], w, h)
            ank_xy = to_xy(lm[ANK], w, h)
            toe_xy = to_xy(lm[TOE], w, h)

            # angles
            knee_ang = angle_abc(hip_xy, knee_xy, ank_xy)
            hip_ang = angle_abc(sh_xy, hip_xy, knee_xy)
            ank_ang = angle_abc(knee_xy, ank_xy, toe_xy)

            # smoothing
            knee_ang_s = self._smooth(self.knee_buf, knee_ang)
            hip_ang_s  = self._smooth(self.hip_buf, hip_ang)
            ank_ang_s  = self._smooth(self.ankle_buf, ank_ang)

            # Depth check: hip (y) below knee (y) by margin
            depth_ok = (hip_xy[1] - knee_xy[1]) > self.th.hip_depth_margin_px

            # Simple knee travel metric (side view): knee ahead of toe in x
            knee_over_toe_px = knee_xy[0] - toe_xy[0]

            # State machine with hold requirements
            # target phases: "top" and "bottom"
            if not math.isnan(knee_ang_s):
                if self.state == "top":
                    # Going down condition
                    if knee_ang_s <= self.th.knee_down_deg and depth_ok and self._enough_hold("top"):
                        self.state = "bottom"
                        self.last_state_change_ms = self._now_ms()
                        if self.audible_beep:
                            _beep("low")
                else:  # currently bottom
                    if knee_ang_s >= self.th.knee_up_deg and self._enough_hold("bottom"):
                        self.state = "top"
                        self.last_state_change_ms = self._now_ms()
                        self.rep_count += 1
                        if self.audible_beep:
                            _beep("high")

            # Visuals (minimal in Grandma Mode)
            if self.show_lines and not self.grandma_mode:
                # vertical helper through ankle
                cv2.line(img, (int(ank_xy[0]), 0), (int(ank_xy[0]), h), (60, 60, 60), 2)
                # knee-to-toe horizontal
                cv2.line(img, (int(toe_xy[0]), int(knee_xy[1])), (int(knee_xy[0]), int(knee_xy[1])), (100, 180, 255), 3)
                # hip and knee y lines (depth reference)
                cv2.line(img, (0, int(knee_xy[1])), (w, int(knee_xy[1])), (200, 200, 200), 2)
                cv2.line(img, (0, int(hip_xy[1])), (w, int(hip_xy[1])), (0, 200, 255) if depth_ok else (120, 120, 120), 2)

            if self.show_angles and not self.grandma_mode:
                self._draw_text(img, f"Knee: {knee_ang_s:5.1f}°", (10, 36), scale=1.0)
                self._draw_text(img, f"Hip:  {hip_ang_s:5.1f}°", (10, 72), scale=1.0)
                self._draw_text(img, f"Ankle:{ank_ang_s:5.1f}°", (10, 108), scale=1.0)

        # --------------- Big, simple HUD ---------------
        # Rep counter
        h, w, _ = img.shape
        rep_text = f"Reps: {self.rep_count}"
        self._draw_text(img, rep_text, (20, 60), scale=1.4, color=(255,255,255), thick=3)

        # Phase + traffic light style message
        if self.state == "top":
            # Encourage to sit down if depth not met
            msg = "Stand tall" if depth_ok else "Sit"
            color = (40, 220, 40) if msg == "Stand tall" else (30, 180, 255)
        else:
            # at bottom: remind to hold then stand
            elapsed = self._now_ms() - self.last_state_change_ms
            need = self.th.hold_ms_bottom
            remain = max(0, need - elapsed)
            if remain > 0:
                msg = f"Hold... {remain//100}"
                color = (0, 180, 255)
            else:
                msg = "Good — up!"
                color = (0, 220, 0)

        self._draw_big_status(img, msg, color)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------- WebRTC Runner ----------------------------

webrtc_ctx = webrtc_streamer(
    key="squat-webrtc",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SquatProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    },
)

# Bridge Streamlit UI <-> Processor
if webrtc_ctx and webrtc_ctx.video_processor:
    vp: SquatProcessor = webrtc_ctx.video_processor
    vp.side = side
    vp.mirror = mirror
    vp.th = th
    vp.grandma_mode = grandma_mode
    vp.show_angles = show_angles
    vp.show_lines = show_lines
    vp.show_skeleton = show_skeleton
    vp.audible_beep = audible_beep
    # update buffer lengths if smoothing changed
    vp.knee_buf = deque(list(vp.knee_buf), maxlen=th.smooth_n)
    vp.hip_buf = deque(list(vp.hip_buf), maxlen=th.smooth_n)
    vp.ankle_buf = deque(list(vp.ankle_buf), maxlen=th.smooth_n)

# --------------------------- Right Panel ------------------------------

with st.expander("How it decides depth & reps (read me)", expanded=False):
    st.markdown(
        """
        **Heuristics** (side view):
        - **Depth**: hip landmark (H) must be visually **below** the knee (K) by a margin you set in the sidebar.
        - **Angles**: we compute
          - **Knee angle** ∠(Hip–Knee–Ankle). When it drops to ≤ *Bottom* threshold → you're **down**.
          - When it rises to ≥ *Top* threshold **after** a valid bottom (and required hold time) → **+1 rep**.
        - **Hold‑to‑count**: you must remain at bottom/top for the set milliseconds to avoid false counts.

        **Grandma Mode**:
        - Bigger text, high‑contrast colors, minimal overlays.
        - Higher smoothing, lower FPS, and required holds for a calm, steady rhythm.
        """
    )

st.info("If camera fails to start: check browser permissions, try Chrome, and ensure no other app is using the webcam.")
