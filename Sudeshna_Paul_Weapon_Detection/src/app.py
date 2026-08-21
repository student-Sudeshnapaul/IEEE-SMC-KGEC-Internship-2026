"""
Live Weapon Detection System
-----------------------------
Streamlit app that runs a YOLOv8 model on a live webcam feed (via browser),
uploaded images, or uploaded videos, drawing bounding boxes on detections
and logging them.
"""

import time
import queue
import tempfile
import os
from datetime import datetime

import av
import cv2
import imageio
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_PATH = "best.pt"
CLASS_NAMES = {0: "Weapon", 1: "Weapon"}

# Set your detection confidence threshold here (0.1 - 0.9).
# Lower = more sensitive (more detections, more false positives).
# Higher = stricter (fewer detections, may miss real weapons).
CONFIDENCE_THRESHOLD = 0.5

st.set_page_config(page_title="Weapon Detection System", layout="wide")

# --------------------------------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------------------------------
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []  # list of dicts: {time, class, confidence}
if "detection_queue" not in st.session_state:
    st.session_state.detection_queue = queue.Queue()

# --------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.caption(f"Detection confidence threshold: **{CONFIDENCE_THRESHOLD}** (set in code)")

if st.sidebar.button("Clear alert log"):
    st.session_state.alert_log = []

# --------------------------------------------------------------------------
# LOAD MODEL (cached so it only loads once)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    m = YOLO(MODEL_PATH)
    # Override the model's own internal class names so the labels drawn
    # directly on the image (via r.plot()) match CLASS_NAMES above, not
    # just the text summary below the image. `m.names` is a read-only
    # property on the YOLO wrapper, so we set it on the underlying model.
    m.model.names = CLASS_NAMES
    return m

model = load_model()


def log_detection(detected_this_frame):
    """Push the top detection from this frame/image into the alert log."""
    if not detected_this_frame:
        return
    top_class, top_conf = max(detected_this_frame, key=lambda x: x[1])
    st.session_state.alert_log.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "class": top_class,
            "confidence": top_conf,
        },
    )


def run_detection_on_image(image_bgr):
    """Runs the model on a single BGR image (numpy array) and returns
    (annotated_bgr_image, list_of_(class_name, confidence))."""
    results = model.predict(image_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append((CLASS_NAMES.get(cls_id, str(cls_id)), conf))

    annotated = r.plot()
    return annotated, detections


# --------------------------------------------------------------------------
# VIDEO PROCESSOR (runs per-frame, server side)
# --------------------------------------------------------------------------
class WeaponDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.detection_queue = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # un-mirror the webcam feed (natural, non-selfie view)

        results = model.predict(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        r = results[0]

        detected_this_frame = []
        if len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = CLASS_NAMES.get(cls_id, str(cls_id))
                detected_this_frame.append((class_name, conf))

        annotated = r.plot()

        if detected_this_frame and self.detection_queue is not None:
            top_class, top_conf = max(detected_this_frame, key=lambda x: x[1])
            try:
                self.detection_queue.put_nowait(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "class": top_class,
                        "confidence": top_conf,
                    }
                )
            except queue.Full:
                pass

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# --------------------------------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------------------------------
st.title("🔫 Live Weapon Detection System")
st.caption("Real-time knife & handgun detection powered by YOLOv8")

tab_live, tab_image, tab_video = st.tabs(["📷 Live Webcam", "🖼️ Upload Image", "🎞️ Upload Video"])

# ---- TAB 1: Live webcam ----------------------------------------------------
with tab_live:
    col_video, col_log = st.columns([2, 1])

    with col_video:
        st.subheader("Live Feed")

        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        ctx = webrtc_streamer(
            key="weapon-detection",
            video_processor_factory=WeaponDetectionProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.detection_queue = st.session_state.detection_queue

        st.caption("Note: camera access requires the page be loaded via `localhost`, not a network IP.")

    with col_log:
        st.subheader("Alert Log")
        log_placeholder = st.empty()

        # drain any new detections from the queue into session state
        while not st.session_state.detection_queue.empty():
            try:
                st.session_state.alert_log.insert(0, st.session_state.detection_queue.get_nowait())
            except queue.Empty:
                break

        if st.session_state.alert_log:
            log_placeholder.dataframe(
                st.session_state.alert_log,
                use_container_width=True,
                hide_index=True,
            )
        else:
            log_placeholder.info("No detections yet.")

# ---- TAB 2: Upload a single image -----------------------------------------
with tab_image:
    st.subheader("Detect weapons in an image")

    uploaded_image = st.file_uploader(
        "Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], key="image_uploader"
    )

    if uploaded_image is not None:
        try:
            pil_image = Image.open(uploaded_image).convert("RGB")
        except Exception:
            st.error(
                "Couldn't read this file as an image. It may be corrupted, "
                "an unsupported format despite its extension, or an incomplete "
                "download. Try re-downloading it or picking a different file."
            )
            st.stop()

        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        with st.spinner("Running detection..."):
            annotated_bgr, detections = run_detection_on_image(image_bgr)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col_before, col_after = st.columns(2)
        with col_before:
            st.image(pil_image, caption="Original", use_column_width=True)
        with col_after:
            st.image(annotated_rgb, caption="Detected", use_column_width=True)

        if detections:
            st.success(f"Found {len(detections)} weapon(s): " +
                       ", ".join(f"{c} ({conf:.0%})" for c, conf in detections))
            log_detection(detections)
        else:
            st.info("No weapons detected.")

# ---- TAB 3: Upload a video, process frame by frame -------------------------
with tab_video:
    st.subheader("Detect weapons in a video")
    st.caption("The video is processed frame by frame on the server, then an annotated version is shown below. Longer videos take longer to process.")

    uploaded_video = st.file_uploader(
        "Upload a video (mp4, mov, avi)", type=["mp4", "mov", "avi"], key="video_uploader"
    )

    frame_skip = st.slider(
        "Process every Nth frame (higher = faster, less thorough)",
        min_value=1, max_value=10, value=2, key="frame_skip"
    )

    if uploaded_video is not None and st.button("Run detection on video"):
        # save upload to a temp file so cv2.VideoCapture can read it
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
        in_tmp.write(uploaded_video.read())
        in_tmp.close()

        cap = cv2.VideoCapture(in_tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = in_tmp.name + "_annotated.mp4"
        # Use imageio's bundled FFmpeg with H.264 so the output plays directly
        # in-browser. cv2.VideoWriter's mp4v codec often produces files that
        # Chrome/Firefox/Edge refuse to play ("No video with supported format").
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)

        progress_bar = st.progress(0, text="Processing video...")
        video_detections_found = False
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                annotated, detections = run_detection_on_image(frame)
                if detections:
                    video_detections_found = True
                    log_detection(detections)
                writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            else:
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # skipped frame, write as-is to keep video length/timing

            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0), text=f"Processing frame {frame_idx}/{total_frames}")

        cap.release()
        writer.close()
        progress_bar.empty()

        if video_detections_found:
            st.success("Weapons detected in this video — see Alert Log in the Live Webcam tab and annotated video below.")
        else:
            st.info("No weapons detected in this video.")

        st.video(out_path)

        with open(out_path, "rb") as f:
            st.download_button("Download annotated video", f, file_name="annotated_video.mp4")

        # cleanup temp input file (leave output so st.video/download_button can still serve it this run)
        os.unlink(in_tmp.name)