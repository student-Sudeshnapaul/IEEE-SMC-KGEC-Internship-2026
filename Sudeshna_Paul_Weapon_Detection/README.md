# 🔫 Live Weapon Detection System

**Real-time knife & handgun detection, right in your browser.**
Powered by a custom-trained YOLOv8 model, wrapped in a Streamlit app that watches your webcam, your photos, or your videos — and flags what it sees.

<p align="center">
  <img src="sample1.png" alt="Weapon detection sample output" width="800"/>
</p>
<p align="center">
  <img src="sample2.png" alt="Weapon detection sample output" width="800"/>
</p>
<p align="center"><i>Detected weapons, boxed and scored, in real time.</i></p>

---

## ✨ What it does

This isn't just a model in a notebook — it's a full pipeline from raw footage to flagged threat, with three ways to feed it:

| Mode | What happens |
|---|---|
| 📷 **Live Webcam** | Streams your camera feed to the model over WebRTC and draws detections live, frame by frame |
| 🖼️ **Upload Image** | Drop in a photo, get back the original and the annotated version side by side |
| 🎞️ **Upload Video** | Processes a video frame by frame and hands you back a fully annotated, downloadable copy |

Every detection — from any mode — lands in a shared alert log, so you always have a record of what the system caught.

---

## 🧠 The model

- **Architecture:** YOLOv8n (nano) — small enough for real-time inference, still sharp enough to catch what matters
- **Classes:** trained to detect knives and handguns, displayed under a unified **"Weapon"** label
- **Training:** fine-tuned on a large annotated dataset, trained on free-tier Google Colab GPU compute
- **Speed:** sub-millisecond inference per frame on GPU — the bottleneck in this app is your camera, not the model

---

## 🚀 Getting started

### 1. Clone and set up

```bash
git clone <your-repo-url>
cd weapon-detection-app
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Add your model weights

Make sure `best.pt` (your trained YOLOv8 weights) sits in the same folder as `app.py`.

### 3. Run it

```bash
streamlit run app.py
```

Open the app at **`http://localhost:8501`** — note: it must be `localhost`, not a network IP, or your browser will block camera access.

---

## 📦 What's in this repo

```
.
├── app.py              # the Streamlit app — all three detection modes live here
├── best.pt              # your trained YOLOv8 weights
├── requirements.txt      # Python dependencies
├── packages.txt          # system-level dependencies (for Hugging Face Spaces)
└── assets/
    └── sample.png        # example detection output, shown above
```

---

## ☁️ Deploying for free

### Option A — Hugging Face Spaces *(recommended)*

Handles system-level video/OpenCV dependencies most reliably.

1. Create a free account at [huggingface.co](https://huggingface.co)
2. **New Space** → SDK: **Streamlit** → name it → **Create Space**
3. Upload `app.py`, `best.pt`, `requirements.txt`, and `packages.txt`
4. Wait for the build — your app goes live at `https://huggingface.co/spaces/<you>/<space-name>`

CPU-only on the free tier means somewhat higher latency than local GPU testing, but it's more than enough for image and video detection, and workable for live webcam use.

### Option B — Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. **New app** → select repo/branch → entry point: `app.py` → **Deploy**

If you hit `libGL` or video codec errors here, switch to Option A — Hugging Face Spaces handles these dependencies more consistently.

---

## 🎛️ Tuning

- **Confidence threshold** is fixed in code (`CONFIDENCE_THRESHOLD` near the top of `app.py`) rather than exposed as a live control — edit it directly and restart the app if detections feel too sensitive or too strict.
- **Frame sampling** on video uploads is adjustable per run — process every frame for thoroughness, or skip frames for speed on longer clips.

---

## ⚠️ A note on responsible use

This system is built for **detection assistance**, not autonomous decision-making. It will miss things (rifles and unusual weapon types are a known gap — see the project report for details), and it will occasionally flag things that aren't weapons. Always pair automated detection with human review before acting on an alert.

---

<p align="center">Built with YOLOv8, Streamlit, and a lot of debugging. 🛠️</p>