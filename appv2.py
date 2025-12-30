import os
import json
from pathlib import Path
import base64

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

import numpy as np
import cv2
import joblib
from skimage.feature import hog as sk_hog


       
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

# Load and encode the branding GIF
def get_branding_gif_base64():
    gif_path = ROOT / "branding.gif"
    if gif_path.exists():
        with open(gif_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


                                         
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class LettersCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


                                                             
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


                  
@st.cache_resource
def load_digits_cnn():
    p = MODELS_DIR / "digit_model.pth"
    if not p.exists():
        return None
    m = DigitCNN()
    m.load_state_dict(torch.load(str(p), map_location="cpu"))
    m.eval()
    return m


@st.cache_resource
def load_letters_cnn():
    p = MODELS_DIR / "letters_model.pth"
    if not p.exists():
        return None
    m = LettersCNN(num_classes=26)
    state = torch.load(str(p), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        m.load_state_dict(state["model_state_dict"])
    else:
        m.load_state_dict(state)
    m.eval()
    return m


@st.cache_resource
def load_digits_baseline():
    p = MODELS_DIR / "digits_logreg.pkl"
    return joblib.load(p) if p.exists() else None


@st.cache_resource
def load_letters_baseline():
    p = MODELS_DIR / "letters_logreg.pkl"
    return joblib.load(p) if p.exists() else None


@st.cache_resource
def load_digits_advanced():
    p = MODELS_DIR / "digits_advanced.pth"
    if not p.exists():
        return None, None
    ckpt = torch.load(str(p), map_location="cpu")
    num_classes = 10
    m = AdvancedCNN(num_classes=num_classes)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, ckpt


@st.cache_resource
def load_letters_advanced():
    p = MODELS_DIR / "letters_advanced.pth"
    if not p.exists():
        return None, None
    ckpt = torch.load(str(p), map_location="cpu")
    num_classes = 26
    m = AdvancedCNN(num_classes=num_classes)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, ckpt


def load_hog_meta() -> dict | None:
    """
    Optional: if you saved meta during training, use it so HOG params match perfectly.
    We try a few common key formats.
    """
    p = MODELS_DIR / "letters_logreg_meta.json"
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            for key in ("hog", "hog_params", "hog_kwargs"):
                if key in meta and isinstance(meta[key], dict):
                    return meta[key]
            keys = ["orientations", "pixels_per_cell", "cells_per_block", "block_norm", "transform_sqrt"]
            flat = {k: meta[k] for k in keys if k in meta}
            return flat if flat else None
    except Exception:
        return None
    return None


HOG_META = load_hog_meta()


                                 
def hog_features(img28_u8: np.ndarray) -> np.ndarray:
    kwargs = dict(
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    if isinstance(HOG_META, dict):
        for k in ("orientations", "block_norm", "transform_sqrt"):
            if k in HOG_META:
                kwargs[k] = HOG_META[k]
        for k in ("pixels_per_cell", "cells_per_block"):
            if k in HOG_META:
                v = HOG_META[k]
                if isinstance(v, list):
                    v = tuple(v)
                kwargs[k] = v

    feat = sk_hog(img28_u8, **kwargs)
    return feat.astype(np.float32).reshape(1, -1)


def get_feat_dim(clf) -> int:
    try:
        return int(clf.named_steps["logreg"].coef_.shape[1])
    except Exception:
        pass
    try:
        return int(clf.coef_.shape[1])
    except Exception:
        pass
    return int(getattr(clf, "n_features_in_", 0) or 0)


def baseline_predict(clf, img_u8: np.ndarray):
    feat_dim = get_feat_dim(clf)

                                            
    if feat_dim == 784:
        x_raw = (img_u8.astype(np.float32) / 255.0).reshape(1, -1)
        probs = clf.predict_proba(x_raw)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        return pred, conf, probs, feat_dim

                  
    x = hog_features(img_u8)
    probs = clf.predict_proba(x)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred, conf, probs, feat_dim

                                                          

def _get_classes(clf):
    if hasattr(clf, "named_steps") and "logreg" in getattr(clf, "named_steps", {}):
        return getattr(clf.named_steps["logreg"], "classes_", None)
    return getattr(clf, "classes_", None)


def _letter_from_class_value(v):
                                                   
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="ignore")
    if isinstance(v, str) and len(v) == 1 and v.isalpha():
        return v.upper()

    v = int(v)
    if 1 <= v <= 26:
        v = v - 1
    return chr(ord("A") + v)


def _letter_label_fn_for_probs(classes):
    def fn(i: int):
        if classes is None:
            return chr(ord("A") + int(i))
        return _letter_from_class_value(classes[int(i)])
    return fn


def _emnist_letters_fix_candidates(u8: np.ndarray):
    """
    EMNIST letters are often rotated + mirrored relative to normal photos.
    Try a couple of common canonical fixes and let the baseline choose.
    """
    a = np.fliplr(np.rot90(u8, 1)).astype(np.uint8)                                     
    b = np.rot90(np.fliplr(u8), 3).astype(np.uint8)                                          
    return [("rot90+flipLR", a), ("flipLR+rot-90", b)]


                                       
def advanced_predict(model, ckpt: dict, img_u8: np.ndarray):
    img_size = int(ckpt.get("img_size", 96))
    mean = float(ckpt.get("norm_mean", 0.5))
    std = float(ckpt.get("norm_std", 0.5))

    pil = Image.fromarray(img_u8)
    x = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])(pil).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred, conf, probs, img_size


                                                            
                                       
                       
                                         
                                             
                                                
                                  

def _center_by_mass(img28_u8: np.ndarray) -> np.ndarray:
    if img28_u8.sum() == 0:
        return img28_u8
    m = cv2.moments(img28_u8)
    if m["m00"] == 0:
        return img28_u8
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    shiftx = int(round(14 - cx))
    shifty = int(round(14 - cy))
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    shifted = cv2.warpAffine(img28_u8, M, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)
    return shifted.astype(np.uint8)


def preprocess_photo(img: Image.Image, thickening_level: int = 2):
    img = img.convert("L")
    if np.mean(np.array(img)) > 127:
        img = ImageOps.invert(img)
    gray = np.array(img).astype(np.uint8)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if thickening_level >= 2:
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    if thickening_level >= 3:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = mask[y0:y1 + 1, x0:x1 + 1]

    ch, cw = crop.shape
    if ch == 0 or cw == 0:
        return None, None, None

    size = max(ch, cw)
    margin = max(2, int(0.15 * size))
    canvas_size = size + 2 * margin
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    xoff = (canvas_size - cw) // 2
    yoff = (canvas_size - ch) // 2
    canvas[yoff:yoff + ch, xoff:xoff + cw] = crop

    resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    pil28 = Image.fromarray(resized)
    x_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])(pil28).unsqueeze(0)

    return x_tensor, pil28, resized


def rotate_u8(img_u8: np.ndarray, k: int):
    return np.rot90(img_u8, k=k).astype(np.uint8)


def label_letter(i: int) -> str:
    return chr(ord("A") + int(i))


def show_topk(probs, labels_fn, k=5):
    top = np.argsort(probs)[::-1][:k]
    cols = st.columns(k)
    for i, idx in enumerate(top):
        with cols[i]:
            st.markdown(f"**{labels_fn(idx)}**")
            st.write(f"{probs[idx] * 100:.1f}%")


    
st.set_page_config(page_title="ScribML", page_icon="ScribML", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
  --bg: #0d1117;
  --panel: #151b26;
  --panel-2: #1b2230;
  --panel-3: #202838;
  --text: #e6edf3;
  --muted: #9aa4b2;
  --accent: #6ad8ff;
  --border: rgba(255, 255, 255, 0.1);
  --glass-bg: rgba(255, 255, 255, 0.05);
  --glass-strong: rgba(255, 255, 255, 0.08);
}
html, body, [class*="css"] {
  font-family: "Space Grotesk", "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
}
.stApp {
  background:
    radial-gradient(1200px 700px at 10% -10%, rgba(110, 163, 255, 0.28), transparent 60%),
    radial-gradient(900px 700px at 90% 10%, rgba(102, 233, 196, 0.18), transparent 60%),
    radial-gradient(1000px 700px at 50% 110%, rgba(255, 140, 122, 0.16), transparent 60%),
    linear-gradient(180deg, #0a0f18 0%, #0b111c 100%);
  background-size: 140% 140%;
  animation: meshShift 18s ease-in-out infinite;
  color: var(--text);
}
@keyframes meshShift {
  0% { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 0%; }
  50% { background-position: 80% 20%, 20% 60%, 50% 20%, 0% 0%; }
  100% { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 0%; }
}

/* IMPROVED SIDEBAR GLASSMORPHISM */
section[data-testid="stSidebar"] {
  background: transparent;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}
section[data-testid="stSidebar"] > div:first-child {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 20px;
  margin: 16px 12px 16px 16px;
  padding: 20px 16px;
  height: calc(100vh - 32px);
  box-shadow: 
    0 8px 32px rgba(0, 0, 0, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  overflow: auto;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
}

/* Hide sidebar resizer */
div[data-testid="stSidebarResizer"],
div[data-testid="stSidebarResizer"] > div,
div[aria-label="Resize sidebar"],
div[role="separator"][data-testid="stSidebarResizer"] {
  display: none !important;
  width: 0 !important;
  pointer-events: none !important;
}

/* Sidebar text styling - improved readability */
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
  color: var(--text);
  font-weight: 400;
  line-height: 1.6;
}

section[data-testid="stSidebar"] label {
  font-weight: 500;
  font-size: 14px;
  letter-spacing: 0.01em;
}

div[data-testid="stSidebarNav"] { display: none; }
header[data-testid="stHeader"] {
  background: transparent;
  box-shadow: none;
}
div[data-testid="stDecoration"] { display: none; }
div[data-testid="stToolbar"] {
  background: transparent;
  box-shadow: none;
}
.block-container { padding-top: 1.5rem; }
.stApp header { background: transparent; }

/* IMPROVED RADIO BUTTONS - uniform width, better styling */
.stRadio > div[role="radiogroup"] {
  gap: 8px;
}

.stRadio div[role="radiogroup"] > label {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 6px;
  transition: all 0.2s ease;
  width: 100%;
  max-width: 100%;
}

.stRadio div[role="radiogroup"] > label:hover {
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(106, 216, 255, 0.3);
}

.stRadio div[role="radiogroup"] > label > div:first-child {
  border-color: var(--accent);
  background: transparent;
}

/* Radio button selected state - using cyan/blue instead of orange */
.stRadio div[role="radiogroup"] > label[aria-checked="true"] {
  background: rgba(106, 216, 255, 0.08);
  border-color: var(--accent);
}

.stRadio div[role="radiogroup"] > label[aria-checked="true"] > div:first-child {
  background: var(--accent);
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(106, 216, 255, 0.2);
}

.stRadio div[role="radiogroup"] > label[aria-checked="true"] > div:first-child > div {
  background-color: var(--accent) !important;
}

.stRadio div[role="radiogroup"] > label > div:last-child {
  font-weight: 400;
  font-size: 14px;
}

/* Slider accent color */
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] div {
  background: var(--accent) !important;
}
.stSlider [data-baseweb="slider"] > div > div > div > div {
  background: rgba(106, 216, 255, 0.35) !important;
}
.stSlider [data-baseweb="slider"] div[role="presentation"] div {
  background: rgba(106, 216, 255, 0.35) !important;
}
.stSlider [data-baseweb="slider"] div[aria-valuetext] {
  color: var(--accent) !important;
}
.stSlider [data-baseweb="slider"] div[role="progressbar"] {
  background: var(--accent) !important;
}
.stSlider [data-baseweb="slider"] svg path,
.stSlider [data-baseweb="slider"] svg circle {
  fill: var(--accent) !important;
  stroke: var(--accent) !important;
}

/* IMPROVED HEADER with ScribML branding on left */
.hero {
  margin: 0 0 24px 0;
  text-align: left;
}
.hero h1 {
  font-size: 48px;
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: -0.02em;
  margin: 0 0 8px 0;
  background: linear-gradient(135deg, #6ad8ff 0%, #8fb7d9 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero .subtitle {
  font-size: 16px;
  font-weight: 400;
  color: rgba(230, 237, 243, 0.55);
  margin: 0 0 8px 0;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}
.hero p {
  color: var(--muted);
  margin: 0;
  font-size: 15px;
  line-height: 1.6;
  max-width: 600px;
}

/* Bottom right GIF branding */
.afk-branding {
  position: fixed;
  bottom: 12px;
  right: 12px;
  z-index: 999;
}

.afk-branding img {
  width: 120px;
  height: 120px;
  border-radius: 8px;
  opacity: 0.9;
  mix-blend-mode: lighten;
  filter: brightness(1.1) contrast(1.1);
}

.panel {
  background: rgba(17, 25, 40, 0.55);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}
.panel.soft {
  background: rgba(24, 35, 54, 0.5);
}
.panel.notice {
  background: linear-gradient(180deg, rgba(26, 42, 65, 0.75) 0%, rgba(22, 34, 54, 0.75) 100%);
  border-color: rgba(120, 170, 220, 0.35);
  color: #cfe3ff;
  text-align: center;
  max-width: 760px;
  margin: 20px auto;
}

.section-title {
  font-size: 11px;
  letter-spacing: 0.1em;
  color: rgba(154, 164, 178, 0.8);
  text-transform: uppercase;
  margin: 20px 0 12px 0;
  font-weight: 600;
}

.subtle {
  color: var(--muted);
  font-size: 13px;
}

.stSelectbox > label {
  font-weight: 500;
  color: var(--text);
  font-size: 14px;
}

.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div > div {
  border-radius: 10px !important;
}

.stSelectbox [data-baseweb="select"] > div {
  background: rgba(255, 255, 255, 0.06) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}

/* IMPROVED FILE UPLOADER - matched width */
.upload-container {
  max-width: 760px;
  margin: 0 auto;
}

.stFileUploader {
  background: rgba(22, 32, 48, 0.55);
  border: 1px dashed rgba(255, 255, 255, 0.25);
  border-radius: 12px;
  padding: 8px 12px 12px 12px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}

.stFileUploader [data-testid="stFileUploaderDropzone"] {
  background: transparent;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
}

.stFileUploader:hover,
.stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(106, 216, 255, 0.2);
}

.stFileUploader button {
  border-radius: 10px;
  background: rgba(25, 35, 53, 0.7);
  border: 1px solid rgba(106, 216, 255, 0.35);
  color: var(--text);
}

.stFileUploader > label {
  font-weight: 500;
  color: var(--text);
  font-size: 14px;
}

.stCaption { color: var(--muted); }
.divider {
  height: 1px;
  background: var(--border);
  margin: 16px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

branding_gif = get_branding_gif_base64()
branding_img_tag = f'<img src="data:image/gif;base64,{branding_gif}" alt="branding">' if branding_gif else ''

st.markdown(
    f"""
<div class="hero">
  <h1>ScribML</h1>
  <div class="subtitle">Handwritten Character Recognition System</div>
  <p>Upload a photo of your handwritten character for instant classification.</p>
</div>
<div class="afk-branding">
  {branding_img_tag}
</div>
""",
    unsafe_allow_html=True,
)

task = st.sidebar.selectbox("Task", ["Digits (0-9)", "Letters (A-Z)"])
model_type = st.sidebar.radio(
    "Model Type",
    options=["cnn", "advanced", "baseline"],
    format_func={
        "cnn": "Deep Learning (CNN)",
        "advanced": "Advanced (Custom CNN)",
        "baseline": "Traditional ML (Logistic Regression)",
    }.get,
)
thickening = 2

auto_orient = False
if task == "Letters (A-Z)":
    st.sidebar.markdown("---")
    auto_orient = st.sidebar.checkbox("Auto-orient (only if confidence is low)", value=True)

st.markdown('<div class="section-title">Upload</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

digits_cnn = letters_cnn = digits_bl = letters_bl = None
digits_adv = letters_adv = None
digits_adv_ckpt = letters_adv_ckpt = None

if model_type == "cnn":
    digits_cnn = load_digits_cnn() if task.startswith("Digits") else None
    letters_cnn = load_letters_cnn() if task.startswith("Letters") else None
elif model_type == "baseline":
    digits_bl = load_digits_baseline() if task.startswith("Digits") else None
    letters_bl = load_letters_baseline() if task.startswith("Letters") else None
else:
    if task.startswith("Digits"):
        digits_adv, digits_adv_ckpt = load_digits_advanced()
    else:
        letters_adv, letters_adv_ckpt = load_letters_advanced()

if not file:
    st.stop()

original = Image.open(file)
x_tensor, processed_pil, img_u8 = preprocess_photo(original, thickening_level=thickening)

if x_tensor is None:
    st.error("No character detected. Try darker ink, bigger writing, better lighting.")
    st.stop()

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("#### Original")
    st.image(original, use_container_width=True)

with col2:
    st.markdown("#### Processed (Model Input = 28x28)")
    preview = processed_pil.resize((280, 280), resample=Image.Resampling.BICUBIC)
    st.image(preview, width=220)
    st.caption("Preview is upscaled smoothly. Model uses 28x28.")


            
if task == "Digits (0-9)":
    if model_type == "cnn":
        if digits_cnn is None:
            st.error("Missing models/digit_model.pth")
            st.stop()
        with torch.no_grad():
            out = digits_cnn(x_tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        label = str(pred)

    elif model_type == "baseline":
                                       
        if digits_bl is None:
            st.error("Missing models/digits_logreg.pkl")
            st.stop()
        pred, conf, probs, feat_dim = baseline_predict(digits_bl, img_u8)
        label = str(pred)
        st.caption(f"Baseline feature dim: {feat_dim}")

    else:
        if digits_adv is None or digits_adv_ckpt is None:
            st.error("Missing models/digits_advanced.pth (run training_scripts/advanced_train.py --task digits)")
            st.stop()

        pred, conf, probs, img_size = advanced_predict(digits_adv, digits_adv_ckpt, img_u8)
        label = str(pred)
        st.caption(f"Advanced input: resize->{img_size}x{img_size}")

    with col3:
        st.markdown("#### Prediction")
        st.markdown(f"## {label}")
        st.write(f"Confidence: {conf * 100:.1f}%")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Top 5 Predictions")
    show_topk(probs, lambda i: str(i), k=5)

else:
                   
    if model_type == "cnn":
        if letters_cnn is None:
            st.error("Missing models/letters_model.pth")
            st.stop()

        def eval_cnn(u8):
            pil = Image.fromarray(u8)
            x = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])(pil).unsqueeze(0)
            with torch.no_grad():
                out = letters_cnn(x)
                p = torch.softmax(out, dim=1)[0].cpu().numpy()
            return p, float(np.max(p))

        probs, conf = eval_cnn(img_u8)

        pred = int(np.argmax(probs))
        label = label_letter(pred)

        with col3:
            st.markdown("#### Prediction")
            st.markdown(f"## {label}")
            st.write(f"Confidence: {conf * 100:.1f}%")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Top 5 Predictions")
        show_topk(probs, lambda i: label_letter(i), k=5)

    elif model_type == "baseline":
                                                                                             
        if letters_bl is None:
            st.error("Missing models/letters_logreg.pkl")
            st.stop()

        feat_dim = get_feat_dim(letters_bl)
        classes = _get_classes(letters_bl)
        st.caption(f"Baseline feature dim: {feat_dim} | classes: {len(classes) if classes is not None else '?'}")

        best_pred_idx, best_conf, best_probs, best_img = None, -1.0, None, img_u8
        best_desc = "orig"

                                                                                            
        for desc, cand_u8 in [("orig", img_u8)]:
            try:
                p_idx, c, pr, _ = baseline_predict(letters_bl, cand_u8)
            except Exception as e:
                st.error(f"Letters baseline error (feature mismatch / HOG params): {e}")
                st.stop()
            if c > best_conf:
                best_pred_idx, best_conf, best_probs = p_idx, c, pr
                best_img = cand_u8
                best_desc = desc

                                                                                     
        probs = best_probs
        conf = float(best_conf)

        if classes is not None:
            pred_class = classes[int(best_pred_idx)]
            label = _letter_from_class_value(pred_class)
            label_fn = _letter_label_fn_for_probs(classes)
        else:
            label = chr(ord("A") + int(best_pred_idx))
            label_fn = lambda i: chr(ord("A") + int(i))

        with col2:
            st.caption(f"Letters baseline input: {best_desc}")
            st.image(Image.fromarray(best_img).resize((280, 280), Image.Resampling.BICUBIC), width=220)

        with col3:
            st.markdown("#### Prediction")
            st.markdown(f"## {label}")
            st.write(f"Confidence: {conf * 100:.1f}%")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Top 5 Predictions")
        show_topk(probs, label_fn, k=5)

        st.stop()

    else:
        if letters_adv is None or letters_adv_ckpt is None:
            st.error("Missing models/letters_advanced.pth (run training_scripts/advanced_train.py --task letters)")
            st.stop()

        pred, conf, probs, img_size = advanced_predict(letters_adv, letters_adv_ckpt, img_u8)
        label = label_letter(pred)
        st.caption(f"Advanced input: resize->{img_size}x{img_size}")

        with col3:
            st.markdown("#### Prediction")
            st.markdown(f"## {label}")
            st.write(f"Confidence: {conf * 100:.1f}%")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Top 5 Predictions")
        show_topk(probs, lambda i: label_letter(i), k=5)
