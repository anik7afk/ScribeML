import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps 
import os
import numpy as np
import cv2

# Model for 62 classes
class DigitLetterCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(64*5*5, 256),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x): 
        return self.fc(self.conv(x))

# EMNIST index → character
def idx_to_char(idx):
    """Map EMNIST index to character."""
    if idx < 10:
        return str(idx)  # 0-9
    elif idx < 36:
        return chr(ord('A') + idx - 10)  # A-Z
    else:
        return chr(ord('a') + idx - 36)  # a-z

# Load model once
@st.cache_resource
def load_model():
    model_path = "models/digit_letter_model.pth"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at {model_path}")
        st.info("Please train the model first by running: `python train_emnist.py`")
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    num_classes = checkpoint.get('num_classes', 62)
    
    model = DigitLetterCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, num_classes

# Preprocessing for real photos
def advanced_preprocess(img, thickening_level=2):
    """Preprocess smartphone character photo."""
    img = img.convert('L')
    img_array = np.array(img)
    
    # Denoise
    img_array = cv2.fastNlMeansDenoising(img_array, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_array = clahe.apply(img_array)
    
    # Adaptive threshold
    img_array = cv2.adaptiveThreshold(
        img_array, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11, 
        C=2
    )
    
    # Auto-invert
    white_pixel_ratio = np.sum(img_array == 255) / img_array.size
    if white_pixel_ratio > 0.5:
        img_array = cv2.bitwise_not(img_array)
    
    # Remove small noise
    kernel_small = np.ones((2, 2), np.uint8)
    img_array = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel_small)
    
    # Thicken strokes
    if thickening_level == 1:
        kernel_dilate = np.ones((2, 2), np.uint8)
        iterations = 1
    elif thickening_level == 2:
        kernel_dilate = np.ones((3, 3), np.uint8)
        iterations = 2
    else:
        kernel_dilate = np.ones((4, 4), np.uint8)
        iterations = 3
    
    img_array = cv2.dilate(img_array, kernel_dilate, iterations=iterations)
    
    # Find and crop character
    contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    margin = max(10, int(min(w, h) * 0.1))
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_array.shape[1] - x, w + 2 * margin)
    h = min(img_array.shape[0] - y, h + 2 * margin)
    
    img_array = img_array[y:y+h, x:x+w]
    img = Image.fromarray(img_array)
    
    # Make square
    width, height = img.size
    if width > height:
        new_height = width
        new_img = Image.new('L', (width, new_height), 0)
        new_img.paste(img, (0, (new_height - height) // 2))
    else:
        new_width = height
        new_img = Image.new('L', (new_width, height), 0)
        new_img.paste(img, ((new_width - width) // 2, 0))
    
    img = new_img
    
    # Add padding
    width, height = img.size
    new_size = int(width * 1.4)
    final_img = Image.new('L', (new_size, new_size), 0)
    paste_pos = (new_size - width) // 2
    final_img.paste(img, (paste_pos, paste_pos))
    
    # Resize to 28x28
    img = final_img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Normalize
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])(img).unsqueeze(0)
    
    return img_tensor, img

# UI
st.set_page_config(page_title="Handwritten Character AI", page_icon="✍️", layout="wide")
st.title("✍️ Handwritten Character Recognition")
st.write("Upload a photo of a handwritten **digit (0-9)** or **letter (A-Z, a-z)**")

# Sidebar
st.sidebar.header("⚙️ Settings")
thickening = st.sidebar.select_slider(
    "Stroke Thickening",
    options=[1, 2, 3],
    value=2,
    help="Increase if your handwriting is very thin"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**📸 Photo Tips:**
- ✓ **Bright lighting** (no shadows)
- ✓ **Dark ink** on **white paper**
- ✓ **Large character** (fill the paper)
- ✓ **Centered** in photo
- ✓ **Hold camera steady**
- ✓ **Plain white paper** (no lines)

**Supported Characters:**
- Digits: 0-9
- Uppercase: A-Z
- Lowercase: a-z
""")

# Image upload
file = st.file_uploader("📤 Upload character image", type=["jpg", "png", "jpeg"])

# Load cached model
model, num_classes = load_model()

# Prediction
if file and model is not None:
    try:
        # Original image
        original_img = Image.open(file)
        
        # Layout columns
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("#### 📸 Original Photo")
            st.image(original_img, use_container_width=True)
        
        # Preprocess
        img_tensor, processed_img = advanced_preprocess(original_img, thickening_level=thickening)
        
        if img_tensor is None:
            st.error("❌ No character detected in the image!")
            st.info("**Tips:** Ensure good lighting, dark ink on white paper, and the character is clearly visible")
        else:
            # Predict
            with torch.no_grad():
                out = model(img_tensor)
                probabilities = torch.softmax(out, dim=1)[0]
                pred_idx = torch.argmax(out, 1).item()
                confidence = probabilities[pred_idx].item()
                pred_char = idx_to_char(pred_idx)
            
            with col2:
                st.markdown("#### 🔄 Processed (28x28)")
                st.image(processed_img, width=200)
            
            with col3:
                st.markdown("#### 🎯 Prediction")
                
                # Confidence color
                if confidence > 0.9:
                    color = "#4CAF50"
                    status = "✅ High confidence"
                elif confidence > 0.7:
                    color = "#FF9800"
                    status = "⚠️ Medium confidence"
                else:
                    color = "#F44336"
                    status = "❌ Low confidence"
                
                st.markdown(
                    f"<div style='text-align: center; padding: 20px; background-color: #f0f0f0; border-radius: 10px;'>"
                    f"<h1 style='color: {color}; font-size: 64px; margin: 0;'>{pred_char}</h1>"
                    f"<p style='color: #666; font-size: 16px;'>{confidence*100:.1f}% confidence</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**{status}**")
            
            # Top 10 predictions
            st.markdown("---")
            st.markdown("### 📊 Top 10 Predictions")
            
            top10_indices = torch.argsort(probabilities, descending=True)[:10]
            
            cols = st.columns(10)
            for i, idx in enumerate(top10_indices):
                with cols[i]:
                    prob = probabilities[idx].item() * 100
                    char = idx_to_char(idx.item())
                    
                    if i == 0:
                        border_color = "#4CAF50"
                    elif prob > 5:
                        border_color = "#FF9800"
                    else:
                        border_color = "#9E9E9E"
                    
                    st.markdown(
                        f"<div style='text-align: center; padding: 10px; border: 2px solid {border_color}; border-radius: 5px;'>"
                        f"<h2 style='margin: 0;'>{char}</h2>"
                        f"<p style='margin: 0; font-size: 12px; color: #666;'>{prob:.1f}%</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            
            # Tips for low confidence
            if confidence < 0.7:
                st.markdown("---")
                st.warning("""
                **🔧 Low confidence detected. Try these fixes:**
                
                1. **Retake photo** with better lighting
                2. **Increase stroke thickening** (try level 3)
                3. **Use darker ink** (black marker)
                4. **Write larger and clearer**
                5. **Distinguish similar characters** (e.g., O vs 0, I vs l vs 1)
                """)
            
            # Clear button
            st.markdown("---")
            if st.button("🔄 Clear and Upload New Image"):
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

elif file and model is None:
    st.warning("⚠️ Please train the model first by running: `python train_emnist.py`")

else:
    st.info("👆 Upload a photo above to get started!")
    
    st.markdown("---")
    st.markdown("### 🎓 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Advanced Preprocessing:**
        1. Denoise (remove camera grain)
        2. Enhance contrast (handle shadows)
        3. Adaptive thresholding
        4. Auto-invert colors
        5. Clean noise artifacts
        6. Thicken strokes (adjustable)
        7. Crop to character
        8. Resize to 28×28 pixels
        """)
    
    with col2:
        st.markdown("""
        **Model Info:**
        - Custom CNN architecture
        - Trained on EMNIST dataset
        - **62 classes:** 0-9, A-Z, a-z
        - With heavy data augmentation
        - ~90-95% validation accuracy
        
        **Training:**
        ```bash
        python train_emnist.py
        ```
        """)