## ScribeML – Handwritten Digit & Character Recognition

ScribeML is a small suite of Streamlit apps and training scripts for recognizing handwritten content from photos.  
It comes in **two flavors**:

- **ScribeML (Digits-only)**: recognizes handwritten digits **0–9** using MNIST.
- **ScribeML (Digits + Letters)**: recognizes **digits (0–9)** and **letters (A–Z, a–z)** using EMNIST.

Both versions share the same advanced preprocessing pipeline tuned for smartphone photos (denoising, contrast enhancement, adaptive thresholding, stroke thickening, cropping, padding, and normalization).

---

### Project Structure

- **`app.py`** – **ScribeML (Digits-only)**  
  Streamlit app for **digits 0–9** using `models/digit_model.pth`.

- **`train.py`** – training for **ScribeML (Digits-only)**  
  Trains a CNN on **MNIST** with heavy data augmentation; saves `models/digit_model.pth`.

- **`appv2.py`** – **ScribeML (Digits + Letters)**  
  Streamlit app for **digits + letters (0–9, A–Z, a–z)** using `models/digit_letter_model.pth`.

- **`trainv2.py`** – training for **ScribeML (Digits + Letters)**  
  Trains a CNN on **EMNIST (byclass, 62 classes)**; saves `models/digit_letter_model.pth`.

- **`models/`** – trained model weights (`digit_model.pth`, `digit_letter_model.pth`).
- **`data/`** – MNIST/EMNIST data folders (created and/or populated by torchvision).
- **`requirements.txt`** – Python dependencies.

---

### Installation

1. **Create and activate a virtual environment (recommended)**:

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

### Training the Models

- **ScribeML (Digits-only) – MNIST, 0–9**:

```bash
python train.py
```

This trains a CNN on MNIST with heavy data augmentation and saves the weights to `models/digit_model.pth`.

- **ScribeML (Digits + Letters) – EMNIST byclass, 0–9, A–Z, a–z**:

```bash
python trainv2.py
```

This trains a CNN on the EMNIST `byclass` split (62 classes) and saves a checkpoint (including metadata) to `models/digit_letter_model.pth`.

> **Note**: The `data/EMNIST/raw` and `data/MNIST/raw` folders already exist in this project. Torchvision will reuse them if paths/splits match; otherwise, it may download again into `./data`.

---

### Running the Apps

Make sure the corresponding model file exists in `models/` before launching each app.

- **ScribeML (Digits-only) app – 0–9**:

```bash
streamlit run app.py
```

Features:
- Upload a photo of a handwritten digit.
- Advanced preprocessing and confidence scoring.
- Top-5 prediction display.

- **ScribeML (Digits + Letters) app – 0–9, A–Z, a–z**:

```bash
streamlit run appv2.py
```

Features:
- Upload a photo of a handwritten **digit or letter**.
- Same advanced preprocessing pipeline.
- Displays the predicted character and confidence.
- Top-10 predictions (useful with 62 classes).

---

### Usage Tips

- **Input images**:
  - Use **dark ink** on **white paper**.
  - Write large and centered on the page.
  - Avoid strong shadows and reflections.
  - Hold the camera steady when taking photos.

- **Stroke Thickening slider**:
  - Increase for very thin handwriting.
  - Decrease if digits/letters look too “blobby” after processing.

---

### Which Version Should I Use?

- **Use ScribeML (Digits-only)** – `app.py` + `train.py`  
  If you only care about **digits (0–9)** and want maximum accuracy on numbers with a simpler model.

- **Use ScribeML (Digits + Letters)** – `appv2.py` + `trainv2.py`  
  If you need **both digits and letters (0–9, A–Z, a–z)** in a single unified model.

Both ScribeML variants can coexist in the same environment; just run the appropriate training script and Streamlit app for your use case.


