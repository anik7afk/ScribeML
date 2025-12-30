ScribeML - Handwritten Digits & Letters Recognition (CNN + Baseline + Advanced (Custom CNN))

What it does
Streamlit app for digits (0-9) and letters (A-Z).

Models
1. CNN (Deep Learning)
2. Baseline (Traditional ML)
3. Advanced (Custom CNN)

Install
pip install -r requirements.txt

Train
python training_scripts/train_all.py

Run
streamlit run appv2.py

App test (local)
1. Make sure models exist (train first if needed).
2. Run:
   streamlit run appv2.py
3. Open the local URL shown in the terminal (usually http://localhost:8501).

Model files (models/)
- digit_model.pth
- digits_logreg.pkl
- letters_model.pth
- letters_logreg.pkl
- digits_advanced.pth
- letters_advanced.pth

Usage
1. Open the app
2. Pick Task and Model Type
3. Upload an image (.png, .jpg, .jpeg)
4. View prediction + top-5

Notes
- If a model file is missing, the app shows an error.
- On Windows, use num_workers=0 if training hangs.
