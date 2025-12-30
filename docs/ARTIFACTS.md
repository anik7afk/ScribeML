ScribeML artifacts map

This file maps scripts to the files they generate.

Training (models/)
- training_scripts/train.py: digits CNN + baseline (MNIST)
  - models/digit_model.pth
  - models/digits_logreg.pkl
  - models/logreg_model.pkl (compat copy)
- training_scripts/trainv2.py: letters CNN (EMNIST letters)
  - models/letters_model.pth
- training_scripts/train_letters_baseline.py: letters baseline (HOG + LR)
  - models/letters_logreg.pkl
  - models/letters_logreg_meta.json
- training_scripts/advanced_train.py: advanced CNN (digits or letters)
  - models/digits_advanced.pth
  - models/letters_advanced.pth
- training_scripts/train_all.py: runs all the training steps above

Evaluation / figures (figures/)
- training_scripts/make_figures.py: evaluates models and writes reports / confusion matrices
  - figures/digits_cnn_report.txt
  - figures/letters_cnn_report.txt
  - figures/cm_digits_cnn.png
  - figures/cm_letters_cnn.png
  - figures/digits_baseline_report.txt
  - figures/letters_baseline_report.txt
  - figures/cm_digits_baseline.png
  - figures/cm_letters_baseline.png
  - figures/preprocess_pipeline.png (only with --sample_image)

App (runtime)
- appv2.py: Streamlit UI (reads models from models/)

Archive
- archive/figures_baseline/: older baseline reports moved out of figures/
