# Image Classification — Prototype

Clean, images-only interface for simple experiments:
- Load ZIP structured as `<ROOT>/<class>/*`
- Choose features: Raw pixels **or** Compact texture features (DCT)
- Train: SVM / RandomForest / Logistic Regression (and optional Small CNN)
- Evaluate & Predict (confusion matrix, single-image prediction)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional CNN support:
# pip install tensorflow-cpu
