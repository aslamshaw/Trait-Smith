# Trait-Smith 🧬

A FastAPI-based personality prediction web service powered by a machine learning pipeline built from scratch. Trait-Smith processes behavioral inputs and predicts a user's personality type using a stacked ensemble of optimized classifiers.

---

## 🚀 Features

- **Full ML Pipeline**:
  - Custom imputers and encoders (`SimpleImputer`, `LeaveOneOut`, `OneHot`)
  - Feature engineering with custom ratios and binning
  - Mutual Information-based feature selection
  - Automated scaling & encoding
- **Model Ensemble**:
  - Random Forest, Gradient Boosting, XGBoost, SVM
  - Stacked with Logistic Regression as meta-learner
- **FastAPI Backend**:
  - `/predict` endpoint for live inference
  - `/health` endpoint for service status
- **Serialized Stack**:
  - Trained model + label encoder saved via Pickle

---

# 📁 Project Structure

Use this format inside the markdown file (README.md), not inside a comment block.
But if you insist on keeping it here, use indentation:

    Trait-Smith/
    ├── ml_pipeline.py            # Full data preprocessing pipeline
    ├── train_and_save_model.py   # Training logic and model serialization
    ├── app.py                    # FastAPI app with /predict route
    ├── stacking_model.pkl        # Trained ensemble model (after training)
    ├── label_encoder.pkl         # Label encoder for target class decoding
    └── personality_dataset.csv   # Raw dataset

---

## 🧪 Getting Started

### 1. Clone the repo

    git clone https://github.com/yourusername/Trait-Smith.git
    cd Trait-Smith

### 2. Install requirements

    pip install -r requirements.txt

### 3. Train the model

Ensure `personality_dataset.csv` is available at the path hardcoded in `train_and_save_model.py`:

    os.path.join(os.path.dirname(__file__), 'personality_dataset.csv')

Adjust the path if needed. Then run:

    python train_and_save_model.py

This will generate `stacking_model.pkl` and `label_encoder.pkl`.

### 4. Run the API

    uvicorn app:app --reload

---

## 🎯 Usage

### POST /predict

Send JSON like:

    {
      "Time_spent_Alone": 4.5,
      "Stage_fear": "Yes",
      "Social_event_attendance": 3.0,
      "Going_outside": 2.0,
      "Drained_after_socializing": "No",
      "Friends_circle_size": 5.0,
      "Post_frequency": 1.0
    }

Response:

    {
      "prediction": "Extrovert"
    }

### GET /health

    { "status": "ok" }

---

## 📌 Notes

- Leave-One-Out Encoding applied only to high-cardinality categorical features
- Mutual Information feature selector uses threshold tuning during CV
- Input format is strict — any schema drift will cause prediction to fail

---

## 🧠 Future Enhancements

- Add input schema validation
- Extend Swagger UI docs
- Add model version tracking and experiment logging
- Dockerize + CI/CD for production deployment

---

## 🛡️ License

MIT License — free to use, modify responsibly.
