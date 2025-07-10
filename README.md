# 🛡️ DDoS Detection Model – CIC-DDoS2019 | XGBoost + ONNX

This repository provides a powerful, production-ready model for detecting HTTP-level DDoS attacks using the CIC-DDoS2019 dataset. The model is trained using an XGBoost classifier and exported to both `.pkl` and `.onnx` formats for deployment flexibility.

---

## 🚀 Highlights

- ✅ Robust classification using **XGBoost**
- ✅ Imbalanced data handling via **SMOTE**
- ✅ Hyperparameter tuning with **Optuna**
- ✅ Model explainability with **SHAP**
- ✅ Lightweight & fast **ONNX export**
- ✅ Ready for Hugging Face & GitHub deployment

---

## 📂 Files

| File                     | Description                                      |
|--------------------------|--------------------------------------------------|
| `ddos_detection_pipeline.pkl` | Full pipeline with `StandardScaler` + XGBoost model |
| `ddos_model.onnx`        | Optimized ONNX format of the XGBoost model       |
| `ddos_model.zip`         | Compressed archive of model files (optional)     |
| `README.md`              | This documentation                              |
| `train_pipeline.ipynb`   | (Optional) Full training and tuning notebook     |

---

## 📊 Model Overview

- **Model Type**: Binary Classifier (`0 = Normal`, `1 = Attack`)
- **Algorithm**: XGBoost (Gradient Boosted Trees)
- **Dataset**: [CIC-DDoS2019](https://www.kaggle.com/datasets/dhoogla/cicddos2019)
- **Feature Engineering**:
  - `requests_per_sec = packet_count / flow_duration`
  - `pkt_len_variation = fwd_pkt_len_max - fwd_pkt_len_min`
- **Handling Class Imbalance**: SMOTE oversampling
- **Evaluation Metric**: `AUC-PR`, `F1-score`

---

## 📈 Model Performance

```text
F1-Score (Class 0): 1.00
F1-Score (Class 1): 1.00
AUC-PR:             1.00
Accuracy:           100%
```

✅ The model performs near-perfect on test data with high generalization due to Stratified K-Fold CV and optimized hyperparameters.
```python
import joblib
pipeline = joblib.load("ddos_detection_pipeline.pkl")
prediction = pipeline.predict(sample_data)
```
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("ddos_model.onnx")
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: sample_data_np.astype(np.float32)})
```

🧪 Model Explainability

SHAP plots were used to visualize the top contributing features:

    pkt_len_variation

    requests_per_sec

    flow_duration

    fwd_pkt_len_std

These help understand why a flow was classified as an attack.

![Screenshot 2025-07-10 111229](https://github.com/user-attachments/assets/d87b0ca3-a5de-4136-b2b3-09fc3045c6dc)
![Screenshot 2025-07-10 111247](https://github.com/user-attachments/assets/e92a731f-d9c2-487d-8bbf-955cf7bfe7ec)
![Screenshot 2025-07-10 111255](https://github.com/user-attachments/assets/fd3b4a66-59e8-446b-9653-ad2b52de74b0)


📜 License

This project is licensed under the MIT License.

🤖 Built With

    xgboost

    optuna

    shap

    imbalanced-learn

    scikit-learn

    onnx / onnxruntime

📦 Deployment Targets

    ✅ Hugging Face Hub

    ✅ GitHub

    🛠️ Optional: REST API, Edge deployment with ONNX, Docker, etc.

🙋 Author

Name: Sunny / 007
Role: Cybersecurity Expert | ML Engineer | Red Team Ops

LinkedIn: Sunnythakur
GitHub: [github.com/SunnyThakur25]
⭐ Give it a Star!
If this repo helped you detect DDoS attacks or build ML security pipelines, feel free to ⭐ star this repo.




