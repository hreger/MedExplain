
# 💡 MedExplain 🧠💉

MedExplain is an AI-driven medical diagnosis support system focused on transparent diabetes risk assessment using explainable AI (XAI). The system integrates a powerful predictive model with interpretable outputs, enabling medical professionals and patients to understand the _why_ behind the prediction.

---

## 🚀 Features

### ✅ Core Capabilities
- 🔍 **Diabetes Risk Prediction**  
  Uses a trained **XGBoost** classifier on the **PIMA Indians Diabetes Dataset** for accurate risk assessment.
  
- 🧠 **Explainability with XAI**
  - **LIME** explanations showing how each feature contributed to a prediction
  - Ranked feature importance for interpretability
  - Per-input contribution breakdown

- 📊 **Interactive Gradio Interface**
  - Real-time predictions with easy-to-use sliders
  - Live visualizations of results and contributing factors
  - Intuitive and user-friendly design

---

## 🛠️ Tech Stack

| Category         | Tools & Libraries                          |
|------------------|--------------------------------------------|
| Language         | Python 3.11                                |
| Machine Learning | XGBoost, scikit-learn                      |
| Explainability   | LIME, SHAP *(coming soon)*                 |
| Interface        | Gradio                                     |
| Data Handling    | pandas, numpy                              |
| Visualization    | matplotlib                                 |

---

## 🧪 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/MedExplain.git
cd MedExplain
```

### 2️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the Application
```bash
python gradio_ui.py
```

---

## 🧾 Input Parameters

| Parameter                  | Description                                 | Valid Range          |
|---------------------------|---------------------------------------------|----------------------|
| `Pregnancies`             | Number of times pregnant                    | 0–17                 |
| `Glucose`                 | Plasma glucose concentration (mg/dl)        | 0–200                |
| `BloodPressure`           | Diastolic blood pressure (mm Hg)            | 0–122                |
| `SkinThickness`           | Triceps skin fold thickness (mm)            | 0–99                 |
| `Insulin`                 | 2-Hour serum insulin (mu U/ml)              | 0–846                |
| `BMI`                     | Body mass index (kg/m²)                     | 0–67.1               |
| `DiabetesPedigreeFunction`| Hereditary diabetes function score          | 0.078–2.42           |
| `Age`                     | Patient's age (years)                       | 21–81                |

---

## 📈 Output

- **Prediction:** Diabetic / Non-diabetic
- **Probability Score:** Model’s confidence level
- **Top Contributing Features:** Ranked by influence
- **LIME Analysis:**
  - Individual feature contributions
  - Weight of influence per feature

---

## 📷 Screenshots

### 🔹 Prediction Result Display
<img src="screenshots\MedExplain_ScreenShot.png" width="600"/>



## 🗂️ Project Structure

```
MedExplain/
├── data/                     # Dataset files
├── models/                   # Trained ML models
│   ├── diabetes_model.joblib
│   └── scaler.joblib
├── src/                      # Source code
│   ├── predict.py            # Prediction logic
│   └── utils/                # Helper utilities
├── gradio_ui.py              # Main Gradio interface
└── requirements.txt          # Dependencies list
```

---

## 📚 Dataset Info

- **Source:** [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Type:** Binary classification
- **Features:** 8 input variables
- **Target:** Diabetic (1) or Non-diabetic (0)

---

## 🔮 Planned Enhancements

- 📌 SHAP value visualizations for deeper interpretability
- 📌 Confidence intervals for predictions
- 📌 Feature interaction analysis
- 📌 Reference ranges with health indicators
- 📌 Prediction history tracking and logging

---

## 🧑‍💻 Author

Made with ❤️ by **P Sanjeev Pradeep**

Feel free to ⭐ the repo if you find it helpful or open an issue to contribute!

