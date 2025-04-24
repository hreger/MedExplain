# 💡 MedExplain 🧠💉

**MedExplain** is an **AI-driven medical diagnosis support tool** that uses **explainable AI (XAI)** techniques to assist doctors and researchers in understanding and trusting machine learning predictions. It provides **transparent, interpretable results** for disease diagnosis using patient data, and ensures **reproducibility and traceability** through a full MLops pipeline.

> Designed for **healthcare professionals and data scientists**, MedExplain bridges the gap between ML accuracy and medical accountability.

---

## 🚀 MVP Features

### ✅ Core
- 🔍 **Disease Prediction Model** — Trained on medical datasets (e.g., diabetes, heart disease).
- 🧠 **Explainability** — Integrated **LIME** and **SHAP** to visualize feature contributions.
- 📊 **Interactive UI** — Built with **Streamlit**/**Gradio** for demo/testing.
- 📁 **MLflow Tracking** — Logs model metrics, parameters, and artifacts.
- ⏳ **DVC Pipelines** — Version-controlled data, code, and models for reproducibility.

---

## 🛠️ Tech Stack

| Category         | Tools Used                          |
|------------------|-------------------------------------|
| Language         | Python 3.11                         |
| Modeling         | scikit-learn, XGBoost               |
| Explainability   | LIME, SHAP                          |
| Dashboarding     | Streamlit / Gradio                  |
| Experimentation  | MLflow                              |
| Data Management  | DVC + Git                           |
| Visualization    | matplotlib, seaborn                 |

---

## 🧪 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/hreger/MedExplain.git
cd MedExplain
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The requirements.txt file now includes additional dependencies such as `scipy`, `cloudpickle`, `packaging`, `psutil`, and `pyyaml` to ensure full compatibility with the latest model artifacts and utilities.
pip install -r requirements.txt
cd MedExplain

### 3. Run the App (Streamlit)

```bash
streamlit run app.py
```

Or for Gradio:

```bash
python gradio_ui.py
```

### 4. Track Experiments (Optional)

```bash
mlflow ui
```

---

## 🗂️ Directory Structure

```
MedExplain/
├── data/                   # Raw and processed datasets (DVC tracked)
├── models/                 # Saved models (DVC tracked)
├── src/                    # ML pipelines and utils
│   ├── train.py
│   ├── predict.py
│   └── explain.py
├── app.py                  # Streamlit frontend
├── gradio_ui.py            # Gradio alternative frontend
├── dvc.yaml                # Pipeline config
├── mlruns/                 # MLflow run artifacts
├── requirements.txt
└── README.md
```

---

## 📈 Sample Output

- **Prediction**: `High risk of diabetes`
- **Top Features**:
  - Glucose level (↑)
  - BMI (↑)
  - Age (↑)
- **SHAP Summary Plot**: Explains overall model behavior
- **LIME Plot**: Explains single prediction in local context

---

## 📚 Dataset Used (MVP)

- [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- More datasets can be plugged into the DVC pipeline.

---

## 🔮 Future Scope

- ✅ Add multi-disease support (heart, kidney, etc.)
- ✅ Dockerize the app for deployment.
- ✅ HIPAA/GDPR-aligned audit trails.
- ✅ Connect to EHR systems (FHIR APIs).

---

Made with ❤️ by [ P Sanjeev Pradeep ]
