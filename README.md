#  Crop Yield Predictor — India

A machine learning web application that predicts crop yield (kg/hectare) based on Indian state-level agriculture data. Built during a summer internship at **NIT Patna**.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Project Overview

Agriculture contributes ~18% of India's GDP and supports over 50% of its workforce. Despite this, farmers and policymakers often lack data-driven tools to forecast crop output. This project addresses that gap by applying supervised machine learning to real Indian agricultural data — enabling instant yield predictions based on crop type, location, season, and environmental conditions.

---

## 🚀 Live Demo

Run locally with:
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

##  App Preview

| Input Panel | Prediction Result |
|---|---|
| Select State, Crop, Season, District | Yield shown in kg/hectare |
| Enter Area, Temperature, Humidity | Color-coded result (Low / Moderate / High) |

---

##  Models Trained & Compared

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| Linear Regression | ~0.71 | ~0.87 | ~0.65 |
| Decision Tree | ~0.83 | ~0.67 | ~0.48 |
| KNN (k=5) | ~0.76 | ~0.79 | ~0.57 |
| **Random Forest** ✅ | **~0.89** | **~0.53** | **~0.38** |

> Random Forest was selected as the final model due to its highest R² and lowest error.

---

##  Key Insights from EDA

- **Top predictor**: Crop type and Area are the strongest features influencing yield
- **Season matters**: Kharif crops show higher yield variability than Rabi crops
- **State-wise**: Punjab, Haryana, and Andhra Pradesh consistently show higher yields
- **Yield is right-skewed** — log transformation was applied before training

---

##  Project Structure

```
crop-yield-predictor-india/
│
├── app.py                        # Streamlit web application
├── crop_yield_predic.ipynb       # Full ML pipeline notebook
├── crop_yield_india^.csv.xlsx    # Dataset (Indian agriculture data)
├── rf_model.pkl                  # Trained Random Forest model
├── scaler.pkl                    # StandardScaler for KNN
├── README.md                     # This file
```

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data handling | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Web App | Streamlit |
| Environment | VS Code + Jupyter Notebook |

---

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/crop-yield-predictor-india.git
cd crop-yield-predictor-india
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit openpyxl
```

**3. Run the Jupyter notebook** (to retrain models)
```bash
# Open crop_yield_predic.ipynb in VS Code and run all cells
```

**4. Launch the Streamlit app**
```bash
streamlit run app.py
```

---

##  ML Pipeline

```
Raw Data → EDA → Preprocessing → Model Training → Evaluation → Streamlit App
```

1. **EDA** — Yield distribution, top crops, state-wise analysis, correlation heatmap
2. **Preprocessing** — Created Yield column (Production/Area), label encoding, log transform, train/test split
3. **Modeling** — Trained Linear Regression, Decision Tree, KNN, Random Forest
4. **Evaluation** — Compared R², RMSE, MAE across all models
5. **Feature Importance** — Identified top predictors using Random Forest
6. **Deployment** — Streamlit app with real-time prediction

---

##  Dataset

- **Source**: Indian Agriculture Data (Kaggle)
- **Records**: ~49,999 rows
- **Features**: State, District, Crop, Season, Area, Temperature, Humidity, Soil Moisture, Production
- **Target**: Yield (kg/hectare) — derived as Production / Area

---

##  About

This project was built as part of a **Summer Internship at NIT Patna** (2026) by a B.Tech student in Computer Science & Engineering.

**Algorithms used**: Linear Regression · Logistic Regression · Decision Tree · Random Forest · KNN

---

##  License

This project is open source under the [MIT License](LICENSE).

---

