# 🧠 Breast Cancer Diagnosis using Machine Learning

## 📌 Overview
This project presents a comparative analysis of machine learning algorithms for breast cancer diagnosis using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The goal is to classify tumors as **Malignant (M)** or **Benign (B)** and identify the most important features for early detection.

---

## 🎯 Objectives
- Apply and compare multiple ML algorithms
- Evaluate performance using clinical metrics
- Identify key features influencing diagnosis
- Build an interpretable and reproducible ML pipeline

---

## 📊 Dataset
- **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC)
- **Total Samples**: 569
- **Features**: 30 numerical features
- **Classes**:
  - Malignant (37.3%)
  - Benign (62.7%)

---

## ⚙️ Machine Learning Models Used
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)

---

## 🔬 Methodology
1. Data preprocessing (scaling, encoding)
2. Exploratory Data Analysis (EDA)
3. Train-test split (80/20)
4. Model training
5. Evaluation using:
   - Accuracy
   - Precision
   - Recall (Sensitivity)
   - F1-score
   - AUC-ROC
6. 5-fold cross-validation

---

## 📈 Results Summary

| Model                | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---------------------|----------|----------|--------|----------|---------|
| Logistic Regression | 96.49%   | 0.9750   | 0.9286 | 0.9512   | 0.9960  |
| Random Forest       | **97.37%** | **1.0000** | 0.9286 | **0.9630** | 0.9929  |
| KNN                 | 95.61%   | 0.9744   | 0.9048 | 0.9383   | 0.9823  |

---

## 🧠 Key Insights
- Random Forest achieved the highest accuracy (97.37%)
- Logistic Regression showed best generalization and highest AUC
- Worst-case features (e.g., concave points, radius) are strongest predictors
- Size-related features (radius, area, perimeter) are highly discriminative

---

## ⚠️ Clinical Relevance
- False negatives (missed cancer cases) are critical
- Models achieved ~93% recall for malignant detection
- Can be used as a **decision-support tool**, not a replacement for doctors

---

## 📉 Limitations
- Small dataset (569 samples)
- No hyperparameter tuning
- Single dataset used (WDBC only)
- No advanced techniques like SMOTE or PCA applied

---

## 🚀 Future Improvements
- Hyperparameter tuning (Grid Search)
- Add models like SVM, XGBoost
- Apply feature selection / PCA
- Handle class imbalance (SMOTE)
- Add explainability (SHAP, LIME)

---


---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 📎 Research Paper
📄 Full paper available in this repository:  
👉 `research_paper.pdf`

---

## 🙌 Acknowledgements
- UCI Machine Learning Repository
- Wisconsin Breast Cancer Dataset creators

---

## ⭐ If you found this useful
Consider giving this repo a star ⭐
