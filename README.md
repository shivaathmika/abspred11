# 🧪 OralAbsPredict

**OralAbsPredict** is a Streamlit-based web application designed to predict key oral absorption properties of drug-like molecules using machine learning models.

It predicts:

- **Human Intestinal Absorption (HIA)**
- **Human Oral Bioavailability (HOB)** at:
  - 50% threshold
  - 20% threshold
- **Applicability Domain (AD)** status for reliability assessment

---

## 🚀 Features

### 🔹 Single Molecule Prediction
- Input SMILES string or draw molecule using Ketcher
- Visualize 2D structure
- Get:
  - HIA class (High/Low)
  - HOB class (High/Low)
  - AD status

### 🔹 Batch Prediction
- Upload `.xlsx` file with SMILES column
- Predict multiple molecules at once
- Export results

### 🔹 Built-in Documentation
- Explanation of HIA & HOB
- Model details
- Input/output formats

---

## 🧠 Models Used

| Property | Model | Descriptor |
|----------|------|-----------|
| HIA | Support Vector Classifier | FCFP (radius=1) |
| HOB (50%) | Random Forest | Mordred descriptors |
| HOB (20%) | Support Vector Classifier | FCFP (radius=2) |

---
