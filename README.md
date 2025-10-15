# Data-Driven Loan Approval Prediction System

## 📘 Project Overview
The **Data-Driven Loan Approval Prediction System** is a complete machine learning pipeline that predicts whether a loan application should be **approved or rejected** based on applicant details such as income, credit score, and asset values.

The project demonstrates the full lifecycle of an ML project — from **data preprocessing** and **model training** to **evaluation** and **deployment** using Streamlit.

---

---

## Live Demo

Try the live Streamlit app here: *[link](https://aprilyab-data-driven-loan-approval-predict-streamlit-app-wjtgcq.streamlit.app/)*

---


## 🚀 Key Features
- End-to-end **ML pipeline** (data cleaning → training → evaluation → deployment)
- Supports multiple algorithms (Logistic Regression, Decision Tree)
- Handles **imbalanced datasets** using **SMOTE**
- Provides **visual model interpretation**
- Interactive **Streamlit web app** for real-time predictions

---

## 🧠 Dataset
- **Raw Data File:** `data/raw/loan_approval_dataset.csv`
- **Processed File:** `data/processed/processed_loan_approval_dataset.csv`
- **Target Variable:** `loan_status` → Approved (1) or Rejected (0)

### 🏦 Dataset Features

| Feature | Description | Range |
|----------|-------------|--------|
| `no_of_dependents` | Number of dependents | 0–5 |
| `education` | Applicant education level | Graduate / Not Graduate |
| `self_employed` | Employment status | Yes / No |
| `income_annum` | Annual income of applicant | 200,000 – 9,900,000 |
| `loan_amount` | Requested loan amount | 300,000 – 39,500,000 |
| `loan_term` | Duration of loan in years | 2 – 20 |
| `cibil_score` | Credit score of applicant | 300 – 900 |
| `residential_assets_value` | Value of owned residential assets | -100,000 – 29,100,000 |
| `commercial_assets_value` | Value of owned commercial assets | 0 – 19,400,000 |
| `luxury_assets_value` | Value of luxury assets | 300,000 – 39,200,000 |
| `bank_asset_value` | Value of bank savings | 0 – 14,700,000 |

---

## 🧩 Project Structure
```text
Data-Driven Loan Approval Prediction System/
│
├── .venv/                       # Python virtual environment
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Cleaned and preprocessed dataset
│
├── notebooks/
│   ├── Exploratory_Data_Analysis.ipynb
│   ├── data_cleaning.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
│
├── outputs/
│   ├── figures/                 # Visualizations
│   ├── models/                  # Saved ML models
│   └── metrics.json             # Model evaluation metrics
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning_and_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│
├── streamlit_app.py             # Streamlit app for predictions
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🧮 Data Cleaning and Preprocessing

**Steps performed:**
1. Dropped missing and duplicate records.
2. Removed non-essential columns (e.g., `loan_id`).
3. Scaled numerical columns using `StandardScaler`.
4. Encoded categorical variables with **OneHotEncoder**.
5. Encoded the target column using **LabelEncoder**.
6. Balanced the dataset using **SMOTE**.
7. Saved the cleaned dataset to `data/processed/processed_loan_approval_dataset.csv`.

---

## 🤖 Model Training

### Models Implemented
- **Decision Tree Classifier**
- **Logistic Regression**

### Training Steps
1. Split dataset into **70% training** and **30% testing**.
2. Applied **SMOTE** to handle class imbalance.
3. Trained models on the processed dataset.
4. Saved models in `outputs/models/` directory.

---

## 📊 Model Evaluation

| Model | Accuracy Score | Log Loss |
|--------|----------------|----------|
| Decision Tree | 0.9649 | 0.1770 |
| Logistic Regression | 0.9110 | 0.2350 |

**Saved metrics:**
```json
{
  "Decision Tree": {"Accuracy Score": 0.9648711943793911, "Log Loss": 0.1770538263853407},
  "Logistic Regression": {"Accuracy Score": 0.9110070257611241, "Log Loss": 0.23504903392632293}
}
```

The **Decision Tree** model was selected for deployment due to its higher accuracy and lower log loss.

---

## 🌐 Streamlit Application

### Features:
- Interactive form to input applicant details.
- Predicts loan approval or rejection instantly.
- Displays key influencing factors visually.

**Run the app:**
```bash
streamlit run streamlit_app.py
```

---

## 🧰 Core Libraries
- `pandas` → Data manipulation  
- `numpy` → Numerical operations  
- `scikit-learn` → ML algorithms, preprocessing, model evaluation  
- `imblearn` → Handling class imbalance with SMOTE  
- `streamlit` → Web deployment  

---

## 🧾 How to Run Locally

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Data-Driven-Loan-Approval-Prediction-System
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run preprocessing and training scripts:
   ```bash
   python src/data_cleaning_and_preprocessing.py
   python src/model_training.py
   ```

5. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🧭 Main Factors Influencing Loan Approval

| Factor | Description |
|--------|-------------|
| **CIBIL Score** | Higher scores greatly increase approval chances. |
| **Income and Assets** | Applicants with higher income and asset values are more likely to be approved. |
| **Loan Amount & Term** | Large loans or long terms can reduce approval likelihood. |
| **Employment Type** | Self-employed applicants may face higher risk. |
| **Education** | Graduate applicants are statistically more likely to be approved. |

---

## 📚 References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 👨‍💻 Author
**Name:** Henok Yoseph  
**Email:** [henokapril@gmail.com](mailto:henokapril@gmail.com)  
**GitHub:** [aprilyab](https://github.com/aprilyab)
