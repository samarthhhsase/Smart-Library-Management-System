# 📚 Smart Library Management System (SLMS)
### Using Statistical Machine Learning and Data Science
**Third-Year Engineering Practical Examination & Major Project Submission**  
*Department of Computer Science & Engineering / Information Technology / Artificial Intelligence & Data Science*

---

## 🌟 Executive Abstract

The **Smart Library Management System (SLMS)** is an end-to-end, production-grade academic software platform that unifies traditional institutional book circulation workflows with advanced statistical inference and predictive machine learning. Rather than functioning as a passive relational record keeper, SLMS transforms circulation data into an active **Decision Support System (DSS)**.

The system natively integrates all **10 Laboratory Experiments** from the university **Statistical Machine Learning (SML) and Data Science (DS)** practical syllabus into a unified, interactive Streamlit application. Leveraging SQLite for persistent relational storage, Scikit-learn for supervised and unsupervised algorithms, SciPy and Statsmodels for inferential hypothesis testing, and Plotly for responsive telemetry, SLMS equips academic librarians with actionable foresight—from forecasting semester-level book demand to mitigating overdue book risks without punitive student profiling.

---

## 🎯 Problem Statement & Objectives

### The Problem
Traditional college libraries suffer from significant operational inefficiencies:
1. **Unbalanced Inventory Utilization:** Certain academic domains (e.g., Computer Science, AI) face chronic copy shortages during examination periods, while other sections experience shelf stagnation.
2. **High Overdue Attrition Rates:** Overdue books deprive peers in reserve queues, yet manual phone calls or blanket penalty fees cause friction and administrative overhead.
3. **Siloed Practical Experiments:** In university curricula, statistical techniques (ANOVA, t-tests, PCA, regression, tree classifiers) are often taught as isolated scripts on disparate synthetic toy datasets, depriving engineering students of practical full-stack context.

### Project Objectives
- **Centralized Circulation Operations:** Provide full CRUD capabilities for books, student directories, and transaction checkouts/check-ins.
- **Statistical Rigor:** Conduct exploratory data analysis (EDA), central tendency and dispersion calculations, 1.5×IQR outlier detection, and empirical distribution shape diagnostics (skewness and kurtosis).
- **Inferential Hypothesis Testing:** Formulate and test null ($H_0$) and alternative ($H_1$) hypotheses on student loan behaviors using independent two-sample t-tests, Chi-Square ($\chi^2$) tests of independence, and One-Way ANOVA.
- **Predictive Supervised Modeling:** Compare Linear Regression, Regularized Regression (Ridge, Lasso, Elastic Net), Logistic Regression, KNN, SVM, Naive Bayes, Decision Trees, and Random Forests.
- **Unsupervised Pattern Discovery:** Perform orthogonal dimensionality reduction via Principal Component Analysis (PCA) and discover student borrowing cohorts via K-Means, Hierarchical Dendrograms, and DBSCAN.
- **Real-World Decision Support:** Deploy an interactive Librarian Console with real-time risk gauges, inference latency profiling, and ethical AI safeguards.

---

## 💻 System Requirements & Technology Stack

### Hardware Requirements
- **Processor:** Any modern dual-core or quad-core x86/ARM CPU (Intel Core i3/i5/i7/i9, AMD Ryzen, Apple Silicon).
- **RAM:** Minimum 4 GB (8 GB recommended).
- **Storage:** Minimum 500 MB free disk space.
- **GPU:** *None required* (All algorithms run efficiently on CPU via optimized Scikit-learn and SciPy routines).

### Software Requirements
- **Operating System:** Windows 10/11, macOS (10.15+), or Linux (Ubuntu 20.04+).
- **Runtime Environment:** Python 3.10 to Python 3.14.
- **Browser:** Any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, Brave, Safari).

### Technology Stack
| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Frontend UI** | Streamlit | `>= 1.30.0` | Modern, responsive dashboard, tabs, and interactive controls |
| **Data Engine** | Pandas & NumPy | `>= 2.0.0` | Dataframe manipulation, matrix algebra, statistical aggregations |
| **Visualizations** | Plotly & Seaborn | `>= 5.15.0` | Interactive charts, heatmaps, gauge meters, and distributions |
| **Statistical Testing** | SciPy & Statsmodels | `>= 1.11.0` | t-tests, Chi-square, ANOVA, Levene's test, Shapiro-Wilk |
| **Machine Learning** | Scikit-learn | `>= 1.3.0` | Regression, classification, ensemble trees, PCA, clustering |
| **Database** | SQLite3 | Native | ACID-compliant relational persistence (`library.db`) |
| **Model Persistence** | Joblib | `>= 1.3.0` | Serializing and loading trained Scikit-learn pipelines |

---

## 🏗️ System Architecture & Database Schema

### Architecture Diagram
```
+---------------------------------------------------------------------------------------+
|                    Streamlit Frontend Dashboard (app.py)                              |
|   - Dynamic Sidebar Navigation       - Reactive Forms & KPI Cards                     |
|   - Custom Glassmorphic Dark CSS      - Interactive Plotly Visualizations              |
+---------------------------------------------------------------------------------------+
        |                                       |                               |
        v                                       v                               v
+-----------------------+           +-----------------------+           +-----------------------+
| Core Management Layer |           | Statistical DS Lab    |           | Machine Learning Hub  |
| - book_management.py  |           | - eda.py (Exp 1)      |           | - regression_models   |
| - student_management  |           | - preprocessing (Exp 2|           | - classification_mod  |
| - transactions.py     |           | - statistics_analysis |           | - tree_models.py      |
|   (Checkout/Check-in) |           | - hypothesis_testing  |           | - clustering.py (Exp9)|
+-----------------------+           +-----------------------+           +-----------------------+
        \                                       |                               /
         \                                      v                              /
          +-------------------------------------------------------------------+
                                                |
                                                v
+---------------------------------------------------------------------------------------+
|                        Data Access Layer (database/db.py)                             |
|          Thread-Safe SQLite Engine: library.db (Normalized Relational Schema)         |
+---------------------------------------------------------------------------------------+
               |                                |                               |
               v                                v                               v
       [ books Table ]                  [ students Table ]              [ transactions Table ]
       - book_id (PK)                   - student_id (PK)               - transaction_id (PK)
       - title, author                  - name, email                   - student_id (FK)
       - category, isbn                 - department                    - book_id (FK)
       - pub_year, copies               - year_of_study                 - issue_date, due_date
       - available_copies               - contact                       - return_date, overdue_days
       - shelf_location                                                 - overdue_status, loan_duration
```

### Relational Schema Definition
1. **`books` Table:**
   - `book_id` (TEXT, PK): Canonical alphanumeric identifier (e.g. `BK-1001`).
   - `title` (TEXT), `author` (TEXT), `category` (TEXT): Core catalog metadata.
   - `isbn` (TEXT, UNIQUE): International Standard Book Number.
   - `publication_year` (INTEGER): Publication timestamp for computing book age.
   - `total_copies` (INTEGER), `available_copies` (INTEGER): Dynamic stock indicators.
   - `shelf_location` (TEXT): Physical stack rack location (e.g. `CS-1A`).
2. **`students` Table:**
   - `student_id` (TEXT, PK): Unique student enrollment number (e.g. `STU-2001`).
   - `name` (TEXT), `email` (TEXT), `contact` (TEXT): Identity credentials.
   - `department` (TEXT), `year_of_study` (INTEGER 1-4): Academic profile.
3. **`transactions` Table:**
   - `transaction_id` (TEXT, PK): Unique loan event code (e.g. `TXN-5001`).
   - `student_id` (TEXT, FK), `book_id` (TEXT, FK): Relational link.
   - `issue_date` (TEXT YYYY-MM-DD), `due_date` (TEXT YYYY-MM-DD): Loan period.
   - `return_date` (TEXT or NULL): NULL indicates currently checked out.
   - `borrowing_frequency` (INTEGER): Cumulative borrows at checkout timestamp.
   - `overdue_days` (INTEGER): Calculated as $\max(0, \text{ReturnDate} - \text{DueDate})$.
   - `overdue_status` (INTEGER): Binary class label (0: On-Time, 1: Overdue).
   - `loan_duration_days` (INTEGER): Total duration of physical custody.

---

## 📊 Complete 10-Experiment Practical Syllabus Integration

| Exp # | Practical Syllabus Experiment | SLMS Module | Algorithms & Methods | Key Input Features | Primary Outputs | Evaluation Metrics |
|:---:|---|---|---|---|---|---|
| **1** | **Exploratory Data Analysis & Visualization** | `modules/eda.py` | Bar & Donut charts, Histograms, Box plots, Time-series, Pearson correlation heatmap | Catalog inventory, student records, circulation timestamps | Inventory share, seasonality trends, correlation matrix | Five-number summary ($\mu, \sigma, Q1, M, Q3$), Trend graphs |
| **2** | **Data Preprocessing & Feature Selection** | `modules/preprocessing.py` | OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, SelectKBest | Raw transaction logs, student/book metadata | Clean matrices, dummy variables, ANOVA F-scores | Null audit, Variance checks, Feature ranking ($F$, $p$-val) |
| **3** | **Statistical Techniques for Data Exploration** | `modules/statistics_analysis.py` | Central tendency, dispersion, Skewness, Kurtosis, 1.5×IQR Rule, Z-Score ($|Z|>3$) | Overdue days, loan durations, borrowing frequencies | Statistical moments, detected outliers, contingency tables | Mean, Median, Mode, Variance, Skew, Kurtosis, IQR bounds |
| **4** | **Hypothesis Testing with Real-World Data** | `modules/hypothesis_testing.py` | Independent 2-Sample $t$-test, Chi-Square ($\chi^2$) Independence, 1-Way ANOVA, Shapiro-Wilk | Departmental overdue days, category contingency counts | $H_0/H_1$ statements, test statistics, $p$-values, decisions | Critical $\alpha = 0.05$, Degrees of freedom, Levene test |
| **5** | **Linear & Logistic Regression** | `modules/regression_models.py` | OLS Linear Regression, Binary Logistic Regression with Sigmoid Link | Book specs for demand; Pre-issue attributes for overdue | Demand forecast, Overdue probability, Log-odds weights | Regression: $R^2$, Adj $R^2$, MAE, RMSE; Clf: Accuracy, ROC-AUC |
| **6** | **Ridge, Lasso & Elastic Net Regularization** | `modules/regression_models.py` | L2 Ridge, L1 Lasso (Feature sparsity), Elastic Net, $k$-Fold Cross-Validation | Standardized book characteristics | Optimal $\alpha$, Coefficient shrinkage bar comparison | CV Mean Squared Error, Out-of-sample $R^2$, Test RMSE |
| **7** | **Classification Models Comparison** | `modules/classification_models.py` | Logistic Regression, KNN, Support Vector Machine (SVM), Naive Bayes, Random Forest | Checkout transaction features, balanced class weights | Performance matrix, ROC curves, Confusion matrices | Accuracy, Precision, Recall (Sensitivity), F1-Score, ROC-AUC |
| **8** | **Decision Trees & Random Forest** | `modules/tree_models.py` | CART Decision Tree (Gini/Entropy), Random Forest Bagging Ensemble | Circulation attributes, student academic year | Visual tree plot, text rules, Gini feature importance | Overfitting gap (Train vs Test score), Recall, AUC |
| **9** | **PCA & Cluster Analysis** | `modules/clustering.py` | PCA (Eigenvector decomposition, Scree plot), K-Means, Dendrogram, DBSCAN | Aggregated student behavioral profiles | 2D PCA projection, Elbow curve, Behavioral persona profiles | Explained variance ratio, Silhouette Score, Davies-Bouldin |
| **10** | **Real-World Application & Performance Testing** | `modules/predictive_analytics.py` | Real-Time Inference Engine, Risk Gauges, Demand Forecaster, Latency Profiler | Live inputs from librarian checkout console | Risk tier (Low/Med/High), Advisory actions, Latency (ms) | Training time (ms), Inference latency ($\mu$s), Ethical guidelines |

---

## 🔬 Mathematical Formulations & Viva Voce Theory

### 1. Target Leakage Prevention (Crucial Practical Concept)
In predictive modeling, **Target Leakage** occurs when training features contain information unavailable at the moment of prediction.
- *Wrong Approach:* Training a model to predict overdue risk using `return_date`, `actual_loan_duration`, or `overdue_days`.
- *SLMS Correct Approach:* At point-of-checkout, only pre-issue historical attributes are supplied:
  $$\vec{x} = \left[ \text{borrowing\_frequency}, \text{student\_year}, \text{book\_pub\_year}, \text{book\_total\_copies}, \text{book\_age}, \text{issue\_month}, \text{category}, \text{department} \right]$$

### 2. Inferential Hypothesis Testing Equations
- **Independent Two-Sample $t$-Test (Equal Variance):**
  $$t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}, \quad \text{where } s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$
- **Chi-Square ($\chi^2$) Test of Independence:**
  $$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad \text{where } E_{ij} = \frac{\text{Row Total}_i \times \text{Col Total}_j}{N}$$
- **One-Way ANOVA ($F$-Statistic):**
  $$F = \frac{\text{Mean Square Between (MSB)}}{\text{Mean Square Within (MSW)}} = \frac{SSB / (k - 1)}{SSW / (N - k)}$$

### 3. Regularization Loss Formulations
- **Ridge Regression (L2 Weight Penalty):**
  $$\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p \beta_j^2$$
  *Property:* Shrinks coefficients asymptotically toward zero; handles high multicollinearity.
- **Lasso Regression (L1 Sparsity Penalty):**
  $$\mathcal{L}_{\text{Lasso}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p |\beta_j|$$
  *Property:* Shrinks coefficients and forces non-essential predictors to exactly $0$, performing automatic feature selection.
- **Elastic Net (Convex Combination):**
  $$\mathcal{L}_{\text{ElasticNet}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \left[ \rho \sum_{j=1}^p |\beta_j| + \frac{1-\rho}{2} \sum_{j=1}^p \beta_j^2 \right]$$

### 4. Why Recall Trumps Accuracy in Library Risk Management
In institutional lending, classes are naturally imbalanced ($\sim 28\%$ overdue vs $\sim 72\%$ on-time):
- **False Positive (Type I Error):** A student predicted to return late actually returns on time. Consequence: The student receives a courteous automated reminder email 2 days early. Minimal inconvenience.
- **False Negative (Type II Error):** An overdue loan is predicted as on-time. Consequence: No proactive reminder is sent. The book becomes untracked or lost, depriving waiting students.
Therefore, the engineering objective is to maximize **Recall (Sensitivity)**:
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

---

## 🚀 Installation & Execution Instructions

### Prerequisites
Make sure Python 3.10+ and pip are installed. Verify in your terminal:
```bash
python --version
pip --version
```

### Step 1: Clone or Navigate to Project Directory
```bash
cd d:\library
```

### Step 2: Install Dependencies
Install all required libraries using the frozen `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Application
Run the Streamlit application:
```bash
streamlit run app.py
```

The system will start a local web server (typically at `http://localhost:8501`) and open the dashboard in your default browser automatically.

---

## 🧪 College Practical Demonstration Guide

Follow this systematic 6-step walkthrough during your practical exam or project viva:

1. **Demonstrate Core Library Management:**
   - Go to **Book Management** $\rightarrow$ Search for "Algorithms" $\rightarrow$ Add a new book $\rightarrow$ Confirm its assignment to the catalog.
   - Go to **Issue & Return** $\rightarrow$ Issue the book to a student $\rightarrow$ Verify that the available copies count automatically decrements in the database.
2. **Demonstrate EDA & Statistics (Exps 1 & 3):**
   - Open **Exploratory Data Analysis** $\rightarrow$ Showcase the monthly circulation curve highlighting exam season volume bursts $\rightarrow$ Inspect the Pearson correlation heatmap.
   - Open **Statistical Analysis** $\rightarrow$ Point out the right-skewed overdue days distribution $\rightarrow$ Showcase the 1.5×IQR outlier detection table.
3. **Demonstrate Inferential Hypothesis Testing (Exp 4):**
   - Open **Hypothesis Testing** $\rightarrow$ Select the **Chi-Square test** $\rightarrow$ Explain the Contingency Table (Observed vs Expected) $\rightarrow$ Point out the calculated $p$-value and conclude whether book category is independent of overdue status at $\alpha = 0.05$.
4. **Demonstrate Machine Learning Models (Exps 5, 6, 7, 8):**
   - Open **Regression Models** $\rightarrow$ Highlight the **Coefficient Shrinkage Chart** demonstrating how Lasso sets weights to zero while Ridge shrinks them smoothly.
   - Open **Classification Models** $\rightarrow$ Compare ROC curves across Logistic Regression, KNN, SVM, Naive Bayes, and Random Forest $\rightarrow$ Defend why Recall is the key metric.
   - Open **Decision Tree & Random Forest** $\rightarrow$ Inspect the visual tree plot $\rightarrow$ Show the Random Forest Gini feature importance ranking.
5. **Demonstrate Unsupervised Learning (Exp 9):**
   - Open **PCA & Clustering** $\rightarrow$ Show the Scree Plot explaining how PC1 + PC2 capture majority variance $\rightarrow$ Showcase the Elbow curve $\rightarrow$ Inspect K-Means student cohorts projected on 2D PCA space.
6. **Live Decision Support Console (Exp 10):**
   - Open **Predictive Decision Support** $\rightarrow$ Select a student and book $\rightarrow$ Click **Evaluate Overdue Risk Probability** $\rightarrow$ Observe the real-time gauge score, the recommended intervention tier (Low/Moderate/Elevated), and single-sample inference latency ($< 5\text{ ms}$).

---

## 🛡️ Limitations & Future Scope

### Current Limitations
- **Synthetic Base Sample:** The baseline dataset contains 120 books, 65 students, and 700 transactions. While mathematically correlated, real college libraries operate with tens of thousands of records. (A CSV upload tool is provided in the sidebar to ingest real data).
- **Physical Barcode / RFID Integration:** The current interface relies on digital selection rather than hardware RFID scanner inputs.

### Future Scope
- **RFID & Barcode Scanner Integration:** Add hardware scanner support via WebUSB for instant book check-in and checkout.
- **Collaborative Filtering Book Recommender:** Integrate matrix factorization (SVD / Neural Collaborative Filtering) to recommend academic textbooks based on a student's enrolled courses and past reading history.
- **Automated Messaging Integration:** Connect Twilio / WhatsApp Business API to deliver automated SMS courtesy reminders based on predicted high-risk tiers.
- **Multi-Branch Inter-Library Loans:** Extend the relational schema to support inter-library transfers between multiple campus branches.

---

## 👨‍💻 Author & Academic Submission
- **Project Title:** Smart Library Management System (SLMS)
- **Academic Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering
- **Laboratory Subject:** Statistical Machine Learning & Data Science Practical Lab
- **Submission Date:** Academic Term 2026
