"""
Project Information & Practical Syllabus Integration Guide.
Contains the complete 10-experiment syllabus mapping table, architectural diagrams,
theoretical equations, and college viva examination prep.
"""

import streamlit as st
import pandas as pd

def render_project_info():
    st.title("📖 Project Information & Syllabus Mapping")
    st.caption("Statistical Machine Learning and Data Science Practical Syllabus — Unified SLMS Project")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Experiment Mapping Table",
        "🏗️ System Architecture & Schema",
        "🎓 Viva Voce Q&A Prep",
        "📄 College Submission Guide"
    ])

    # TAB 1: Experiment Mapping Table
    with tab1:
        st.subheader("10-Experiment Practical Syllabus Mapping Table")
        st.write("Every laboratory experiment is seamlessly integrated into this unified Smart Library Management System:")

        experiments_data = [
            {
                "Exp #": "Exp 1",
                "Experiment Name": "Exploratory Data Analysis & Visualization",
                "SLMS Module": "modules/eda.py",
                "Key Algorithms / Techniques": "Histograms, Bar/Donut Charts, Box Plots, Pearson Correlation Heatmap, Time-Series",
                "Input Data": "Books inventory, Circulation logs, Student directory",
                "Output / Deliverables": "Inventory distribution, Peak loan seasonality, Correlation matrix",
                "Evaluation Method": "Summary statistics (Mean, Std, IQR), Data distribution diagnostics"
            },
            {
                "Exp #": "Exp 2",
                "Experiment Name": "Data Preprocessing & Feature Selection",
                "SLMS Module": "modules/preprocessing.py",
                "Key Algorithms / Techniques": "OneHotEncoder, StandardScaler, MinMaxScaler, SelectKBest (ANOVA F-test, F-regression)",
                "Input Data": "Raw heterogeneous student transactions",
                "Output / Deliverables": "Leakage-free scaled feature matrices, Ranked feature scores",
                "Evaluation Method": "Before-and-after distribution checks, ANOVA F-scores, p-values"
            },
            {
                "Exp #": "Exp 3",
                "Experiment Name": "Statistical Techniques for Data Exploration",
                "SLMS Module": "modules/statistics_analysis.py",
                "Key Algorithms / Techniques": "Central tendency (μ, M, Mo), Dispersion (σ, σ², IQR), Skewness, Kurtosis, 1.5×IQR Outlier Rule, Z-Score (|Z|>3)",
                "Input Data": "Circulation duration and overdue records",
                "Output / Deliverables": "Statistical moments table, Detected outlier transactions, Bivariate cross-tabulations",
                "Evaluation Method": "Fisher-Pearson skewness, Excess kurtosis, Upper/lower quartile boundaries"
            },
            {
                "Exp #": "Exp 4",
                "Experiment Name": "Hypothesis Testing with Real-World Data",
                "SLMS Module": "modules/hypothesis_testing.py",
                "Key Algorithms / Techniques": "Independent Two-Sample t-Test, Chi-Square (χ²) Test of Independence, One-Way ANOVA, Shapiro-Wilk Normality",
                "Input Data": "Departmental overdue days, Category contingency tables",
                "Output / Deliverables": "Formulated H₀/H₁, Test statistics (t, χ², F), p-values, Decision rules",
                "Evaluation Method": "Comparison against significance threshold (α = 0.05), Levene's variance test"
            },
            {
                "Exp #": "Exp 5",
                "Experiment Name": "Linear & Logistic Regression",
                "SLMS Module": "modules/regression_models.py",
                "Key Algorithms / Techniques": "Ordinary Least Squares (OLS) Linear Regression, Logistic Regression with Sigmoid Link",
                "Input Data": "Book inventory specs for demand; Pre-issue attributes for overdue risk",
                "Output / Deliverables": "Demand predictions, Overdue class probabilities, Odds-ratio feature weights",
                "Evaluation Method": "Regression: R², Adjusted R², MAE, RMSE; Classification: Accuracy, Recall, ROC-AUC"
            },
            {
                "Exp #": "Exp 6",
                "Experiment Name": "Ridge, Lasso & Elastic Net Regression",
                "SLMS Module": "modules/regression_models.py",
                "Key Algorithms / Techniques": "L2 Ridge (Weight penalty), L1 Lasso (Sparse feature selection), Elastic Net (L1+L2 mix), k-Fold Cross-Validation",
                "Input Data": "Book characteristics and circulation demand",
                "Output / Deliverables": "Regularized models with optimal α, Coefficient shrinkage chart",
                "Evaluation Method": "CV Mean Squared Error, Out-of-sample R², Test RMSE comparison"
            },
            {
                "Exp #": "Exp 7",
                "Experiment Name": "Classification Models Comparison",
                "SLMS Module": "modules/classification_models.py",
                "Key Algorithms / Techniques": "Logistic Regression, KNN (k-Nearest Neighbors), Support Vector Machine (SVM), Gaussian Naive Bayes, Random Forest",
                "Input Data": "Circulation feature vectors with balanced class weights",
                "Output / Deliverables": "Side-by-side performance matrix, ROC-AUC comparison curves, Confusion matrices",
                "Evaluation Method": "Accuracy, Precision, Recall (Sensitivity), F1-Score, Specificity, ROC-AUC"
            },
            {
                "Exp #": "Exp 8",
                "Experiment Name": "Decision Trees & Random Forest",
                "SLMS Module": "modules/tree_models.py",
                "Key Algorithms / Techniques": "CART Decision Tree (Gini / Entropy), Random Forest Bootstrap Aggregation (Bagging), Feature Importance",
                "Input Data": "Overdue risk features and historical transactions",
                "Output / Deliverables": "Visualized Tree splits, Gini importance ranking, Overfitting gap analysis",
                "Evaluation Method": "Training vs Testing score gap, Gini impurity reduction, Recall"
            },
            {
                "Exp #": "Exp 9",
                "Experiment Name": "PCA & Cluster Analysis",
                "SLMS Module": "modules/clustering.py",
                "Key Algorithms / Techniques": "PCA (Eigenvalue decomposition, Scree plot), K-Means (Elbow & Silhouette), Agglomerative Hierarchical Dendrogram, DBSCAN",
                "Input Data": "Aggregated student borrowing behavioral profiles",
                "Output / Deliverables": "2D/3D PCA projection, Optimal cluster cohorts, Behavioral persona archetypes",
                "Evaluation Method": "Explained variance ratio, Silhouette Score, Davies-Bouldin Index"
            },
            {
                "Exp #": "Exp 10",
                "Experiment Name": "Real-World Application & Performance Testing",
                "SLMS Module": "modules/predictive_analytics.py",
                "Key Algorithms / Techniques": "Live Inference Engine, Gauge Charts, Semester Demand Forecaster, Execution Latency Benchmarking",
                "Input Data": "Real-time user inputs from active librarian checkout console",
                "Output / Deliverables": "Real-time risk scoring (Low/Med/High), Action recommendations, Execution speed benchmarks (ms)",
                "Evaluation Method": "Training time (ms), Single-sample inference latency (μs), Ethical AI safeguards"
            }
        ]

        df_exp = pd.DataFrame(experiments_data)
        st.dataframe(df_exp, use_container_width=True, hide_index=True)

    # TAB 2: System Architecture & Schema
    with tab2:
        st.subheader("System Architecture & SQLite Schema")
        st.markdown(r"""
        ```
        +-------------------------------------------------------------------------------+
        |                     Streamlit User Interface (Frontend Dashboard)            |
        +-------------------------------------------------------------------------------+
              |                                  |                                |
              v                                  v                                v
        +------------------+           +--------------------+           +---------------+
        | Core Management  |           | Data Science Lab   |           | Predictive ML |
        | - Books CRUD     |           | - EDA & Visuals    |           | - Regression  |
        | - Students CRUD  |           | - Preprocessing    |           | - Classifiers |
        | - Circulation    |           | - Statistics & Hyp |           | - Trees & PCA |
        +------------------+           +--------------------+           +---------------+
              |                                  |                                |
              +----------------------------------+--------------------------------+
                                                 |
                                                 v
        +-------------------------------------------------------------------------------+
        |                   Database Access Layer (database/db.py)                      |
        |            Thread-Safe SQLite Engine: library.db (Normalized Relational)       |
        +-------------------------------------------------------------------------------+
               |                                 |                                 |
               v                                 v                                 v
        [ books Table ]                  [ students Table ]              [ transactions Table ]
        - book_id (PK)                   - student_id (PK)               - transaction_id (PK)
        - title, author                  - name, email                   - student_id (FK)
        - category, isbn                 - department                    - book_id (FK)
        - pub_year, copies               - year_of_study                 - issue_date, due_date
        - available_copies               - contact                       - return_date, overdue_days
        ```
        """)

    # TAB 3: Viva Voce Q&A Prep
    with tab3:
        st.subheader("Third-Year Engineering Viva Voce Questions & Answers")
        
        with st.expander("Q1: What is target leakage and how is it prevented in this library system?"):
            st.markdown("""
            **Answer:** Target leakage occurs when features included in model training contain information that would only be known *after* the target event has taken place.
            In our system, when predicting overdue risk at the time a book is being issued, we **strictly exclude** variables such as `return_date`, `actual_loan_duration`, or `overdue_days`. We only feed features known at checkout (e.g. historical borrowing frequency, student year, book age, issue month).
            """)

        with st.expander("Q2: Why is Recall more important than Accuracy for overdue book prediction?"):
            st.markdown("""
            **Answer:** Because library overdue events are relatively rare (class imbalance, ~25-30%), predicting "always on-time" yields high accuracy but misses all overdue events. A **False Negative** means an overdue book is missed without automated reminders, whereas a **False Positive** merely results in a friendly courtesy email to a diligent student. Hence, maximizing **Recall (Sensitivity)** is mathematically and operationally vital.
            """)

        with st.expander("Q3: What is the difference between L1 (Lasso) and L2 (Ridge) Regularization?"):
            st.markdown(r"""
            **Answer:**
            - **Ridge (L2):** Adds a squared penalty $\alpha \sum \beta_j^2$. It shrinks weights continuously towards zero, effectively handling multicollinearity, but never drives coefficients to absolute zero.
            - **Lasso (L1):** Adds an absolute penalty $\alpha \sum |\beta_j|$. Because its diamond constraint has sharp corners on axes, it sets redundant coefficients strictly to 0, performing automatic **feature selection**.
            """)

        with st.expander("Q4: How does PCA reduce dimensionality without losing essential information?"):
            st.markdown(r"""
            **Answer:** PCA computes the covariance matrix of standardized features and extracts its eigenvectors and eigenvalues. The eigenvectors represent orthogonal axes (Principal Components) along which the data exhibits maximum variance. By retaining only the first $k$ components that explain 80-95% of cumulative variance, we compress dimensionality while preserving the primary geometric structure.
            """)

        with st.expander("Q5: What is the difference between parametric and non-parametric hypothesis tests?"):
            st.markdown("""
            **Answer:** Parametric tests (such as Student's t-test and ANOVA) assume that the continuous variable follows a normal distribution and has equal variances across groups. If the Shapiro-Wilk test indicates significant non-normality (p < 0.05), non-parametric alternatives (such as the Mann-Whitney U test or Kruskal-Wallis test) based on median ranks must be preferred.
            """)

    # TAB 4: College Submission Guide
    with tab4:
        st.subheader("Practical Exam Demonstration Walkthrough")
        st.markdown("""
        ### Step-by-Step Practical Demonstration for the External Examiner:
        1. **Show Core Functionality:**
           - Navigate to **Book Management** -> Search a title -> Add a new book.
           - Navigate to **Issue and Return** -> Issue the book to a student -> Demonstrate available copies decreasing by 1.
        2. **Demonstrate EDA & Statistics (Exps 1 & 3):**
           - Open **Exploratory Data Analysis** -> Showcase the monthly circulation curve and correlation heatmap.
           - Open **Statistical Analysis** -> Highlight the right-skewed distribution of overdue days and the 1.5×IQR outlier detection.
        3. **Demonstrate Hypothesis Testing (Exp 4):**
           - Open **Hypothesis Testing** -> Run the **Chi-Square test** -> Point out the contingency table, calculated p-value, and decision rule at $\alpha = 0.05$.
        4. **Demonstrate ML Models (Exps 5, 6, 7, 8):**
           - Showcase **Regression Models** -> Point out the coefficient shrinkage plot comparing OLS vs Ridge vs Lasso.
           - Open **Classification Models** -> Compare ROC curves across 5 classifiers -> Explain the Recall score.
           - Open **Decision Tree & Random Forest** -> Inspect the visual tree plot and feature importances.
        5. **Demonstrate Unsupervised Learning (Exp 9):**
           - Open **PCA & Clustering** -> Show the Scree plot -> Show the K-Means cluster projection on 2D PCA.
        6. **Live Real-World Prediction (Exp 10):**
           - Open **Predictive Analytics** -> Choose a student and book -> Hit **Evaluate Overdue Risk** -> Observe the real-time gauge score and inference latency (< 5ms).
        """)
