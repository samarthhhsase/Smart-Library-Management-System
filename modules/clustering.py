"""
Experiment 9: Principal Component Analysis (PCA) and Cluster Analysis.
Performs PCA dimensionality reduction, scree plot analysis, K-Means clustering,
elbow/silhouette optimization, hierarchical dendrograms, and DBSCAN anomaly clustering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

from database.db import get_master_dataset, get_all_students
from utils.helpers import render_lab_banner, download_csv_button
from utils.metrics import compute_clustering_metrics
from utils.visualizations import plot_pca_2d, plot_bar, DARK_LAYOUT
import plotly.express as px
import plotly.graph_objects as go

def render_clustering():
    render_lab_banner(
        experiment_num=9,
        title="PCA and Cluster Analysis on Borrowing Behaviour",
        description="Unsupervised pattern recognition: orthogonal projection via Principal Component Analysis (PCA) and segmentation of student borrowing cohorts via K-Means, Hierarchical Clustering, and DBSCAN.",
        tags=["Unsupervised Learning", "PCA Scree Plot", "K-Means", "Elbow Method", "Silhouette Score", "Hierarchical Dendrogram", "DBSCAN"]
    )

    master_df = get_master_dataset()
    students_df = get_all_students()

    if master_df.empty or students_df.empty:
        st.warning("Data not available. Please seed database.")
        return

    # Student Behaviour Profile Aggregation
    student_profiles = master_df.groupby("student_id").agg(
        total_borrowed=("transaction_id", "count"),
        total_overdue=("overdue_status", "sum"),
        avg_overdue_days=("overdue_days", "mean"),
        avg_loan_duration=("loan_duration_days", "mean"),
        unique_categories_borrowed=("book_category", "nunique"),
        student_year=("student_year", "first"),
        student_department=("student_department", "first"),
        student_name=("student_name", "first")
    ).reset_index()

    student_profiles["overdue_ratio"] = (student_profiles["total_overdue"] / student_profiles["total_borrowed"]).round(3)

    clustering_features = [
        "total_borrowed", 
        "avg_overdue_days", 
        "avg_loan_duration", 
        "unique_categories_borrowed", 
        "overdue_ratio"
    ]

    X_raw = student_profiles[clustering_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Principal Component Analysis (PCA)",
        "🎯 K-Means Student Segmentation",
        "🌳 Hierarchical & DBSCAN",
        "👤 Discovered Persona Profiles"
    ])

    # TAB 1: Principal Component Analysis (PCA)
    with tab1:
        st.subheader("Dimensionality Reduction via PCA")
        st.markdown(r"""
        PCA transforms correlated variables into orthogonal, uncorrelated Principal Components ordered by the variance they explain:
        $$Z = X W \quad \text{where } W \text{ contains eigenvectors of } X^T X$$
        """)

        pca_full = PCA()
        pca_full.fit(X_scaled)
        exp_var = pca_full.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            # Scree Plot
            scree_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(len(exp_var))],
                "Individual Explained Variance": exp_var.round(4),
                "Cumulative Explained Variance": cum_var.round(4)
            })
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(x=scree_df["Component"], y=scree_df["Individual Explained Variance"], name="Individual Variance", marker_color="#6366F1"))
            fig_scree.add_trace(go.Scatter(x=scree_df["Component"], y=scree_df["Cumulative Explained Variance"], name="Cumulative Variance", mode="lines+markers", line=dict(color="#10B981", width=3)))
            fig_scree.update_layout(**DARK_LAYOUT, title="PCA Scree Plot (Explained Variance Ratio)")
            st.plotly_chart(fig_scree, use_container_width=True)

        with col_p2:
            st.markdown("##### Principal Component Loadings (Eigenvector Weights)")
            loadings_df = pd.DataFrame(
                pca_full.components_[:3, :].T,
                index=clustering_features,
                columns=["PC1 Weight", "PC2 Weight", "PC3 Weight"]
            ).round(3)
            st.dataframe(loadings_df, use_container_width=True)
            st.caption(f"PC1 + PC2 capture **{cum_var[1]*100:.1f}%** of total statistical variance.")

        # 2D PCA Scatter
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        student_profiles["PC1"] = X_pca_2d[:, 0]
        student_profiles["PC2"] = X_pca_2d[:, 1]

        fig_pca_proj = plot_pca_2d(student_profiles, "PC1", "PC2", color_col="student_department", title="2D PCA Space Colored by Student Department")
        st.plotly_chart(fig_pca_proj, use_container_width=True)

    # TAB 2: K-Means Student Segmentation
    with tab2:
        st.subheader("K-Means Clustering: Optimal k Selection & Partitioning")

        # Optimal k calculation
        k_values = range(2, 9)
        inertias = []
        silhouettes = []
        for k in k_values:
            km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = km_temp.fit_predict(X_scaled)
            metrics_temp = compute_clustering_metrics(X_scaled, labels_temp)
            inertias.append(km_temp.inertia_)
            silhouettes.append(metrics_temp["Silhouette_Score"])

        col_k_opt1, col_k_opt2 = st.columns(2)
        with col_k_opt1:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=list(k_values), y=inertias, mode="lines+markers", line=dict(color="#EC4899", width=3)))
            fig_elbow.update_layout(**DARK_LAYOUT, title="Elbow Method (Inertia vs. k)", xaxis_title="Number of Clusters (k)", yaxis_title="Sum of Squared Errors (Inertia)")
            st.plotly_chart(fig_elbow, use_container_width=True)

        with col_k_opt2:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(x=list(k_values), y=silhouettes, mode="lines+markers", line=dict(color="#10B981", width=3)))
            fig_sil.update_layout(**DARK_LAYOUT, title="Silhouette Score vs. k (Peak Indicates Optimal Quality)", xaxis_title="k", yaxis_title="Silhouette Score")
            st.plotly_chart(fig_sil, use_container_width=True)

        selected_k = st.slider("Select Number of Clusters (k)", min_value=2, max_value=6, value=3)

        kmeans_final = KMeans(n_clusters=selected_k, random_state=42, n_init=15)
        cluster_labels = kmeans_final.fit_predict(X_scaled)
        student_profiles["KMeans_Cluster"] = [f"Cluster {c}" for c in cluster_labels]

        k_eval = compute_clustering_metrics(X_scaled, cluster_labels)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Silhouette Score (Quality)", f"{k_eval['Silhouette_Score']:.4f}")
        with c2:
            st.metric("Davies-Bouldin Index (Separation)", f"{k_eval['Davies_Bouldin_Index']:.4f}")
        with c3:
            st.metric("Calinski-Harabasz Score", f"{k_eval['Calinski_Harabasz']:.1f}")

        # Cluster projection on PCA
        fig_cluster_pca = px.scatter(
            student_profiles,
            x="PC1",
            y="PC2",
            color="KMeans_Cluster",
            hover_data=["student_name", "total_borrowed", "avg_overdue_days", "overdue_ratio"],
            title=f"K-Means (k={selected_k}) Clusters Projected onto 2D PCA Space"
        )
        fig_cluster_pca.update_layout(**DARK_LAYOUT)
        fig_cluster_pca.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="#ffffff")))
        st.plotly_chart(fig_cluster_pca, use_container_width=True)

    # TAB 3: Hierarchical & DBSCAN
    with tab3:
        st.subheader("Alternative Clustering Paradigms")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("##### 1. Hierarchical Agglomerative Clustering (Dendrogram)")
            linkage_matrix = linkage(X_scaled[:30], method="ward") # Subsample for dendrogram readability
            fig_dendro, ax_d = plt.subplots(figsize=(10, 5), dpi=100)
            dendrogram(linkage_matrix, labels=student_profiles["student_name"].iloc[:30].tolist(), leaf_rotation=90, ax=ax_d)
            ax_d.set_title("Agglomerative Hierarchical Dendrogram (Ward's Linkage)", fontsize=11)
            ax_d.set_ylabel("Euclidean Distance")
            plt.tight_layout()
            st.pyplot(fig_dendro)
            plt.close(fig_dendro)

        with col_h2:
            st.markdown("##### 2. Density-Based Clustering (DBSCAN)")
            eps_val = st.slider("DBSCAN Epsilon (eps)", min_value=0.5, max_value=3.0, value=1.4, step=0.1)
            min_samples_val = st.slider("Min Samples", min_value=2, max_value=8, value=3)

            dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
            db_labels = dbscan.fit_predict(X_scaled)
            student_profiles["DBSCAN_Cluster"] = [f"Noise (-1)" if l == -1 else f"Density Group {l}" for l in db_labels]

            noise_count = np.sum(db_labels == -1)
            st.metric("DBSCAN Detected Outliers (Noise)", noise_count)

            fig_db = px.scatter(
                student_profiles,
                x="PC1",
                y="PC2",
                color="DBSCAN_Cluster",
                title=f"DBSCAN Clustering Projection (eps={eps_val})"
            )
            fig_db.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_db, use_container_width=True)

    # TAB 4: Discovered Persona Profiles
    with tab4:
        st.subheader("Actionable Student Cohort Archetypes")
        st.write("Examine the mean behavioral attributes of each discovered K-Means cluster to extract operational library personas.")

        cluster_summary = student_profiles.groupby("KMeans_Cluster")[clustering_features].mean().round(2).reset_index()

        st.dataframe(cluster_summary, use_container_width=True)

        st.markdown("""
        #### Empirical Persona Interpretations:
        - 🌟 **Diligent Avid Scholars (High Borrowing, Low Overdue):** Students who borrow extensively across varied domains and consistently return titles within the 14-day loan window.
        - ⚠️ **High-Risk Overdue Group (Moderate Borrowing, High Overdue Days):** Students whose loans frequently drift past due dates; prime candidates for proactive SMS/Email return reminders.
        - 📚 **Occasional Syllabus Borrowers (Low Borrowing, Standard Duration):** Students who borrow only 1-3 times per semester, primarily around mid-term/final exams.

        > [!IMPORTANT]
        > **Ethical Principle:** These clusters are purely descriptive and exploratory. They must NEVER be used to penalize, restrict borrowing access, or stigmatize students.
        """)
        download_csv_button(student_profiles, "slms_student_clusters.csv", "📥 Export Clustered Student Profiles (CSV)")
