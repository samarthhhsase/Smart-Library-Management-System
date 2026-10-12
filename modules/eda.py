"""
Experiment 1: Exploratory Data Analysis and Data Visualization.
Analyzes library inventory, borrowing patterns, categories, seasonality, and correlations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from database.db import get_all_books, get_all_students, get_all_transactions, get_master_dataset
from utils.helpers import render_lab_banner, download_csv_button
from utils.visualizations import (
    plot_bar,
    plot_pie,
    plot_histogram,
    plot_box,
    plot_correlation_heatmap,
    plot_time_series
)

def render_eda():
    render_lab_banner(
        experiment_num=1,
        title="Exploratory Data Analysis and Data Visualization",
        description="Comprehensive exploratory analysis of book inventories, student circulation logs, category preferences, and temporal borrowing trends.",
        tags=["Descriptive Analytics", "Visualizations", "Correlation Analysis", "Summary Statistics"]
    )

    books_df = get_all_books()
    students_df = get_all_students()
    txns_df = get_all_transactions()
    master_df = get_master_dataset()

    if master_df.empty:
        st.warning("No transaction data available for analysis. Please check data generator.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Inventory & Categories", 
        "📈 Borrowing Trends & Popularity", 
        "📦 Distribution & Box Plots", 
        "🔍 Dynamic Dataset Explorer"
    ])

    # TAB 1: Inventory & Categories
    with tab1:
        st.subheader("Book Inventory and Category Composition")
        col1, col2 = st.columns(2)
        
        with col1:
            cat_counts = books_df["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Book Titles Count"]
            fig_cat = plot_bar(cat_counts, x="Category", y="Book Titles Count", title="Catalog Distribution Across Academic Domains", color="Category")
            st.plotly_chart(fig_cat, use_container_width=True)

        with col2:
            copies_by_cat = books_df.groupby("category")[["total_copies", "available_copies"]].sum().reset_index()
            fig_pie = plot_pie(copies_by_cat, names="category", values="total_copies", title="Total Physical Copies Share by Domain")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Department distribution of students
        col3, col4 = st.columns(2)
        with col3:
            dept_counts = students_df["department"].value_counts().reset_index()
            dept_counts.columns = ["Department", "Enrolled Students"]
            fig_dept = plot_bar(dept_counts, x="Department", y="Enrolled Students", title="Registered Students per Department", color="Department")
            st.plotly_chart(fig_dept, use_container_width=True)

        with col4:
            yr_counts = students_df["year_of_study"].value_counts().sort_index().reset_index()
            yr_counts.columns = ["Year of Study", "Student Count"]
            yr_counts["Year of Study"] = "Year " + yr_counts["Year of Study"].astype(str)
            fig_yr = plot_pie(yr_counts, names="Year of Study", values="Student Count", title="Student Body by Academic Year")
            st.plotly_chart(fig_yr, use_container_width=True)

    # TAB 2: Borrowing Trends & Popularity
    with tab2:
        st.subheader("Popular Titles and Peak Circulation Periods")
        
        # Top 10 most popular books
        top_books = txns_df["book_title"].value_counts().head(10).reset_index()
        top_books.columns = ["Book Title", "Times Borrowed"]
        fig_top = plot_bar(top_books, x="Times Borrowed", y="Book Title", orientation="h", title="Top 10 Most Frequently Borrowed Books")
        st.plotly_chart(fig_top, use_container_width=True)

        # Category borrowing demand
        cat_demand = txns_df["book_category"].value_counts().reset_index()
        cat_demand.columns = ["Domain Category", "Total Issues"]
        fig_cat_demand = plot_bar(cat_demand, x="Domain Category", y="Total Issues", title="Total Borrowing Transactions by Category", color="Domain Category")
        st.plotly_chart(fig_cat_demand, use_container_width=True)

        # Monthly circulation timeline
        master_df["issue_year_month"] = master_df["issue_date_dt"].dt.to_period("M").astype(str)
        monthly_trend = master_df.groupby("issue_year_month")["transaction_id"].count().reset_index()
        monthly_trend.columns = ["Year-Month", "Circulation Count"]
        fig_trend = plot_time_series(monthly_trend, x="Year-Month", y="Circulation Count", title="Monthly Borrowing Volume Timeline (Exam & Semester Peaks)")
        st.plotly_chart(fig_trend, use_container_width=True)

        st.info("💡 **Observation:** Circulation peaks visibly during October/November and March/April, directly corresponding with university mid-term and end-semester examination periods.")

    # TAB 3: Distributions & Box Plots
    with tab3:
        st.subheader("Statistical Distributions and Correlation Analysis")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_hist = plot_histogram(master_df, x="overdue_days", title="Distribution of Overdue Days (Right-Skewed)", nbins=25)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_d2:
            fig_dur = plot_histogram(master_df, x="loan_duration_days", title="Loan Duration Distribution (Days)", nbins=20)
            st.plotly_chart(fig_dur, use_container_width=True)

        # Boxplots
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_box_dept = plot_box(master_df, x="student_department", y="overdue_days", title="Overdue Days Spread Across Departments")
            st.plotly_chart(fig_box_dept, use_container_width=True)
        with col_b2:
            fig_box_cat = plot_box(master_df, x="book_category", y="loan_duration_days", title="Loan Duration Across Subject Categories")
            st.plotly_chart(fig_box_cat, use_container_width=True)

        # Correlation Heatmap
        st.markdown("#### Pearson Correlation Matrix")
        numeric_cols = ["borrowing_frequency", "overdue_days", "loan_duration_days", "book_pub_year", "book_total_copies", "book_age", "overdue_status"]
        fig_corr = plot_correlation_heatmap(master_df, numeric_cols, title="Correlation Heatmap of Numerical Features")
        st.plotly_chart(fig_corr, use_container_width=True)

    # TAB 4: Dynamic Dataset Explorer
    with tab4:
        st.subheader("Dynamic Attribute Inspector")
        dataset_choice = st.selectbox("Select Target Dataset for Exploration", ["Master Joined Dataset", "Books Inventory", "Students Directory", "Transactions Ledger"])
        
        target_df = {
            "Master Joined Dataset": master_df,
            "Books Inventory": books_df,
            "Students Directory": students_df,
            "Transactions Ledger": txns_df
        }[dataset_choice]

        st.markdown("##### Summary Statistics")
        st.dataframe(target_df.describe(include="all").T, use_container_width=True)

        st.markdown("##### Raw Dataset Preview")
        st.dataframe(target_df.head(50), use_container_width=True)
        download_csv_button(target_df, f"slms_dataset_{dataset_choice.lower().replace(' ', '_')}.csv")
