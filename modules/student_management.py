"""
Student Management Module for SLMS.
Provides registration, profile inspection, department filtering, and detailed borrowing history.
"""

import streamlit as st
import pandas as pd
from database.db import (
    get_all_students,
    add_student,
    update_student,
    delete_student,
    get_student_by_id,
    get_all_transactions
)
from utils.helpers import download_csv_button

DEPARTMENTS = [
    "Computer Science",
    "Information Technology",
    "Electronics & Comm.",
    "Mechanical Eng.",
    "Civil Eng."
]

def render_student_management():
    st.title("👨‍🎓 Student Management")
    st.caption("Register students, manage academic profiles, and monitor individual reading histories.")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Directory", "➕ Register Student", "👤 Student Profile & History", "✏️ Update / Delete"])

    students_df = get_all_students()
    txns_df = get_all_transactions()

    # TAB 1: Directory
    with tab1:
        st.subheader("Student Directory")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            search_query = st.text_input("Search Student by Name or ID", placeholder="e.g. Aarav, STU-2001...")
        with c2:
            dept_filter = st.selectbox("Department", ["All Departments"] + DEPARTMENTS)
        with c3:
            year_filter = st.selectbox("Year of Study", ["All Years", 1, 2, 3, 4])

        filtered_df = students_df.copy()
        if search_query:
            q = search_query.lower()
            filtered_df = filtered_df[
                filtered_df["name"].str.lower().str.contains(q, na=False) |
                filtered_df["student_id"].str.lower().str.contains(q, na=False) |
                filtered_df["email"].str.lower().str.contains(q, na=False)
            ]
        if dept_filter != "All Departments":
            filtered_df = filtered_df[filtered_df["department"] == dept_filter]
        if year_filter != "All Years":
            filtered_df = filtered_df[filtered_df["year_of_study"] == int(year_filter)]

        st.dataframe(
            filtered_df[["student_id", "name", "department", "year_of_study", "email", "contact"]],
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Showing {len(filtered_df)} of {len(students_df)} students.")
        download_csv_button(filtered_df, "slms_students_directory.csv", "📥 Export Students Directory (CSV)")

    # TAB 2: Register Student
    with tab2:
        st.subheader("Register a New Student")
        with st.form("register_student_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name*", placeholder="e.g. Ananya Sen")
                email = st.text_input("Email*", placeholder="e.g. ananya.sen@college.edu")
                contact = st.text_input("Phone Number", placeholder="e.g. +91 98765 43210")
            with col2:
                dept = st.selectbox("Department*", DEPARTMENTS)
                year = st.selectbox("Year of Study*", [1, 2, 3, 4], index=2)

            submitted = st.form_submit_button("➕ Register Student", use_container_width=True)
            if submitted:
                if not name.strip() or not email.strip():
                    st.error("Name and Email are required.")
                else:
                    new_id = add_student(
                        name=name,
                        email=email,
                        department=dept,
                        year_of_study=year,
                        contact=contact
                    )
                    st.success(f"Student successfully registered! Assigned Student ID: **{new_id}**")
                    st.rerun()

    # TAB 3: Student Profile & History
    with tab3:
        st.subheader("Student Academic Profile & Loan History")
        if students_df.empty:
            st.info("No students found in database.")
            return

        student_options = {f"{row['student_id']} - {row['name']} ({row['department']})": row['student_id'] for _, row in students_df.iterrows()}
        selected_label = st.selectbox("Select Student", list(student_options.keys()))
        selected_id = student_options[selected_label]
        student_data = get_student_by_id(selected_id)

        if student_data:
            # Calculate student stats
            stu_txns = txns_df[txns_df["student_id"] == selected_id] if not txns_df.empty else pd.DataFrame()
            total_borrowed = len(stu_txns)
            active_loans = len(stu_txns[stu_txns["return_date"].isna()]) if not stu_txns.empty else 0
            overdue_count = len(stu_txns[stu_txns["overdue_status"] == 1]) if not stu_txns.empty else 0
            overdue_rate = (overdue_count / total_borrowed * 100) if total_borrowed > 0 else 0.0

            # Render summary cards
            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            with col_k1:
                st.metric("Total Books Borrowed", total_borrowed)
            with col_k2:
                st.metric("Currently Active Loans", active_loans)
            with col_k3:
                st.metric("Overdue Incidents", overdue_count)
            with col_k4:
                st.metric("Historical Overdue Rate", f"{overdue_rate:.1f}%")

            st.markdown("#### Borrowing History")
            if stu_txns.empty:
                st.info("No borrowing history on record for this student.")
            else:
                display_cols = ["transaction_id", "book_title", "book_category", "issue_date", "due_date", "return_date", "overdue_days", "overdue_status"]
                st.dataframe(stu_txns[[c for c in display_cols if c in stu_txns.columns]], use_container_width=True, hide_index=True)

    # TAB 4: Update / Delete
    with tab4:
        st.subheader("Modify Student Record")
        if students_df.empty:
            st.info("No students found.")
            return

        selected_label_mod = st.selectbox("Select Student to Modify", list(student_options.keys()), key="mod_stu_select")
        selected_id_mod = student_options[selected_label_mod]
        stu_to_mod = get_student_by_id(selected_id_mod)

        if stu_to_mod:
            col_m1, col_m2 = st.columns([3, 1])
            with col_m1:
                with st.form(f"mod_stu_{selected_id_mod}"):
                    m_name = st.text_input("Name", value=stu_to_mod["name"])
                    m_email = st.text_input("Email", value=stu_to_mod["email"] or "")
                    m_dept_idx = DEPARTMENTS.index(stu_to_mod["department"]) if stu_to_mod["department"] in DEPARTMENTS else 0
                    m_dept = st.selectbox("Department", DEPARTMENTS, index=m_dept_idx)
                    m_yr = st.selectbox("Year of Study", [1, 2, 3, 4], index=int(stu_to_mod["year_of_study"]) - 1)
                    m_contact = st.text_input("Contact", value=stu_to_mod["contact"] or "")

                    save_btn = st.form_submit_button("💾 Update Details", use_container_width=True)
                    if save_btn:
                        update_student(selected_id_mod, m_name, m_email, m_dept, m_yr, m_contact)
                        st.success("Student details updated successfully.")
                        st.rerun()

            with col_m2:
                st.markdown("#### Danger Zone")
                st.warning("Deleting this student will remove their records.")
                if st.button("🗑️ Delete Student", type="primary", use_container_width=True):
                    delete_student(selected_id_mod)
                    st.success("Student deleted successfully.")
                    st.rerun()
