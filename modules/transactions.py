"""
Book Circulation Module for SLMS.
Handles issuing books, recording returns, calculating overdue periods, and monitoring active loans.
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
from database.db import (
    get_all_books,
    get_all_students,
    get_all_transactions,
    issue_book,
    return_book
)
from utils.helpers import download_csv_button

def render_transactions():
    st.title("🔄 Book Circulation (Issue & Return)")
    st.caption("Manage book checkout and checkin workflows, calculate overdue fines/days, and view circulation ledgers.")

    tab1, tab2, tab3, tab4 = st.tabs(["📤 Issue Book", "📥 Return Book", "⏳ Active Loans & Overdue", "📜 Circulation Ledger"])

    books_df = get_all_books()
    students_df = get_all_students()
    txns_df = get_all_transactions()

    # TAB 1: Issue Book
    with tab1:
        st.subheader("Issue a Book to a Student")
        if books_df.empty or students_df.empty:
            st.warning("Ensure both books and students are registered in the system.")
            return

        # Filter only books that have available copies > 0
        available_books = books_df[books_df["available_copies"] > 0]
        if available_books.empty:
            st.error("No books currently have available copies in stock.")
            return

        with st.form("issue_book_form"):
            col1, col2 = st.columns(2)
            with col1:
                student_options = {f"{r['student_id']} - {r['name']} ({r['department']}, Yr {r['year_of_study']})": r['student_id'] for _, r in students_df.iterrows()}
                selected_student = st.selectbox("Select Student*", list(student_options.keys()))
                student_id = student_options[selected_student]

                issue_d = st.date_input("Issue Date*", value=date.today())
                notes = st.text_input("Circulation Notes", placeholder="e.g. Standard 14-day student checkout")

            with col2:
                book_options = {f"{r['book_id']} - {r['title']} [Avail: {r['available_copies']}/{r['total_copies']}] ({r['category']})": r['book_id'] for _, r in available_books.iterrows()}
                selected_book = st.selectbox("Select Book*", list(book_options.keys()))
                book_id = book_options[selected_book]

                due_d = st.date_input("Due Date*", value=date.today() + timedelta(days=14))

            submit_issue = st.form_submit_button("📤 Confirm Book Issue", use_container_width=True)
            if submit_issue:
                if due_d <= issue_d:
                    st.error("Due Date must be strictly after the Issue Date.")
                else:
                    success, msg = issue_book(
                        student_id=student_id,
                        book_id=book_id,
                        issue_date_str=issue_d.strftime("%Y-%m-%d"),
                        due_date_str=due_d.strftime("%Y-%m-%d"),
                        notes=notes
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # TAB 2: Return Book
    with tab2:
        st.subheader("Process Book Return")
        # Find active transactions
        active_txns = txns_df[txns_df["return_date"].isna()]
        if active_txns.empty:
            st.info("There are currently no active issued books awaiting return.")
        else:
            active_options = {
                f"{r['transaction_id']} | {r['book_title']} (Issued to: {r['student_name']} | Due: {r['due_date']})": r['transaction_id'] 
                for _, r in active_txns.iterrows()
            }
            selected_active = st.selectbox("Select Active Transaction to Check-In", list(active_options.keys()))
            selected_txn_id = active_options[selected_active]
            
            # Retrieve transaction row
            txn_row = active_txns[active_txns["transaction_id"] == selected_txn_id].iloc[0]
            due_date_obj = datetime.strptime(txn_row["due_date"], "%Y-%m-%d").date()

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                return_d = st.date_input("Return Date", value=date.today())
                overdue_calc = max(0, (return_d - due_date_obj).days)
                if overdue_calc > 0:
                    st.warning(f"⚠️ Overdue Warning: Return date is **{overdue_calc} days** past the due date ({txn_row['due_date']}).")
                else:
                    st.success(f"✅ On-Time Return: Book returned on or before due date ({txn_row['due_date']}).")

            with col_r2:
                st.write("")
                st.write("")
                if st.button("📥 Confirm Book Return", type="primary", use_container_width=True):
                    success, msg, days = return_book(selected_txn_id, return_d.strftime("%Y-%m-%d"))
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # TAB 3: Active Loans & Overdue Tracker
    with tab3:
        st.subheader("Active Loans & Real-Time Overdue Tracker")
        active_txns = txns_df[txns_df["return_date"].isna()].copy()
        if active_txns.empty:
            st.info("No books are currently checked out.")
        else:
            today_str = date.today().strftime("%Y-%m-%d")
            active_txns["current_status"] = active_txns["due_date"].apply(
                lambda d: "OVERDUE" if d < today_str else "ACTIVE"
            )
            
            overdue_count = len(active_txns[active_txns["current_status"] == "OVERDUE"])
            
            k1, k2 = st.columns(2)
            with k1:
                st.metric("Total Active Loans", len(active_txns))
            with k2:
                st.metric("Currently Overdue Active Loans", overdue_count)

            display_cols = ["transaction_id", "book_id", "book_title", "student_id", "student_name", "student_department", "issue_date", "due_date", "current_status"]
            st.dataframe(
                active_txns[[c for c in display_cols if c in active_txns.columns]],
                use_container_width=True,
                hide_index=True
            )

    # TAB 4: Circulation Ledger
    with tab4:
        st.subheader("All Transactions Ledger")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            status_filter = st.selectbox("Filter by Status", ["All Records", "Active Only", "Returned On-Time", "Returned Overdue"])
        with col_f2:
            search_txn = st.text_input("Search by Student, Book, or Transaction ID", placeholder="e.g. TXN-5001...")

        filtered_ledger = txns_df.copy()
        if status_filter == "Active Only":
            filtered_ledger = filtered_ledger[filtered_ledger["return_date"].isna()]
        elif status_filter == "Returned On-Time":
            filtered_ledger = filtered_ledger[(filtered_ledger["return_date"].notna()) & (filtered_ledger["overdue_status"] == 0)]
        elif status_filter == "Returned Overdue":
            filtered_ledger = filtered_ledger[(filtered_ledger["return_date"].notna()) & (filtered_ledger["overdue_status"] == 1)]

        if search_txn:
            q = search_txn.lower()
            filtered_ledger = filtered_ledger[
                filtered_ledger["transaction_id"].str.lower().str.contains(q, na=False) |
                filtered_ledger["student_name"].str.lower().str.contains(q, na=False) |
                filtered_ledger["book_title"].str.lower().str.contains(q, na=False) |
                filtered_ledger["student_id"].str.lower().str.contains(q, na=False) |
                filtered_ledger["book_id"].str.lower().str.contains(q, na=False)
            ]

        st.dataframe(
            filtered_ledger[[
                "transaction_id", "student_id", "student_name", "book_id", "book_title", 
                "issue_date", "due_date", "return_date", "overdue_days", "overdue_status", "loan_duration_days"
            ]],
            use_container_width=True,
            hide_index=True
        )
        download_csv_button(filtered_ledger, "slms_circulation_ledger.csv", "📥 Export Circulation Ledger (CSV)")
