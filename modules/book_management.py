"""
Book Management Module for SLMS.
Allows adding, updating, deleting, searching, and filtering book records.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from database.db import (
    get_all_books,
    add_book,
    update_book,
    delete_book,
    get_book_by_id
)
from utils.helpers import download_csv_button

CATEGORIES = [
    "Computer Science",
    "Data Science & AI",
    "Electronics & Comm.",
    "Mechanical Eng.",
    "Civil Eng.",
    "Mathematics & Statistics",
    "Management & Soft Skills"
]

def render_book_management():
    st.title("📚 Book Management")
    st.caption("Manage catalog inventory, register new titles, and track physical copy availability.")

    tab1, tab2, tab3 = st.tabs(["🔍 Search & Inventory", "➕ Add New Book", "✏️ Edit / Delete Book"])

    books_df = get_all_books()

    # TAB 1: Search & Inventory
    with tab1:
        st.subheader("Book Catalog")
        col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
        with col_s1:
            search_query = st.text_input("Search by Title, Author, or ISBN", placeholder="e.g. Algorithms, Cormen, 978...")
        with col_s2:
            cat_filter = st.selectbox("Category Filter", ["All Categories"] + CATEGORIES)
        with col_s3:
            avail_filter = st.checkbox("Only show available books", value=False)

        filtered_df = books_df.copy()
        if search_query:
            q = search_query.lower()
            filtered_df = filtered_df[
                filtered_df["title"].str.lower().str.contains(q, na=False) |
                filtered_df["author"].str.lower().str.contains(q, na=False) |
                filtered_df["isbn"].str.lower().str.contains(q, na=False) |
                filtered_df["book_id"].str.lower().str.contains(q, na=False)
            ]
        if cat_filter != "All Categories":
            filtered_df = filtered_df[filtered_df["category"] == cat_filter]
        if avail_filter:
            filtered_df = filtered_df[filtered_df["available_copies"] > 0]

        st.dataframe(
            filtered_df[[
                "book_id", "title", "author", "category", "publication_year", 
                "total_copies", "available_copies", "shelf_location", "isbn"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Showing {len(filtered_df)} of {len(books_df)} books.")
        download_csv_button(filtered_df, "slms_books_inventory.csv", "📥 Export Books Inventory (CSV)")

    # TAB 2: Add New Book
    with tab2:
        st.subheader("Add a New Book to the Library")
        with st.form("add_book_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Book Title*", placeholder="e.g. Modern Information Retrieval")
                author = st.text_input("Author(s)*", placeholder="e.g. Ricardo Baeza-Yates")
                category = st.selectbox("Category*", CATEGORIES)
                isbn = st.text_input("ISBN", placeholder="e.g. 978-0131103627")
            with col2:
                pub_year = st.number_input("Publication Year*", min_value=1950, max_value=datetime.now().year + 1, value=2021)
                total_copies = st.number_input("Total Copies*", min_value=1, max_value=100, value=3)
                shelf_location = st.text_input("Shelf Location*", value="CS-1A")

            submitted = st.form_submit_button("➕ Register Book", use_container_width=True)
            if submitted:
                if not title.strip() or not author.strip():
                    st.error("Title and Author are required fields.")
                else:
                    new_id = add_book(
                        title=title,
                        author=author,
                        category=category,
                        isbn=isbn if isbn.strip() else f"AUTO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        publication_year=int(pub_year),
                        total_copies=int(total_copies),
                        shelf_location=shelf_location
                    )
                    st.success(f"Book successfully registered! Assigned Book ID: **{new_id}**")
                    st.rerun()

    # TAB 3: Edit / Delete Book
    with tab3:
        st.subheader("Modify Existing Book Record")
        if books_df.empty:
            st.info("No books found in the database.")
            return

        book_options = {f"{row['book_id']} - {row['title']} ({row['author']})": row['book_id'] for _, row in books_df.iterrows()}
        selected_label = st.selectbox("Select Book to Edit or Delete", list(book_options.keys()))
        selected_id = book_options[selected_label]
        book_data = get_book_by_id(selected_id)

        if book_data:
            col_e1, col_e2 = st.columns([3, 1])
            with col_e1:
                with st.form(f"edit_book_{selected_id}"):
                    e_title = st.text_input("Title", value=book_data["title"])
                    e_author = st.text_input("Author", value=book_data["author"])
                    cat_idx = CATEGORIES.index(book_data["category"]) if book_data["category"] in CATEGORIES else 0
                    e_cat = st.selectbox("Category", CATEGORIES, index=cat_idx)
                    e_isbn = st.text_input("ISBN", value=book_data["isbn"] or "")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        e_pub = st.number_input("Publication Year", min_value=1900, max_value=datetime.now().year + 1, value=int(book_data["publication_year"] or 2020))
                    with c2:
                        e_total = st.number_input("Total Copies", min_value=1, max_value=200, value=int(book_data["total_copies"]))
                    with c3:
                        e_avail = st.number_input("Available Copies", min_value=0, max_value=e_total, value=min(int(book_data["available_copies"]), e_total))

                    e_shelf = st.text_input("Shelf Location", value=book_data["shelf_location"] or "A1")

                    update_btn = st.form_submit_button("💾 Save Changes", use_container_width=True)
                    if update_btn:
                        update_book(
                            book_id=selected_id,
                            title=e_title,
                            author=e_author,
                            category=e_cat,
                            isbn=e_isbn,
                            publication_year=int(e_pub),
                            total_copies=int(e_total),
                            available_copies=int(e_avail),
                            shelf_location=e_shelf
                        )
                        st.success(f"Book {selected_id} updated successfully!")
                        st.rerun()

            with col_e2:
                st.markdown("#### Danger Zone")
                st.warning("Deleting this book will permanently remove it.")
                if st.button("🗑️ Delete Book", type="primary", use_container_width=True):
                    delete_book(selected_id)
                    st.success(f"Book {selected_id} deleted successfully.")
                    st.rerun()
