"""
Database operations module for Smart Library Management System (SLMS).
Uses SQLite with thread-safe connections and helper methods for analytics and ML.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), "library.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if tables do not exist."""
    with get_connection() as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()

def reset_database():
    """Drops and recreates all tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS transactions;")
        cursor.execute("DROP TABLE IF EXISTS students;")
        cursor.execute("DROP TABLE IF EXISTS books;")
        conn.commit()
    init_db()

# ==========================================
# BOOK OPERATIONS
# ==========================================

def get_all_books() -> pd.DataFrame:
    """Fetch all books as a pandas DataFrame."""
    init_db()
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM books ORDER BY book_id ASC", conn)
        return df
    finally:
        conn.close()

def get_book_by_id(book_id: str):
    """Fetch a single book by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM books WHERE book_id = ?", (book_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def generate_next_book_id() -> str:
    """Generates the next sequential book ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT book_id FROM books WHERE book_id LIKE 'BK-%' ORDER BY CAST(SUBSTR(book_id, 4) AS INTEGER) DESC LIMIT 1")
        row = cursor.fetchone()
        if row and row[0]:
            try:
                num = int(row[0].replace("BK-", "")) + 1
                return f"BK-{num:04d}"
            except ValueError:
                pass
        return "BK-1001"

def add_book(title: str, author: str, category: str, isbn: str, publication_year: int, total_copies: int, shelf_location: str = "A1") -> str:
    """Adds a new book record and returns the assigned book_id."""
    book_id = generate_next_book_id()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO books (book_id, title, author, category, isbn, publication_year, total_copies, available_copies, shelf_location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (book_id, title.strip(), author.strip(), category.strip(), isbn.strip(), int(publication_year), int(total_copies), int(total_copies), shelf_location.strip())
        )
        conn.commit()
    return book_id

def update_book(book_id: str, title: str, author: str, category: str, isbn: str, publication_year: int, total_copies: int, available_copies: int, shelf_location: str):
    """Updates an existing book record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE books
            SET title = ?, author = ?, category = ?, isbn = ?, publication_year = ?, total_copies = ?, available_copies = ?, shelf_location = ?
            WHERE book_id = ?
            """,
            (title.strip(), author.strip(), category.strip(), isbn.strip(), int(publication_year), int(total_copies), int(available_copies), shelf_location.strip(), book_id)
        )
        conn.commit()

def delete_book(book_id: str):
    """Deletes a book record and associated records."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM books WHERE book_id = ?", (book_id,))
        conn.commit()

# ==========================================
# STUDENT OPERATIONS
# ==========================================

def get_all_students() -> pd.DataFrame:
    """Fetch all students as a pandas DataFrame."""
    init_db()
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM students ORDER BY student_id ASC", conn)
        return df
    finally:
        conn.close()

def get_student_by_id(student_id: str):
    """Fetch a single student by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def generate_next_student_id() -> str:
    """Generates the next sequential student ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id FROM students WHERE student_id LIKE 'STU-%' ORDER BY CAST(SUBSTR(student_id, 5) AS INTEGER) DESC LIMIT 1")
        row = cursor.fetchone()
        if row and row[0]:
            try:
                num = int(row[0].replace("STU-", "")) + 1
                return f"STU-{num:04d}"
            except ValueError:
                pass
        return "STU-2001"

def add_student(name: str, email: str, department: str, year_of_study: int, contact: str = "") -> str:
    """Adds a new student record and returns the assigned student_id."""
    student_id = generate_next_student_id()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO students (student_id, name, email, department, year_of_study, contact)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (student_id, name.strip(), email.strip(), department.strip(), int(year_of_study), contact.strip())
        )
        conn.commit()
    return student_id

def update_student(student_id: str, name: str, email: str, department: str, year_of_study: int, contact: str):
    """Updates an existing student record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE students
            SET name = ?, email = ?, department = ?, year_of_study = ?, contact = ?
            WHERE student_id = ?
            """,
            (name.strip(), email.strip(), department.strip(), int(year_of_study), contact.strip(), student_id)
        )
        conn.commit()

def delete_student(student_id: str):
    """Deletes a student record and cascades."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        conn.commit()

# ==========================================
# TRANSACTION OPERATIONS
# ==========================================

def get_all_transactions() -> pd.DataFrame:
    """Fetch all transactions with joined book title and student name."""
    init_db()
    conn = get_connection()
    try:
        query = """
            SELECT 
                t.*,
                b.title AS book_title,
                b.category AS book_category,
                s.name AS student_name,
                s.department AS student_department,
                s.year_of_study AS student_year
            FROM transactions t
            LEFT JOIN books b ON t.book_id = b.book_id
            LEFT JOIN students s ON t.student_id = s.student_id
            ORDER BY t.issue_date DESC
        """
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def generate_next_transaction_id() -> str:
    """Generates the next sequential transaction ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT transaction_id FROM transactions WHERE transaction_id LIKE 'TXN-%' ORDER BY CAST(SUBSTR(transaction_id, 5) AS INTEGER) DESC LIMIT 1")
        row = cursor.fetchone()
        if row and row[0]:
            try:
                num = int(row[0].replace("TXN-", "")) + 1
                return f"TXN-{num:04d}"
            except ValueError:
                pass
        return "TXN-5001"

def issue_book(student_id: str, book_id: str, issue_date_str: str, due_date_str: str, notes: str = "") -> tuple[bool, str]:
    """Issues a book to a student if copies are available."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Verify book availability
        cursor.execute("SELECT available_copies FROM books WHERE book_id = ?", (book_id,))
        book_row = cursor.fetchone()
        if not book_row:
            return False, "Book ID not found."
        if book_row["available_copies"] <= 0:
            return False, "No copies available for this book."

        # Verify student exists
        cursor.execute("SELECT student_id FROM students WHERE student_id = ?", (student_id,))
        student_row = cursor.fetchone()
        if not student_row:
            return False, "Student ID not found."

        # Compute historical borrowing frequency for this student
        cursor.execute("SELECT COUNT(*) AS cnt FROM transactions WHERE student_id = ?", (student_id,))
        freq_row = cursor.fetchone()
        borrowing_frequency = (freq_row["cnt"] if freq_row else 0) + 1

        txn_id = generate_next_transaction_id()

        # Parse loan duration
        d1 = datetime.strptime(issue_date_str, "%Y-%m-%d")
        d2 = datetime.strptime(due_date_str, "%Y-%m-%d")
        loan_duration_days = max(1, (d2 - d1).days)

        cursor.execute(
            """
            INSERT INTO transactions (
                transaction_id, student_id, book_id, issue_date, due_date, 
                return_date, borrowing_frequency, overdue_days, overdue_status, 
                loan_duration_days, notes
            )
            VALUES (?, ?, ?, ?, ?, NULL, ?, 0, 0, ?, ?)
            """,
            (txn_id, student_id, book_id, issue_date_str, due_date_str, borrowing_frequency, loan_duration_days, notes)
        )

        # Decrement available copies
        cursor.execute("UPDATE books SET available_copies = available_copies - 1 WHERE book_id = ?", (book_id,))
        conn.commit()
        return True, f"Book successfully issued! Transaction ID: {txn_id}"

def return_book(transaction_id: str, return_date_str: str) -> tuple[bool, str, int]:
    """Records a book return, computes overdue days and status, restores book copies."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,))
        txn = cursor.fetchone()
        if not txn:
            return False, "Transaction not found.", 0
        if txn["return_date"]:
            return False, "Book has already been returned.", 0

        due_date = datetime.strptime(txn["due_date"], "%Y-%m-%d").date()
        ret_date = datetime.strptime(return_date_str, "%Y-%m-%d").date()
        
        overdue_days = max(0, (ret_date - due_date).days)
        overdue_status = 1 if overdue_days > 0 else 0

        # Calculate actual loan duration
        issue_date = datetime.strptime(txn["issue_date"], "%Y-%m-%d").date()
        actual_duration = max(1, (ret_date - issue_date).days)

        cursor.execute(
            """
            UPDATE transactions
            SET return_date = ?, overdue_days = ?, overdue_status = ?, loan_duration_days = ?
            WHERE transaction_id = ?
            """,
            (return_date_str, overdue_days, overdue_status, actual_duration, transaction_id)
        )

        # Restore available copies
        cursor.execute("UPDATE books SET available_copies = available_copies + 1 WHERE book_id = ?", (txn["book_id"],))
        conn.commit()

        msg = f"Book returned successfully. Overdue Days: {overdue_days} ({'Overdue' if overdue_status == 1 else 'On Time'})."
        return True, msg, overdue_days

def get_dashboard_kpis() -> dict:
    """Computes headline KPIs for the SLMS main dashboard."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) AS total_books, SUM(total_copies) AS total_copies, SUM(available_copies) AS total_available FROM books")
        books_stat = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) AS total_students FROM students")
        students_stat = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) AS active_issued FROM transactions WHERE return_date IS NULL")
        active_stat = cursor.fetchone()

        # Check currently overdue active books (due_date < current date and not returned)
        today_str = date.today().strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) AS active_overdue FROM transactions WHERE return_date IS NULL AND due_date < ?", (today_str,))
        active_overdue_stat = cursor.fetchone()

        cursor.execute("SELECT COUNT(*) AS total_txns, SUM(CASE WHEN overdue_status = 1 THEN 1 ELSE 0 END) AS total_overdue_historical FROM transactions WHERE return_date IS NOT NULL")
        historical_stat = cursor.fetchone()

        return {
            "total_titles": books_stat["total_books"] or 0,
            "total_copies": books_stat["total_copies"] or 0,
            "available_copies": books_stat["total_available"] or 0,
            "total_students": students_stat["total_students"] or 0,
            "active_issued": active_stat["active_issued"] or 0,
            "active_overdue": active_overdue_stat["active_overdue"] or 0,
            "total_transactions": (historical_stat["total_txns"] or 0) + (active_stat["active_issued"] or 0),
            "historical_overdue_txns": historical_stat["total_overdue_historical"] or 0
        }

def get_master_dataset() -> pd.DataFrame:
    """
    Returns an enriched joined dataset suitable for EDA, Statistical Testing, and ML.
    Derives feature columns while strictly avoiding target leakage.
    """
    init_db()
    conn = get_connection()
    try:
        query = """
            SELECT 
                t.transaction_id,
                t.student_id,
                t.book_id,
                t.issue_date,
                t.due_date,
                t.return_date,
                t.borrowing_frequency,
                t.overdue_days,
                t.overdue_status,
                t.loan_duration_days,
                b.title AS book_title,
                b.author AS book_author,
                b.category AS book_category,
                b.publication_year AS book_pub_year,
                b.total_copies AS book_total_copies,
                b.available_copies AS book_available_copies,
                s.name AS student_name,
                s.department AS student_department,
                s.year_of_study AS student_year
            FROM transactions t
            INNER JOIN books b ON t.book_id = b.book_id
            INNER JOIN students s ON t.student_id = s.student_id
        """
        df = pd.read_sql_query(query, conn)
        if df.empty:
            return df

        # Additional feature engineering
        df["issue_date_dt"] = pd.to_datetime(df["issue_date"])
        df["due_date_dt"] = pd.to_datetime(df["due_date"])
        df["planned_loan_days"] = (df["due_date_dt"] - df["issue_date_dt"]).dt.days
        df["issue_month"] = df["issue_date_dt"].dt.month
        df["issue_day_of_week"] = df["issue_date_dt"].dt.day_name()
        df["issue_day_num"] = df["issue_date_dt"].dt.dayofweek
        df["book_age"] = datetime.now().year - df["book_pub_year"]

        # Book historical popularity (number of times borrowed in dataset)
        book_pop = df.groupby("book_id")["transaction_id"].transform("count")
        df["book_borrow_count"] = book_pop

        return df
    finally:
        conn.close()
