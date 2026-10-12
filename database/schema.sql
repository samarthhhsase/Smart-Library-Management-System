-- SQLite Schema for Smart Library Management System (SLMS)

CREATE TABLE IF NOT EXISTS books (
    book_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    category TEXT NOT NULL,
    isbn TEXT UNIQUE,
    publication_year INTEGER,
    total_copies INTEGER NOT NULL DEFAULT 1,
    available_copies INTEGER NOT NULL DEFAULT 1,
    shelf_location TEXT
);

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    department TEXT NOT NULL,
    year_of_study INTEGER NOT NULL,
    contact TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    book_id TEXT NOT NULL,
    issue_date TEXT NOT NULL,
    due_date TEXT NOT NULL,
    return_date TEXT,
    borrowing_frequency INTEGER DEFAULT 1,
    overdue_days INTEGER DEFAULT 0,
    overdue_status INTEGER DEFAULT 0,
    loan_duration_days INTEGER DEFAULT 14,
    notes TEXT,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_txn_student ON transactions(student_id);
CREATE INDEX IF NOT EXISTS idx_txn_book ON transactions(book_id);
CREATE INDEX IF NOT EXISTS idx_txn_issue_date ON transactions(issue_date);
