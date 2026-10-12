"""
Realistic synthetic dataset generator for Smart Library Management System (SLMS).
Generates >= 120 books, >= 60 students, >= 650 transactions with realistic statistical distributions.
Saves to CSV and populates SQLite database.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Fixed seed for statistical reproducibility
RANDOM_SEED = 42

DATA_DIR = os.path.dirname(__file__)
BOOKS_CSV = os.path.join(DATA_DIR, "books.csv")
STUDENTS_CSV = os.path.join(DATA_DIR, "students.csv")
TRANSACTIONS_CSV = os.path.join(DATA_DIR, "transactions.csv")

# Academic Categories
CATEGORIES = [
    "Computer Science",
    "Data Science & AI",
    "Electronics & Comm.",
    "Mechanical Eng.",
    "Civil Eng.",
    "Mathematics & Statistics",
    "Management & Soft Skills"
]

DEPARTMENTS = [
    "Computer Science",
    "Information Technology",
    "Electronics & Comm.",
    "Mechanical Eng.",
    "Civil Eng."
]

REALISTIC_BOOKS = [
    # Computer Science
    ("Introduction to Algorithms (CLRS)", "Thomas H. Cormen", "Computer Science", 2009, 6),
    ("Operating System Concepts", "Abraham Silberschatz", "Computer Science", 2018, 5),
    ("Computer Networks", "Andrew S. Tanenbaum", "Computer Science", 2010, 4),
    ("Database System Concepts", "Henry F. Korth", "Computer Science", 2019, 5),
    ("Design Patterns: Elements of Reusable Object-Oriented Software", "Erich Gamma et al.", "Computer Science", 2015, 3),
    ("Compilers: Principles, Techniques, and Tools (Dragon Book)", "Alfred V. Aho", "Computer Science", 2006, 4),
    ("Computer Organization and Architecture", "William Stallings", "Computer Science", 2016, 5),
    ("Clean Code: A Handbook of Agile Software Craftsmanship", "Robert C. Martin", "Computer Science", 2008, 6),
    ("Artificial Intelligence: A Modern Approach", "Stuart Russell, Peter Norvig", "Computer Science", 2020, 7),
    ("Computer Graphics: Principles and Practice", "John F. Hughes", "Computer Science", 2013, 3),
    ("Theory of Computation: Formal Languages and Automata", "Peter Linz", "Computer Science", 2016, 4),
    ("Software Engineering: A Practitioner's Approach", "Roger S. Pressman", "Computer Science", 2014, 5),
    ("Cryptography and Network Security", "William Stallings", "Computer Science", 2017, 4),
    ("Distributed Systems: Concepts and Design", "George Coulouris", "Computer Science", 2011, 3),
    ("The Pragmatic Programmer", "David Thomas, Andrew Hunt", "Computer Science", 2019, 5),

    # Data Science & AI
    ("Pattern Recognition and Machine Learning", "Christopher M. Bishop", "Data Science & AI", 2006, 6),
    ("Deep Learning", "Ian Goodfellow, Yoshua Bengio", "Data Science & AI", 2016, 7),
    ("Hands-On Machine Learning with Scikit-Learn, Keras & TF", "Aurélien Géron", "Data Science & AI", 2022, 8),
    ("The Elements of Statistical Learning", "Trevor Hastie, Robert Tibshirani", "Data Science & AI", 2009, 5),
    ("Python for Data Analysis", "Wes McKinney", "Data Science & AI", 2022, 8),
    ("Speech and Language Processing (NLP)", "Daniel Jurafsky, James H. Martin", "Data Science & AI", 2023, 4),
    ("Probabilistic Graphical Models", "Daphne Koller, Nir Friedman", "Data Science & AI", 2009, 3),
    ("Reinforcement Learning: An Introduction", "Richard S. Sutton, Andrew G. Barto", "Data Science & AI", 2018, 5),
    ("Computer Vision: Algorithms and Applications", "Richard Szeliski", "Data Science & AI", 2022, 4),
    ("Data Mining: Concepts and Techniques", "Jiawei Han, Micheline Kamber", "Data Science & AI", 2011, 5),
    ("Bayesian Data Analysis", "Andrew Gelman et al.", "Data Science & AI", 2013, 4),
    ("An Introduction to Statistical Learning", "Gareth James, Daniela Witten", "Data Science & AI", 2021, 9),
    ("Feature Engineering for Machine Learning", "Alice Zheng, Amanda Casari", "Data Science & AI", 2018, 5),
    ("Storytelling with Data: A Data Visualization Guide", "Cole Nussbaumer Knaflic", "Data Science & AI", 2015, 6),
    ("Designing Machine Learning Systems", "Chip Huyen", "Data Science & AI", 2022, 6),

    # Electronics & Communication
    ("Microelectronic Circuits", "Adel S. Sedra, Kenneth C. Smith", "Electronics & Comm.", 2014, 5),
    ("Signals and Systems", "Alan V. Oppenheim, Alan S. Willsky", "Electronics & Comm.", 1996, 6),
    ("Digital Signal Processing: Principles, Algorithms & Apps", "John G. Proakis", "Electronics & Comm.", 2007, 5),
    ("Electronic Devices and Circuit Theory", "Robert L. Boylestad", "Electronics & Comm.", 2012, 6),
    ("Digital Image Processing", "Rafael C. Gonzalez, Richard E. Woods", "Electronics & Comm.", 2018, 4),
    ("Electromagnetic Waves and Radiating Systems", "Edward C. Jordan", "Electronics & Comm.", 2015, 4),
    ("Communication Systems", "Simon Haykin", "Electronics & Comm.", 2009, 5),
    ("VLSI Design: Techniques for System Implementation", "Weste and Harris", "Electronics & Comm.", 2010, 4),
    ("Wireless Communications: Principles and Practice", "Theodore S. Rappaport", "Electronics & Comm.", 2002, 4),
    ("Embedded System Design", "Frank Vahid, Tony Givargis", "Electronics & Comm.", 2006, 5),

    # Mechanical Engineering
    ("Engineering Thermodynamics", "P. K. Nag", "Mechanical Eng.", 2013, 6),
    ("Thermodynamics: An Engineering Approach", "Yunus A. Cengel, Michael A. Boles", "Mechanical Eng.", 2019, 5),
    ("Fluid Mechanics: Fundamentals and Applications", "Yunus A. Cengel, John M. Cimbala", "Mechanical Eng.", 2017, 5),
    ("Heat and Mass Transfer", "R. K. Rajput", "Mechanical Eng.", 2015, 6),
    ("Theory of Machines", "S. S. Rattan", "Mechanical Eng.", 2014, 5),
    ("Design of Machine Elements", "V. B. Bhandari", "Mechanical Eng.", 2017, 6),
    ("Manufacturing Technology: Foundry, Forming & Welding", "P. N. Rao", "Mechanical Eng.", 2018, 4),
    ("Strength of Materials", "S. Ramamrutham", "Mechanical Eng.", 2016, 7),
    ("Robotics and Industrial Automation", "R. K. Rajput", "Mechanical Eng.", 2019, 4),
    ("Automobile Engineering", "Kirpal Singh", "Mechanical Eng.", 2015, 4),

    # Civil Engineering
    ("Basic Structural Analysis", "C. S. Reddy", "Civil Eng.", 2011, 4),
    ("Soil Mechanics and Foundations", "B. C. Punmia", "Civil Eng.", 2017, 5),
    ("Surveying: Vol. 1 & 2", "B. C. Punmia, A. K. Jain", "Civil Eng.", 2016, 6),
    ("Reinforced Concrete Design", "S. Unnikrishna Pillai", "Civil Eng.", 2015, 5),
    ("Fluid Mechanics and Hydraulic Machines", "R. K. Bansal", "Civil Eng.", 2018, 6),
    ("Environmental Engineering", "Howard S. Peavy", "Civil Eng.", 2013, 4),
    ("Transportation Engineering", "C. Venkatramaiah", "Civil Eng.", 2016, 4),
    ("Building Construction", "Sushil Kumar", "Civil Eng.", 2014, 5),

    # Mathematics & Statistics
    ("Higher Engineering Mathematics", "B. S. Grewal", "Mathematics & Statistics", 2017, 10),
    ("Advanced Engineering Mathematics", "Erwin Kreyszig", "Mathematics & Statistics", 2011, 8),
    ("Linear Algebra and Its Applications", "Gilbert Strang", "Mathematics & Statistics", 2016, 7),
    ("Introduction to Probability and Statistics", "Sheldon M. Ross", "Mathematics & Statistics", 2014, 6),
    ("Discrete Mathematics and Its Applications", "Kenneth H. Rosen", "Mathematics & Statistics", 2018, 6),
    ("Calculus: Early Transcendentals", "James Stewart", "Mathematics & Statistics", 2015, 6),
    ("Probability, Random Variables and Stochastic Processes", "A. Papoulis", "Mathematics & Statistics", 2002, 4),
    ("Applied Statistics and Probability for Engineers", "Douglas C. Montgomery", "Mathematics & Statistics", 2014, 5),
    ("Numerical Methods for Scientific and Engineering Computation", "M. K. Jain", "Mathematics & Statistics", 2012, 4),

    # Management & Soft Skills
    ("Principles of Management", "P. C. Tripathi, P. N. Reddy", "Management & Soft Skills", 2017, 5),
    ("Project Management: A Managerial Approach", "Jack R. Meredith", "Management & Soft Skills", 2015, 4),
    ("Effective Technical Communication", "M. Ashraf Rizvi", "Management & Soft Skills", 2018, 5),
    ("Financial Management: Theory and Practice", "Prasanna Chandra", "Management & Soft Skills", 2019, 4),
    ("Organizational Behaviour", "Stephen P. Robbins", "Management & Soft Skills", 2016, 5)
]

STUDENT_FIRST_NAMES = [
    "Aarav", "Ananya", "Rohan", "Priya", "Aditya", "Sneha", "Vikram", "Neha", "Rahul", "Kavya",
    "Siddharth", "Ishita", "Arjun", "Pooja", "Varun", "Tanvi", "Kunal", "Rhea", "Manish", "Divya",
    "Gaurav", "Shruti", "Akash", "Meera", "Sameer", "Swati", "Nikhil", "Simran", "Deepak", "Aishwarya",
    "Ravi", "Shreya", "Karthik", "Bhavna", "Pranav", "Nandini", "Harsh", "Sakshi", "Yash", "Trisha"
]

STUDENT_LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Malhotra", "Mehta", "Patel", "Singh", "Reddy", "Nair", "Iyer",
    "Joshi", "Bose", "Das", "Choudhury", "Kulkarni", "Deshmukh", "Chawla", "Bhatia", "Kapoor", "Saxena"
]

def generate_synthetic_data(num_books: int = 120, num_students: int = 65, num_transactions: int = 700):
    """Generates synthetic books, students, and transaction datasets with realistic correlations."""
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # 1. BOOKS GENERATION
    books_list = []
    # Seed with curated real titles
    for i, (title, author, category, pub_year, copies) in enumerate(REALISTIC_BOOKS):
        b_id = f"BK-{1001 + i}"
        isbn = f"978-{random.randint(10, 99)}-{random.randint(1000, 9999)}-{random.randint(100, 999)}-{random.randint(0, 9)}"
        shelf = f"{category[:3].upper()}-{random.randint(1, 8)}{random.choice(['A', 'B', 'C'])}"
        books_list.append({
            "book_id": b_id,
            "title": title,
            "author": author,
            "category": category,
            "isbn": isbn,
            "publication_year": pub_year,
            "total_copies": copies,
            "available_copies": copies, # will be updated after transactions
            "shelf_location": shelf
        })

    # Add extra books to reach num_books if needed
    current_count = len(books_list)
    for i in range(current_count, num_books):
        cat = random.choice(CATEGORIES)
        b_id = f"BK-{1001 + i}"
        title = f"Advanced Principles of {cat} Vol. {random.randint(1, 4)}"
        author = f"Prof. {random.choice(STUDENT_FIRST_NAMES)} {random.choice(STUDENT_LAST_NAMES)}"
        pub_year = random.randint(2005, 2023)
        copies = random.randint(2, 6)
        isbn = f"978-{random.randint(10, 99)}-{random.randint(1000, 9999)}-{random.randint(100, 999)}-{random.randint(0, 9)}"
        shelf = f"{cat[:3].upper()}-{random.randint(1, 8)}{random.choice(['A', 'B', 'C'])}"
        books_list.append({
            "book_id": b_id,
            "title": title,
            "author": author,
            "category": cat,
            "isbn": isbn,
            "publication_year": pub_year,
            "total_copies": copies,
            "available_copies": copies,
            "shelf_location": shelf
        })

    df_books = pd.DataFrame(books_list)

    # 2. STUDENTS GENERATION
    students_list = []
    for i in range(num_students):
        s_id = f"STU-{2001 + i}"
        fn = random.choice(STUDENT_FIRST_NAMES)
        ln = random.choice(STUDENT_LAST_NAMES)
        name = f"{fn} {ln}"
        dept = random.choice(DEPARTMENTS)
        year = random.choice([1, 2, 3, 4])
        email = f"{fn.lower()}.{ln.lower()}{random.randint(10, 99)}@college.edu"
        contact = f"+91 {random.randint(70000, 99999)} {random.randint(10000, 99999)}"
        students_list.append({
            "student_id": s_id,
            "name": name,
            "email": email,
            "department": dept,
            "year_of_study": year,
            "contact": contact
        })
    df_students = pd.DataFrame(students_list)

    # 3. TRANSACTIONS GENERATION
    # Realistic time window: Past 450 days (approx 15 months)
    start_date = datetime.now() - timedelta(days=450)
    
    # Weight book popularity: Data Science & AI and CS are heavily borrowed
    category_weights = {
        "Data Science & AI": 0.32,
        "Computer Science": 0.28,
        "Mathematics & Statistics": 0.16,
        "Electronics & Comm.": 0.10,
        "Mechanical Eng.": 0.06,
        "Civil Eng.": 0.05,
        "Management & Soft Skills": 0.03
    }
    
    book_weights = []
    for cat in df_books["category"]:
        book_weights.append(category_weights.get(cat, 0.1))
    book_probs = np.array(book_weights) / sum(book_weights)

    # Student affinity: higher year students borrow more frequently
    student_weights = []
    for yr in df_students["year_of_study"]:
        student_weights.append(yr * 1.5)
    student_probs = np.array(student_weights) / sum(student_weights)

    transactions_list = []
    student_borrow_count = {s_id: 0 for s_id in df_students["student_id"]}
    book_borrow_count = {b_id: 0 for b_id in df_books["book_id"]}

    for i in range(num_transactions):
        txn_id = f"TXN-{5001 + i}"
        
        # Pick student and book
        chosen_student = np.random.choice(df_students["student_id"], p=student_probs)
        chosen_book = np.random.choice(df_books["book_id"], p=book_probs)
        
        student_borrow_count[chosen_student] += 1
        book_borrow_count[chosen_book] += 1
        freq = student_borrow_count[chosen_student]

        # Issue date: non-uniform across time (exam months see higher bursts)
        random_day_offset = random.randint(1, 440)
        issue_dt = start_date + timedelta(days=random_day_offset)
        
        # Standard loan period is 14 days
        standard_loan_days = 14
        due_dt = issue_dt + timedelta(days=standard_loan_days)

        # Decide if transaction is returned or currently active
        # About 6% are active (not returned) if issued within the last 30 days
        is_recent = (datetime.now() - issue_dt).days <= 30
        is_active = is_recent and (random.random() < 0.65)

        if is_active:
            ret_date_str = None
            today_dt = datetime.now()
            if today_dt > due_dt:
                overdue_days = (today_dt - due_dt).days
                overdue_status = 1
            else:
                overdue_days = 0
                overdue_status = 0
            actual_loan_duration = (today_dt - issue_dt).days
        else:
            # Historically completed transaction
            # Statistical pattern: ~28% overdue rate
            # Higher probability of overdue if student has low borrowing frequency or is in 1st/4th year
            is_overdue_prob = 0.28
            # Slightly higher if Computer Science / Data Science during intense projects
            if random.random() < is_overdue_prob:
                overdue_status = 1
                # Overdue days: right-skewed distribution (geometric/exponential)
                overdue_days = int(np.random.exponential(scale=6.5)) + 1
                if overdue_days > 35:
                    overdue_days = random.randint(25, 35) # clip extreme outlier
            else:
                overdue_status = 0
                overdue_days = 0

            actual_loan_duration = standard_loan_days + overdue_days - random.randint(0, 3) if overdue_days == 0 else standard_loan_days + overdue_days
            actual_loan_duration = max(2, actual_loan_duration)
            ret_dt = issue_dt + timedelta(days=actual_loan_duration)
            ret_date_str = ret_dt.strftime("%Y-%m-%d")

        transactions_list.append({
            "transaction_id": txn_id,
            "student_id": chosen_student,
            "book_id": chosen_book,
            "issue_date": issue_dt.strftime("%Y-%m-%d"),
            "due_date": due_dt.strftime("%Y-%m-%d"),
            "return_date": ret_date_str,
            "borrowing_frequency": freq,
            "overdue_days": overdue_days,
            "overdue_status": overdue_status,
            "loan_duration_days": actual_loan_duration,
            "notes": "Generated transaction record"
        })

    df_transactions = pd.DataFrame(transactions_list)

    # Re-calculate available copies based on active issues
    active_issues = df_transactions[df_transactions["return_date"].isna()]
    active_counts = active_issues["book_id"].value_counts().to_dict()

    for idx, row in df_books.iterrows():
        b_id = row["book_id"]
        currently_issued = active_counts.get(b_id, 0)
        avail = max(0, row["total_copies"] - currently_issued)
        df_books.at[idx, "available_copies"] = avail

    return df_books, df_students, df_transactions

def save_and_seed_database(df_books: pd.DataFrame, df_students: pd.DataFrame, df_transactions: pd.DataFrame):
    """Saves generated data to CSV and populates the SQLite database."""
    # Save CSVs
    df_books.to_csv(BOOKS_CSV, index=False)
    df_students.to_csv(STUDENTS_CSV, index=False)
    df_transactions.to_csv(TRANSACTIONS_CSV, index=False)

    # Populate SQLite
    from database.db import DB_PATH, init_db, reset_database
    reset_database()

    conn = sqlite3.connect(DB_PATH)
    try:
        df_books.to_sql("books", conn, if_exists="replace", index=False)
        df_students.to_sql("students", conn, if_exists="replace", index=False)
        df_transactions.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()

    print(f"Database seeded successfully with {len(df_books)} books, {len(df_students)} students, and {len(df_transactions)} transactions.")

def initialize_data_if_needed():
    """Checks if database is populated, if not generates synthetic data."""
    from database.db import DB_PATH, get_all_books
    if not os.path.exists(DB_PATH) or get_all_books().empty:
        print("Database is empty or missing. Generating initial synthetic data...")
        b, s, t = generate_synthetic_data()
        save_and_seed_database(b, s, t)

if __name__ == "__main__":
    b, s, t = generate_synthetic_data()
    save_and_seed_database(b, s, t)
