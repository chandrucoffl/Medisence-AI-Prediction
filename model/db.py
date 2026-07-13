"""
Patient record store for the unified "Comprehensive AI Health Report".
Every prediction made in the ML / DL / NLP modules can be attached to a
patient, so a doctor can pull one combined report showing everything
the AI platform has found across all modules for that patient.
"""
import sqlite3, os, json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medisense.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        doctor TEXT,
        created_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        module TEXT NOT NULL,
        item_name TEXT NOT NULL,
        result TEXT NOT NULL,
        confidence TEXT,
        risk_level TEXT,
        details TEXT,
        created_at TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(id)
    )""")
    conn.commit()
    conn.close()

def create_patient(name, age, gender, doctor):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO patients (name, age, gender, doctor, created_at) VALUES (?,?,?,?,?)",
        (name, age, gender, doctor, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid

def list_patients():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_patient(patient_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def add_assessment(patient_id, module, item_name, result, confidence, risk_level, details=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO assessments (patient_id, module, item_name, result, confidence, risk_level, details, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (patient_id, module, item_name, result, confidence, risk_level,
         json.dumps(details) if details else None, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

def get_assessments(patient_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM assessments WHERE patient_id=? ORDER BY created_at DESC", (patient_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
