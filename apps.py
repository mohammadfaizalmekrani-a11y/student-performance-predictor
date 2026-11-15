
import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import joblib
import pandas as pd
import os
from datetime import datetime

# ==========================================
# 1. LOAD AI MODEL
# ==========================================

MODEL_PATH = "models/student_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run python train_models.py first."
    )

model = joblib.load(MODEL_PATH)

FEATURES = [
    "study_hours",
    "attendance",
    "physics",
    "mathematics",
    "c_programming",
    "ai_marks",
    "electronics",
    "assignment_marks"
]

# ==========================================
# 2. DATABASE
# ==========================================

os.makedirs("database", exist_ok=True)

conn = sqlite3.connect("database/students.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    study_hours REAL,
    attendance REAL,
    previous_marks REAL,
    assignment_marks REAL,
    predicted_marks REAL,
    performance TEXT
)
""")

# Add new columns while preserving old history
new_columns = {
    "physics": "REAL",
    "mathematics": "REAL",
    "c_programming": "REAL",
    "ai_marks": "REAL",
    "electronics": "REAL",
    "created_at": "TEXT"
}

cursor.execute("PRAGMA table_info(predictions)")
existing_columns = {row[1] for row in cursor.fetchall()}

for column, datatype in new_columns.items():
    if column not in existing_columns:
        cursor.execute(
            f"ALTER TABLE predictions ADD COLUMN {column} {datatype}"
        )

conn.commit()

# ==========================================
# 3. WINDOW
# ==========================================

root = tk.Tk()
root.title("Student Performance Predictor | V3.0")
root.geometry("1120x900")
root.minsize(950, 780)
root.configure(bg="#F1F5F9")

NAVY = "#142B4A"
BLUE = "#2563EB"
TEAL = "#0F9D8A"
WHITE = "#FFFFFF"
BG = "#F1F5F9"
DARK = "#1E293B"
GREY = "#64748B"
RED = "#DC4545"

# ==========================================
# 4. STYLES
# ==========================================

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "Treeview",
    background=WHITE,
    foreground=DARK,
    rowheight=32,
    fieldbackground=WHITE,
    font=("Segoe UI", 9)
)

style.configure(
    "Treeview.Heading",
    background=NAVY,
    foreground=WHITE,
    font=("Segoe UI", 9, "bold"),
    padding=8
)

style.configure(
    "TEntry",
    padding=8,
    font=("Segoe UI", 10)
)

# ==========================================
# 5. HEADER
# ==========================================

header = tk.Frame(root, bg=NAVY, height=105)
header.pack(fill="x")
header.pack_propagate(False)

tk.Label(
    header,
    text="STUDENT PERFORMANCE PREDICTOR",
    bg=NAVY,
    fg=WHITE,
    font=("Segoe UI", 23, "bold")
).pack(pady=(18, 4))

tk.Label(
    header,
    text="SMART ACADEMIC ANALYTICS  |  AI & MACHINE LEARNING  |  V3.0",
    bg=NAVY,
    fg="#BDD4F5",
    font=("Segoe UI", 9)
).pack()

# ==========================================
# 6. SCROLLABLE CONTENT
# ==========================================

outer = tk.Frame(root, bg=BG)
outer.pack(fill="both", expand=True)

canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
scrollbar = ttk.Scrollbar(
    outer, orient="vertical", command=canvas.yview
)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

canvas.configure(yscrollcommand=scrollbar.set)

content = tk.Frame(canvas, bg=BG)
window_id = canvas.create_window(
    (0, 0), window=content, anchor="nw"
)

def update_scroll(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

def resize_content(event):
    canvas.itemconfigure(window_id, width=event.width)

content.bind("<Configure>", update_scroll)
canvas.bind("<Configure>", resize_content)

# ==========================================
# 7. STUDENT DETAILS
# ==========================================

tk.Label(
    content,
    text="Student Information",
    bg=BG,
    fg=NAVY,
    font=("Segoe UI", 17, "bold")
).pack(anchor="w", padx=25, pady=(18, 10))

form = tk.Frame(
    content,
    bg=WHITE,
    padx=22,
    pady=18,
    highlightbackground="#DCE3EC",
    highlightthickness=1
)
form.pack(fill="x", padx=25)

entries = {}

fields = [
    ("Student Name", "name", "Enter full name"),
    ("Study Hours Per Day (0-10)", "study", "Example: 5"),
    ("Attendance Percentage (40-100)", "attendance", "Example: 85"),
    ("Physics Marks (0-50)", "physics", "Enter physics marks"),
    ("Mathematics Marks (0-50)", "mathematics", "Enter maths marks"),
    ("C Programming Marks (0-50)", "c_programming", "Enter C marks"),
    ("AI Marks (0-50)", "ai_marks", "Enter AI marks"),
    ("Electronics Marks (0-50)", "electronics", "Enter electronics marks"),
    ("Assignment Marks (0-50)", "assignment_marks", "Enter assignment marks")
]

for i, (label, key, placeholder) in enumerate(fields):

    row = i // 2
    col = i % 2

    cell = tk.Frame(form, bg=WHITE)
    cell.grid(
        row=row,
        column=col,
        sticky="ew",
        padx=12,
        pady=9
    )

    tk.Label(
        cell,
        text=label,
        bg=WHITE,
        fg=DARK,
        font=("Segoe UI", 10, "bold")
    ).pack(anchor="w", pady=(0, 5))

    entry = ttk.Entry(cell)
    entry.pack(fill="x")

    # Display placeholder-like help below the field
    tk.Label(
        cell,
        text=placeholder,
        bg=WHITE,
        fg=GREY,
        font=("Segoe UI", 8)
    ).pack(anchor="w", pady=(3, 0))

    entries[key] = entry

form.columnconfigure(0, weight=1)
form.columnconfigure(1, weight=1)

# ==========================================
# 8. SUBJECT AVERAGE
# ==========================================

average_card = tk.Frame(
    content,
    bg="#E6F4F1",
    padx=18,
    pady=12
)
average_card.pack(fill="x", padx=25, pady=15)

tk.Label(
    average_card,
    text="SUBJECT AVERAGE",
    bg="#E6F4F1",
    fg="#167766",
    font=("Segoe UI", 10, "bold")
).pack()

average_label = tk.Label(
    average_card,
    text="-- / 100",
    bg="#E6F4F1",
    fg="#087F6D",
    font=("Segoe UI", 25, "bold")
)
average_label.pack()

subject_keys = [
    "physics",
    "mathematics",
    "c_programming",
    "ai_marks",
    "electronics"
]

def update_average(event=None):

    try:
        values = [
            float(entries[key].get())
            for key in subject_keys
            if entries[key].get().strip()
        ]

        if len(values) == 5 and all(0 <= v <= 50 for v in values):
            avg = sum(values) / 5
            average_label.config(text=f"{avg:.2f} / 50")
        else:
            average_label.config(text="-- / 50")

    except ValueError:
        average_label.config(text="-- / 50")

for key in subject_keys:
    entries[key].bind("<KeyRelease>", update_average)

# ==========================================
# 9. PREDICTION RESULT
# ==========================================

result_card = tk.Frame(
    content,
    bg=WHITE,
    padx=20,
    pady=16,
    highlightbackground="#DCE3EC",
    highlightthickness=1
)
result_card.pack(fill="x", padx=25, pady=5)

tk.Label(
    result_card,
    text="AI PREDICTION RESULT",
    bg=WHITE,
    fg=GREY,
    font=("Segoe UI", 11, "bold")
).pack()

result_label = tk.Label(
    result_card,
    text="-- / 100",
    bg=WHITE,
    fg=BLUE,
    font=("Segoe UI", 32, "bold")
)
result_label.pack(pady=5)

category_label = tk.Label(
    result_card,
    text="Enter student information to begin",
    bg=WHITE,
    fg=GREY,
    font=("Segoe UI", 12, "bold")
)
category_label.pack()

# ==========================================
# 10. PREDICTION FUNCTION
# ==========================================

def predict_performance():

    try:
        name = entries["name"].get().strip()

        if not name:
            raise ValueError("Please enter student name.")

        study = float(entries["study"].get())
        attendance = float(entries["attendance"].get())

        values = {}

        for key in subject_keys + ["assignment_marks"]:
            values[key] = float(entries[key].get())

        if not 0 <= study <= 10:
            raise ValueError("Study hours must be between 0 and 10.")

        if not 40 <= attendance <= 100:
            raise ValueError("Attendance must be between 40 and 100.")

        for key, value in values.items():
            if not 0 <= value <= 50:
                raise ValueError(
                    f"{key.replace('_', ' ').title()} must be between 0 and 50."
                )

        features = pd.DataFrame([{
            "study_hours": study,
            "attendance": attendance,
            "physics": values["physics"],
            "mathematics": values["mathematics"],
            "c_programming": values["c_programming"],
            "ai_marks": values["ai_marks"],
            "electronics": values["electronics"],
            "assignment_marks": values["assignment_marks"]
        }], columns=FEATURES)

        predicted = float(model.predict(features)[0])
        predicted = round(max(0, min(50, predicted)), 2)

        if predicted >= 45:
            category = "Excellent"
            color = TEAL
        elif predicted >= 30:
            category = "Good"
            color = BLUE
        elif predicted >= 20:
            category = "Average"
            color = "#D18B22"
        else:
            category = "Needs Improvement"
            color = RED

        avg = sum(values[key] for key in subject_keys) / 5

        result_label.config(
            text=f"{predicted:.2f} / 50",
            fg=color
        )

        category_label.config(
            text=f"Performance: {category}",
            fg=color
        )

        average_label.config(text=f"{avg:.2f} / 50")

        cursor.execute("""
            INSERT INTO predictions (
                name, study_hours, attendance,
                previous_marks, assignment_marks,
                predicted_marks, performance,
                physics, mathematics, c_programming,
                ai_marks, electronics, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, study, attendance,
            avg, values["assignment_marks"],
            predicted, category,
            values["physics"], values["mathematics"],
            values["c_programming"], values["ai_marks"],
            values["electronics"],
            datetime.now().strftime("%d-%m-%Y %I:%M %p")
        ))

        conn.commit()
        refresh_history()

    except ValueError as error:
        messagebox.showerror("Input Error", str(error))

    except Exception as error:
        messagebox.showerror("Prediction Error", str(error))

# ==========================================
# 11. CLEAR FORM
# ==========================================

def clear_form():

    for entry in entries.values():
        entry.delete(0, tk.END)

    result_label.config(text="-- / 50", fg=BLUE)
    category_label.config(
        text="Enter student information to begin",
        fg=GREY
    )

    average_label.config(text="-- / 50")

# ==========================================
# 12. RESET HISTORY
# ==========================================

def reset_history():

    answer = messagebox.askyesno(
        "Confirm Reset",
        "Delete all prediction history?\nThis cannot be undone."
    )

    if answer:
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        refresh_history()

        messagebox.showinfo(
            "Success",
            "Prediction history has been reset."
        )

# ==========================================
# 13. ACTION BUTTONS
# ==========================================

buttons = tk.Frame(content, bg=BG)
buttons.pack(fill="x", padx=25, pady=15)

tk.Button(
    buttons,
    text="PREDICT PERFORMANCE",
    command=predict_performance,
    bg=BLUE,
    fg=WHITE,
    font=("Segoe UI", 10, "bold"),
    padx=18,
    pady=12,
    relief="flat",
    cursor="hand2"
).pack(side="left", padx=5)

tk.Button(
    buttons,
    text="CLEAR FORM",
    command=clear_form,
    bg="#64748B",
    fg=WHITE,
    font=("Segoe UI", 10, "bold"),
    padx=18,
    pady=12,
    relief="flat",
    cursor="hand2"
).pack(side="left", padx=5)

tk.Button(
    buttons,
    text="RESET HISTORY",
    command=reset_history,
    bg=RED,
    fg=WHITE,
    font=("Segoe UI", 10, "bold"),
    padx=18,
    pady=12,
    relief="flat",
    cursor="hand2"
).pack(side="right", padx=5)

# ==========================================
# 14. HISTORY TABLE
# ==========================================

tk.Label(
    content,
    text="Recent Prediction History",
    bg=BG,
    fg=NAVY,
    font=("Segoe UI", 16, "bold")
).pack(anchor="w", padx=25, pady=(5, 10))

history_frame = tk.Frame(content, bg=WHITE)
history_frame.pack(fill="x", padx=25, pady=(0, 20))

columns = ("name", "marks", "category", "date")

tree = ttk.Treeview(
    history_frame,
    columns=columns,
    show="headings",
    height=6
)

tree.heading("name", text="Student")
tree.heading("marks", text="Predicted Marks")
tree.heading("category", text="Performance")
tree.heading("date", text="Date & Time")

tree.column("name", width=200)
tree.column("marks", width=130, anchor="center")
tree.column("category", width=150, anchor="center")
tree.column("date", width=190, anchor="center")

tree.pack(fill="x", padx=8, pady=8)

def refresh_history():

    for item in tree.get_children():
        tree.delete(item)

    cursor.execute("""
        SELECT name, predicted_marks, performance, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 100
    """)

    for row in cursor.fetchall():
        tree.insert(
            "",
            "end",
            values=(
                row[0],
                f"{row[1]:.2f}",
                row[2],
                row[3] or "Older record"
            )
        )

refresh_history()

# ==========================================
# 15. FOOTER
# ==========================================

tk.Label(
    root,
    text="STUDENT PERFORMANCE PREDICTOR  |  V3.0",
    bg=NAVY,
    fg=WHITE,
    font=("Segoe UI", 9),
    pady=10
).pack(fill="x")

# ==========================================
# 16. CLOSE
# ==========================================

def close_application():
    conn.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", close_application)

root.mainloop()
