from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from typing import Any

from flask import Flask, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model .py"
MODEL_PKL = BASE_DIR / "model .pkl"
DB_PATH = BASE_DIR / "student_predictions.db"


def load_model_module():
    spec = importlib.util.spec_from_file_location("student_performance_model", MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load model module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_model_module()
model_module.load_model(MODEL_PKL)


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                branch_name TEXT NOT NULL,
                gender TEXT NOT NULL,
                study_hours REAL NOT NULL,
                attendance REAL NOT NULL,
                assignment_score REAL NOT NULL,
                sleep_hours REAL NOT NULL,
                tutoring_hours REAL NOT NULL,
                parental_support REAL NOT NULL,
                stress_level REAL NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def save_prediction(profile: dict[str, str], payload: dict[str, float], result: dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO prediction_records (
                student_name, branch_name, gender,
                study_hours, attendance, assignment_score,
                sleep_hours, tutoring_hours, parental_support, stress_level,
                prediction, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.get("StudentName", ""),
                profile.get("BranchName", ""),
                profile.get("Gender", ""),
                payload.get("StudyHours", 0.0),
                payload.get("Attendance", 0.0),
                payload.get("AssignmentScore", 0.0),
                payload.get("SleepHours", 0.0),
                payload.get("TutoringHours", 0.0),
                payload.get("ParentalSupport", 0.0),
                payload.get("StressLevel", 0.0),
                result.get("label", ""),
                result.get("confidence", 0.0),
            ),
        )
        connection.commit()


init_db()
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    with sqlite3.connect(DB_PATH) as connection:
        total_predictions = connection.execute(
            "SELECT COUNT(*) FROM prediction_records"
        ).fetchone()[0]
        branch_counts = connection.execute(
            """
            SELECT branch_name, COUNT(*) AS prediction_count
            FROM prediction_records
            GROUP BY branch_name
            ORDER BY prediction_count DESC, branch_name
            """
        ).fetchall()
    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        branch_counts=branch_counts,
    )


@app.route("/records")
def records():
    search = request.args.get("search", "").strip()
    escaped_search = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        if search:
            rows = connection.execute(
                """
                SELECT id, student_name, branch_name, gender, study_hours, attendance,
                       assignment_score, sleep_hours, tutoring_hours, parental_support,
                       stress_level, prediction, confidence, created_at
                FROM prediction_records
                WHERE student_name LIKE ? ESCAPE '\\'
                ORDER BY id DESC
                """,
                (f"%{escaped_search}%",),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, student_name, branch_name, gender, study_hours, attendance,
                       assignment_score, sleep_hours, tutoring_hours, parental_support,
                       stress_level, prediction, confidence, created_at
                FROM prediction_records
                ORDER BY id DESC
                """
            ).fetchall()
    return render_template("records.html", records=rows, search=search)


@app.route("/records/<int:record_id>/delete", methods=["POST"])
def delete_record(record_id: int):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM prediction_records WHERE id = ?", (record_id,))
        connection.commit()
    return redirect(url_for("records", search=request.form.get("search", "")))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = {}
        profile = {}

        profile_fields = {
            "StudentName": "Student Name",
            "BranchName": "Branch Name",
            "Gender": "Gender",
        }
        for key, label in profile_fields.items():
            value = request.form.get(key)
            if value is None or value.strip() == "":
                raise ValueError(f"Missing required value: {label}")
            profile[key] = value.strip()

        required_fields = {
            "StudyHours": (1.0, 12.0),
            "Attendance": (0.0, 100.0),
            "AssignmentScore": (0.0, 100.0),
            "SleepHours": (4.0, 12.0),
            "TutoringHours": (0.0, 10.0),
            "ParentalSupport": (1.0, 3.0),
            "StressLevel": (1.0, 10.0),
        }

        for key, (minimum, maximum) in required_fields.items():
            raw_value = request.form.get(key)
            if raw_value is None or raw_value == "":
                raise ValueError(f"Missing required value: {key}")

            value = float(raw_value)
            if value < minimum or value > maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}.")
            payload[key] = value

        result = model_module.predict_performance(payload)
        save_prediction(profile, payload, result)
        return render_template(
            "result.html",
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            payload=payload,
            profile=profile,
        )
    except ValueError as exc:
        return render_template("index.html", error=str(exc))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
