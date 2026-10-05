from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).resolve().parent / "student_performance.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model .pkl"


def build_student_dataset(path: Path) -> pd.DataFrame:
    """Create a deterministic, balanced dataset for the model."""
    expected_columns = [
        "StudyHours",
        "Attendance",
        "AssignmentScore",
        "SleepHours",
        "TutoringHours",
        "ParentalSupport",
        "StressLevel",
        "Performance",
    ]

    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
        if list(df.columns) == expected_columns and not df.empty:
            labels = set(df["Performance"].dropna().unique())
            if {"Low", "Average", "High"}.issubset(labels):
                return df

    rng = np.random.default_rng(42)
    rows = []
    label_ranges = {
        "Low": (0, 59),
        "Average": (60, 79),
        "High": (80, 100),
    }

    for performance, (min_score, max_score) in label_ranges.items():
        for _ in range(170):
            study_hours = float(rng.integers(1, 11))
            attendance = float(rng.integers(40, 101))
            assignment_score = float(rng.integers(25, 101))
            sleep_hours = float(rng.integers(4, 10))
            tutoring_hours = float(rng.integers(0, 6))
            parental_support = float(rng.integers(1, 4))
            stress_level = float(rng.integers(1, 11))

            if performance == "High":
                study_hours = float(rng.integers(5, 11))
                attendance = float(rng.integers(70, 101))
                assignment_score = float(rng.integers(70, 101))
                sleep_hours = float(rng.integers(6, 10))
                tutoring_hours = float(rng.integers(1, 6))
                parental_support = float(rng.integers(1, 4))
                stress_level = float(rng.integers(1, 8))
            elif performance == "Average":
                study_hours = float(rng.integers(3, 9))
                attendance = float(rng.integers(55, 90))
                assignment_score = float(rng.integers(55, 85))
                sleep_hours = float(rng.integers(5, 9))
                tutoring_hours = float(rng.integers(1, 5))
                parental_support = float(rng.integers(1, 4))
                stress_level = float(rng.integers(4, 9))
            else:
                study_hours = float(rng.integers(1, 7))
                attendance = float(rng.integers(40, 75))
                assignment_score = float(rng.integers(25, 70))
                sleep_hours = float(rng.integers(4, 8))
                tutoring_hours = float(rng.integers(0, 4))
                parental_support = float(rng.integers(1, 4))
                stress_level = float(rng.integers(5, 11))

            score = (
                (study_hours * 5)
                + (attendance * 0.5)
                + (assignment_score * 0.55)
                + (sleep_hours * 5)
                + (tutoring_hours * 3)
                + (parental_support * 10)
                - (stress_level * 4)
            )
            score = max(0, min(100, score))

            if score < min_score or score > max_score:
                score = float(rng.integers(min_score, max_score + 1))

            rows.append(
                {
                    "StudyHours": float(study_hours),
                    "Attendance": float(attendance),
                    "AssignmentScore": float(assignment_score),
                    "SleepHours": float(sleep_hours),
                    "TutoringHours": float(tutoring_hours),
                    "ParentalSupport": float(parental_support),
                    "StressLevel": float(stress_level),
                    "Performance": performance,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def train_model() -> RandomForestClassifier:
    df = build_student_dataset(DATASET_PATH)
    features = [
        "StudyHours",
        "Attendance",
        "AssignmentScore",
        "SleepHours",
        "TutoringHours",
        "ParentalSupport",
        "StressLevel",
    ]
    X = df[features]
    y = df["Performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print(f"Model accuracy: {accuracy:.2%}")
    return model


def save_model(model: RandomForestClassifier, path: Path = MODEL_PATH) -> None:
    with path.open("wb") as file:
        pickle.dump(model, file)


def load_model(path: Path = MODEL_PATH) -> RandomForestClassifier:
    if not path.exists() or path.stat().st_size == 0:
        model = train_model()
        save_model(model, path)
        return model

    with path.open("rb") as file:
        return pickle.load(file)


def predict_performance(raw_features: dict) -> dict:
    required_keys = [
        "StudyHours",
        "Attendance",
        "AssignmentScore",
        "SleepHours",
        "TutoringHours",
        "ParentalSupport",
        "StressLevel",
    ]
    for key in required_keys:
        if key not in raw_features:
            raise ValueError(f"Missing required value: {key}")

    features = pd.DataFrame(
        [[
            float(raw_features["StudyHours"]),
            float(raw_features["Attendance"]),
            float(raw_features["AssignmentScore"]),
            float(raw_features["SleepHours"]),
            float(raw_features["TutoringHours"]),
            float(raw_features["ParentalSupport"]),
            float(raw_features["StressLevel"]),
        ]],
        columns=[
            "StudyHours",
            "Attendance",
            "AssignmentScore",
            "SleepHours",
            "TutoringHours",
            "ParentalSupport",
            "StressLevel",
        ],
    )

    model = load_model()
    probability = model.predict_proba(features)[0]
    predicted_index = int(np.argmax(probability))
    predicted_label = model.classes_[predicted_index]
    confidence = float(probability[predicted_index]) * 100

    return {
        "label": predicted_label,
        "confidence": round(confidence, 2),
        "probabilities": {
            label: round(float(p), 4) for label, p in zip(model.classes_, probability)
        },
    }


if __name__ == "__main__":
    model = train_model()
    save_model(model, MODEL_PATH)
