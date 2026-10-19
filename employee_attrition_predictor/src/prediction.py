"""Predict attrition for a single employee entered by the user."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_loader import DEPARTMENT_ROLES, FEATURE_COLUMNS, OUTPUT_DIR

WARNING_TEXT = (
    "WARNING: This tool is for EDUCATIONAL analysis only. It is trained on synthetic data\n"
    "and its predictions are estimates, not certainties. Never use it to make hiring,\n"
    "firing, promotion or other consequential employment decisions."
)


def _match_choice(value, choices, field):
    """Match text to one of the allowed choices, ignoring upper/lower case."""
    text = str(value).strip().lower()
    for choice in choices:
        if text == choice.lower():
            return choice
    raise ValueError(f"{field} must be one of: {', '.join(choices)}")


def _to_number(value, field, low, high, integer=True):
    try:
        number = float(str(value).strip())
    except ValueError:
        raise ValueError(f"{field} must be a number.") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be a finite number.")
    if integer and number != int(number):
        raise ValueError(f"{field} must be a whole number.")
    if not low <= number <= high:
        raise ValueError(f"{field} must be between {low} and {high}.")
    return int(number) if integer else number


def validate_employee_input(info):
    """Check raw user input and return a clean dictionary. Raises ValueError if invalid."""
    missing = [c for c in FEATURE_COLUMNS if c not in info]
    if missing:
        raise ValueError(f"Missing field(s): {missing}")

    clean = {}
    clean["Age"] = _to_number(info["Age"], "Age", 18, 65)
    clean["Department"] = _match_choice(info["Department"], list(DEPARTMENT_ROLES), "Department")
    clean["JobRole"] = _match_choice(info["JobRole"], DEPARTMENT_ROLES[clean["Department"]],
                                     f"JobRole for {clean['Department']}")
    clean["MonthlyIncome"] = _to_number(info["MonthlyIncome"], "MonthlyIncome", 1000, 100000)
    clean["JobSatisfaction"] = _to_number(info["JobSatisfaction"], "JobSatisfaction", 1, 4)
    clean["OverTime"] = _match_choice(info["OverTime"], ["Yes", "No"], "OverTime")
    clean["YearsAtCompany"] = _to_number(info["YearsAtCompany"], "YearsAtCompany", 0, 47)
    clean["DistanceFromHome"] = _to_number(info["DistanceFromHome"], "DistanceFromHome", 0, 100)

    if clean["YearsAtCompany"] > clean["Age"] - 16:
        raise ValueError("YearsAtCompany is too large for this Age.")
    return clean


def predict_employee(model, info):
    """Predict attrition for one employee. Returns a dictionary with the result."""
    clean = validate_employee_input(info)
    row = pd.DataFrame([clean], columns=FEATURE_COLUMNS)
    label = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])  # probability of 'Yes'
    return {
        **clean,
        "Prediction": "Yes" if label == 1 else "No",
        "ProbabilityYes": None if probability is None else round(probability, 4),
    }


def format_prediction(result):
    """Human-readable text for one prediction."""
    lines = [f"\nPredicted attrition: {result['Prediction']}"]
    if result["ProbabilityYes"] is not None:
        lines.append(f"Estimated probability of leaving: {result['ProbabilityYes'] * 100:.1f}%")
    lines.append("This is only an estimate from a simple model trained on synthetic data - not a certainty.")
    return "\n".join(lines)


def save_prediction_csv(result, model_name, output_dir=OUTPUT_DIR):
    """Append one prediction to outputs/predictions.csv (file and folder created if needed)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predictions.csv"
    record = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Model": model_name, **result}
    pd.DataFrame([record]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def _ask(prompt, validator, input_func):
    """Keep asking until the answer is valid."""
    while True:
        answer = input_func(prompt)
        try:
            return validator(answer)
        except ValueError as exc:
            print(f"  Invalid input: {exc}")


def prompt_employee_details(input_func=input):
    """Ask the user for each attribute (re-asks on invalid input) and return a clean dict."""
    print("\nEnter employee details:")
    info = {}
    info["Age"] = _ask("  Age (18-65): ",
                       lambda v: _to_number(v, "Age", 18, 65), input_func)
    info["Department"] = _ask(f"  Department ({'/'.join(DEPARTMENT_ROLES)}): ",
                              lambda v: _match_choice(v, list(DEPARTMENT_ROLES), "Department"), input_func)
    roles = DEPARTMENT_ROLES[info["Department"]]
    info["JobRole"] = _ask(f"  Job role ({'/'.join(roles)}): ",
                           lambda v: _match_choice(v, roles, "JobRole"), input_func)
    info["MonthlyIncome"] = _ask("  Monthly income in rupees (1000-100000): ",
                                 lambda v: _to_number(v, "MonthlyIncome", 1000, 100000), input_func)
    info["JobSatisfaction"] = _ask("  Job satisfaction (1=low ... 4=high): ",
                                   lambda v: _to_number(v, "JobSatisfaction", 1, 4), input_func)
    info["OverTime"] = _ask("  Works overtime? (Yes/No): ",
                            lambda v: _match_choice(v, ["Yes", "No"], "OverTime"), input_func)
    info["YearsAtCompany"] = _ask("  Years at company (0-47): ",
                                  lambda v: _to_number(v, "YearsAtCompany", 0, 47), input_func)
    info["DistanceFromHome"] = _ask("  Distance from home in km (0-100): ",
                                    lambda v: _to_number(v, "DistanceFromHome", 0, 100), input_func)
    return info
