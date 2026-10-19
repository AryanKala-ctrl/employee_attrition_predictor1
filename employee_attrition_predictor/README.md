# Employee Attrition Predictor Using Data Mining

A small Data Warehousing and Mining (DWM) mini-project. It cleans an employee dataset, trains a
**Decision Tree** and a **Naive Bayes** classifier, compares them, and predicts whether an employee
might leave the company.

> **Educational use only.** The dataset is **synthetic** (computer-generated, not real people).
> Results on it do **not** show how the models would perform on real data. Predictions are
> estimates, not certainties, and must never be used for hiring, firing, promotion or other
> consequential employment decisions.

Runs fully offline on Windows after the one-time `pip install`. No API keys, cloud, or database.

## Folder structure

```
employee_attrition_predictor/
    main.py                  launches the desktop GUI by default; also supports CLI/demo modes
    gui.py                   Tkinter desktop interface (Overview, Dataset, Evaluation, Prediction)
    run_app.bat              Windows double-click launcher after setup
    data/employee_attrition.csv   synthetic dataset (1005 rows)
    src/
        data_loader.py       generate / save / load / describe the dataset
        preprocessing.py     cleaning, missing values, encoding (inside a Pipeline)
        train_models.py      stratified split, Decision Tree, Naive Bayes
        evaluation.py        accuracy, precision, recall, F1, confusion matrix, CSV + plots
        prediction.py        input validation and single-employee prediction
    outputs/                 created results (metrics CSV, predictions CSV, plots)
    tests/test_project.py    21 automated tests (unittest)
    requirements.txt
    README.md  report.md  viva_prep.md
```

## Setup and run (Windows PowerShell)

Open PowerShell **inside the project folder** (the folder containing `main.py`).

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it (if blocked, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies (needs internet this one time)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the desktop application
python main.py

#    Optional: use the original command-line menu
python main.py --cli

#    or non-interactive demo: overview + training + comparison
python main.py --demo

# 5. Run the tests
python -m unittest -v tests.test_project
#    Optional, if pytest is installed: python -m pytest -v

# 6. Regenerate the dataset (same seed = same data every time)
python main.py --regenerate
```

The desktop application has four tabs: **Overview**, **Dataset & Cleaning**, **Train & Evaluate**, and **Predict Employee**. After setup, you can also double-click `run_app.bat` to launch it without opening VS Code. To use the older terminal menu, run `python main.py --cli`.

Outputs are written to `outputs/`: `model_metrics.csv`, `predictions.csv` (one row appended per
prediction), `confusion_matrix_decision_tree.png`, `confusion_matrix_naive_bayes.png`.

## Desktop application walkthrough

1. **Overview:** see dataset size and missing-value count.
2. **Dataset & Cleaning:** inspect the dataset preview and the preprocessing steps.
3. **Train & Evaluate:** train Decision Tree and Naive Bayes, compare metrics, and inspect a confusion matrix.
4. **Predict Employee:** enter a sample employee profile and save the prediction to `outputs/predictions.csv`.

The application is a local Tkinter desktop program. It does not require a browser or network connection once dependencies are installed.

## Dataset attributes (all synthetic)

| Attribute | Type | Meaning |
|---|---|---|
| Age | numeric (int) | Employee age in years (18-60) |
| Department | categorical | Sales, Research, HR, Finance |
| JobRole | categorical | Role inside the department (e.g. Recruiter, Software Engineer) |
| MonthlyIncome | numeric | Monthly salary in rupees (~3% missing) |
| JobSatisfaction | ordinal (1-4) | 1 = low ... 4 = high (~3% missing) |
| OverTime | categorical (binary) | Does the employee work overtime? Yes / No |
| YearsAtCompany | numeric (int) | Years worked at the company |
| DistanceFromHome | numeric (int) | Distance from home to office in km |
| **Attrition** | target (binary) | Did the employee leave? Yes / No (about 18% Yes) |

The generator makes attrition a *probability* that depends on overtime, satisfaction, income, age,
tenure and distance, plus random noise. It is not a fixed rule, so the models cannot be perfect.
It also deliberately includes missing values and 5 duplicate rows so the cleaning step has work to do.

## Preprocessing in simple words

1. **Detect** missing values and duplicate rows.
2. **Clean**: remove duplicates, trim spaces, drop rows with an invalid target.
3. **Fill missing values**: numbers get the median, text gets the most frequent value.
4. **Transform**: numbers are standardised; text categories become 0/1 columns (one-hot encoding).
5. **Split** features (X) from the target (y, Yes=1/No=0), then do a stratified 80/20 train/test split (seed 42).
6. **No data leakage**: the fill values, scaling and encoding are learned from the *training* data
   only (they sit inside a scikit-learn `Pipeline`) and are then just applied to the test data.

**Class imbalance:** only ~18% of rows are "Yes". We use a stratified split, train both models with
balanced sample weights, and report precision, recall and F1 (not only accuracy).

## DWM syllabus connection

**Unit II - Data Mining Introduction**
- *KDD*: selection (load CSV) -> cleaning -> transformation -> mining (classification) -> evaluation -> using the knowledge (prediction).
- *Data cleaning*: duplicates, missing values, invalid target values (`preprocessing.py`).
- *Attribute types*: numeric (Age, MonthlyIncome...), nominal (Department, JobRole), binary (OverTime, Attrition), ordinal (JobSatisfaction).
- *Data transformation*: standardisation and one-hot encoding.

**Unit IV - Classification**
- *Decision Tree induction*: `DecisionTreeClassifier` (max depth 5 to limit overfitting).
- *Bayesian classification*: `GaussianNB`, based on Bayes' theorem.
- *Training and testing*: stratified split; model learns from train, judged on unseen test data.
- *Metrics and confusion matrix*: accuracy, precision, recall, F1, TP/TN/FP/FN.
- *Model comparison*: side-by-side table printed and saved to CSV.

(Clustering is not implemented; the project keeps to the classification core.)

**Unit VI - WEKA** (optional; the Python app does not need WEKA)

1. Open WEKA -> **Explorer** -> **Preprocess** tab -> **Open file...** -> choose
   `data/employee_attrition.csv` (set file type to *CSV data files* if it is not shown).
2. WEKA reads `Attrition` as text, so it may treat it as a string. Use the filter
   *unsupervised > attribute > StringToNominal* (attribute index 9) so it becomes nominal (Yes/No);
   `Department`, `JobRole`, `OverTime` may need the same. Missing values show as blanks - WEKA can
   fill them with the *ReplaceMissingValues* filter.
3. Go to the **Classify** tab -> **Choose** -> *trees > J48*.
4. Under *Test options* choose *Percentage split* 80% (or 10-fold cross-validation), make sure the
   class box shows `Attrition`, then click **Start**.
5. Read *Correctly Classified Instances*, the *Detailed Accuracy By Class* table (precision, recall,
   F-measure) and the *Confusion Matrix* in the output panel. Right-click the result -> *Visualize tree*.

**Important:** WEKA's J48 (C4.5) and WEKA's own split/pruning differ from scikit-learn's
`DecisionTreeClassifier`, and WEKA does not use our seed or balanced weights. Its numbers will
therefore **not match** the Python results. Treat them as a separate experiment and never present
one as the other. (Duplicates are not removed automatically in WEKA either.)

## Tests

`tests/test_project.py` has 21 tests covering dataset generation/loading (including missing file and
missing column errors), duplicate and missing-value handling, the stratified split, small-dataset
errors, model training, metric correctness, CSV/plot export, input validation and prediction output.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized | Install Python 3.9+ from python.org and tick **Add Python to PATH**; or try `py -m venv venv` and `py main.py`. |
| `Activate.ps1 cannot be loaded ... running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. Or use `cmd` and `venv\Scripts\activate.bat`. |
| `pip install` fails / no internet | Needs internet once. Retry; use `python -m pip install -r requirements.txt`; behind a proxy ask your network admin. |
| `ModuleNotFoundError: pandas` (or sklearn) | The virtual environment is not active (prompt should start with `(venv)`) - activate it and run `pip install -r requirements.txt`. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project folder that contains `main.py`, not from inside `src/` or `tests/`. |
| `Dataset not found` | Run `python main.py --regenerate` (it is also recreated automatically on start). |
| `The dataset is missing required column(s)` | Your CSV must contain all columns in the table above; regenerate it with `--regenerate`. |
| `sparse_output` error from OneHotEncoder | scikit-learn is too old: `pip install --upgrade scikit-learn` (need >= 1.2). |
| `pytest` not found | `pip install pytest`, or run `python -m unittest -v tests.test_project`. |
| Confusion-matrix plots not saved | Install matplotlib (`pip install matplotlib`); the rest of the program still works without it. |

## Known limitations

- Synthetic data: metrics only show the code works, not real-world accuracy.
- Gaussian Naive Bayes assumes numeric attributes are bell-shaped and independent; with one-hot columns that is only an approximation, and its probabilities tend to be over-confident.
- One train/test split on ~200 test rows gives noisy metrics; a different seed can change which model "wins".
