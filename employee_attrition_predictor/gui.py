"""Tkinter desktop interface for Employee Attrition Predictor.

Run with: python gui.py (or simply python main.py)
The application and dataset run locally; no network connection is used.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd

from src.data_loader import (
    DATA_PATH, OUTPUT_DIR, DEPARTMENT_ROLES, FEATURE_COLUMNS,
    describe_dataset, load_dataset,
)
from src.evaluation import compare_models, save_confusion_matrix_plots, save_metrics_csv
from src.prediction import predict_employee, save_prediction_csv, WARNING_TEXT
from src.preprocessing import clean_data, data_quality_report, split_features_target
from src.train_models import split_data, train_models, tree_feature_importance


BG = "#f3f6fb"
PANEL = "#ffffff"
INK = "#17233c"
MUTED = "#65728a"
BLUE = "#3157d5"
GREEN = "#13866d"
RED = "#b93838"


class AttritionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Employee Attrition Predictor | DWM Mini Project")
        self.geometry("1120x760")
        self.minsize(930, 650)
        self.configure(bg=BG)
        self.models = None
        self.results = None
        self.cleaned_df = None
        self.original_df = None
        self._configure_style()
        self._build_header()
        self._build_tabs()
        self._load_initial_data()

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=INK, font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=INK, font=("Segoe UI", 12, "bold"))
        style.configure("Metric.TLabel", background=PANEL, foreground=BLUE, font=("Segoe UI", 22, "bold"))
        style.configure("Small.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("Accent.TButton", background=BLUE, foreground="white", padding=(14, 9), font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#2445b4"), ("disabled", "#aab6df")])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", BLUE)])
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=27, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#e8edf7", foreground=INK)
        style.configure("TLabelframe", background=PANEL, foreground=INK)
        style.configure("TLabelframe.Label", background=PANEL, foreground=INK, font=("Segoe UI", 10, "bold"))

    def _build_header(self):
        header = tk.Frame(self, bg=BG, padx=24, pady=18)
        header.pack(fill="x")
        title = tk.Label(header, text="Employee Attrition Predictor", bg=BG, fg=INK,
                         font=("Segoe UI", 22, "bold"))
        title.pack(anchor="w")
        tk.Label(header, text="DATA MINING & WAREHOUSING  •  CLASSIFICATION MINI-PROJECT",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(3, 0))
        self.status_var = tk.StringVar(value="Ready · Running locally")
        tk.Label(header, textvariable=self.status_var, bg=BG, fg=GREEN,
                 font=("Segoe UI", 9, "bold")).pack(anchor="e")

    def _build_tabs(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=20, pady=(0, 14))
        self.dashboard_tab = ttk.Frame(self.tabs, padding=18)
        self.dataset_tab = ttk.Frame(self.tabs, padding=18)
        self.models_tab = ttk.Frame(self.tabs, padding=18)
        self.predict_tab = ttk.Frame(self.tabs, padding=18)
        self.tabs.add(self.dashboard_tab, text="Overview")
        self.tabs.add(self.dataset_tab, text="Dataset & Cleaning")
        self.tabs.add(self.models_tab, text="Train & Evaluate")
        self.tabs.add(self.predict_tab, text="Predict Employee")
        self._build_dashboard()
        self._build_dataset_tab()
        self._build_models_tab()
        self._build_predict_tab()
        footer = tk.Label(self, text="Educational demonstration only · Uses synthetic data · Predictions are not employment advice",
                          bg=BG, fg=MUTED, font=("Segoe UI", 9))
        footer.pack(side="bottom", pady=(0, 9))

    def _card(self, parent, title, value="—", note=""):
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        tk.Label(frame, text=title.upper(), bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        value_label = tk.Label(frame, text=value, bg=PANEL, fg=BLUE,
                               font=("Segoe UI", 23, "bold"))
        value_label.pack(anchor="w", pady=(7, 1))
        note_label = tk.Label(frame, text=note, bg=PANEL, fg=MUTED,
                              font=("Segoe UI", 9), wraplength=220, justify="left")
        note_label.pack(anchor="w")
        return frame, value_label, note_label

    def _build_dashboard(self):
        ttk.Label(self.dashboard_tab, text="Project overview", style="Title.TLabel").pack(anchor="w")
        ttk.Label(self.dashboard_tab, text="Explore the dataset, train two classifiers, then try a prediction.",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(4, 18))
        cards = ttk.Frame(self.dashboard_tab)
        cards.pack(fill="x")
        self.card_rows, self.card_missing, self.card_models, self.card_best = [], None, None, None
        for title, value, note in [
            ("Dataset records", "—", "Rows available before cleaning"),
            ("Missing values", "—", "Values handled by preprocessing"),
            ("Models", "2", "Decision Tree + Naive Bayes"),
            ("Best F1 score", "—", "Calculated after training"),
        ]:
            card, value_lbl, note_lbl = self._card(cards, title, value, note)
            card.pack(side="left", fill="both", expand=True, padx=(0, 10))
            if title == "Dataset records": self.card_rows = value_lbl
            elif title == "Missing values": self.card_missing = value_lbl
            elif title == "Models": self.card_models = value_lbl
            else: self.card_best = value_lbl
        actions = ttk.Frame(self.dashboard_tab)
        actions.pack(fill="x", pady=20)
        ttk.Button(actions, text="View dataset", command=lambda: self.tabs.select(self.dataset_tab)).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Train & compare models", style="Accent.TButton", command=self.train).pack(side="left", padx=8)
        ttk.Button(actions, text="Make a prediction", command=lambda: self.tabs.select(self.predict_tab)).pack(side="left", padx=8)
        info = ttk.Frame(self.dashboard_tab, style="Panel.TFrame", padding=18)
        info.pack(fill="both", expand=True)
        ttk.Label(info, text="What this project demonstrates", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))
        for line in [
            "01   Data preprocessing: missing values, duplicates and categorical encoding.",
            "02   Classification: Decision Tree and Naive Bayes models.",
            "03   Evaluation: accuracy, precision, recall, F1-score and confusion matrices.",
            "04   Prediction: enter employee attributes and save the result locally.",
        ]:
            tk.Label(info, text=line, bg=PANEL, fg=INK, font=("Segoe UI", 10), anchor="w").pack(fill="x", pady=5)
        tk.Label(info, text="Dataset note: the included records are synthetic and are only for learning/demo purposes.",
                 bg=PANEL, fg=RED, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(12, 0))

    def _build_dataset_tab(self):
        top = ttk.Frame(self.dataset_tab)
        top.pack(fill="x")
        ttk.Label(top, text="Dataset & preprocessing", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="Refresh dataset", command=self.refresh_dataset).pack(side="right")
        self.dataset_summary = tk.StringVar(value="Loading dataset…")
        ttk.Label(self.dataset_tab, textvariable=self.dataset_summary, style="Subtitle.TLabel").pack(anchor="w", pady=(8, 12))
        wrap = ttk.Frame(self.dataset_tab)
        wrap.pack(fill="both", expand=True)
        self.data_tree = self._make_tree(wrap, height=16)
        self.preprocessing_text = tk.Text(self.dataset_tab, height=5, wrap="word", bg=PANEL, fg=INK,
                                           relief="flat", font=("Segoe UI", 9), padx=10, pady=8)
        self.preprocessing_text.pack(fill="x", pady=(12, 0))
        self.preprocessing_text.insert("1.0", "Preprocessing: duplicate rows are removed; invalid target values are discarded; missing numeric values are imputed using the training median; categorical values are imputed and one-hot encoded inside the model pipeline.")
        self.preprocessing_text.configure(state="disabled")

    def _build_models_tab(self):
        top = ttk.Frame(self.models_tab)
        top.pack(fill="x")
        ttk.Label(top, text="Model evaluation", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="Train models", style="Accent.TButton", command=self.train).pack(side="right")
        ttk.Label(self.models_tab, text="Metrics are calculated from the held-out test set. Select a model row to view its confusion matrix.",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(7, 12))
        table_wrap = ttk.Frame(self.models_tab)
        table_wrap.pack(fill="x")
        cols = ("model", "accuracy", "precision", "recall", "f1_score")
        self.metrics_tree = ttk.Treeview(table_wrap, columns=cols, show="headings", height=3, selectmode="browse")
        for c, label, width in [("model", "Model", 210), ("accuracy", "Accuracy", 130), ("precision", "Precision", 130), ("recall", "Recall", 130), ("f1_score", "F1-score", 130)]:
            self.metrics_tree.heading(c, text=label)
            self.metrics_tree.column(c, width=width, anchor="center" if c != "model" else "w")
        self.metrics_tree.pack(fill="x")
        self.metrics_tree.bind("<<TreeviewSelect>>", self.show_confusion)
        self.cm_frame = ttk.Frame(self.models_tab, style="Panel.TFrame", padding=16)
        self.cm_frame.pack(fill="both", expand=True, pady=(16, 0))
        self.cm_title = tk.Label(self.cm_frame, text="Confusion matrix", bg=PANEL, fg=INK,
                                 font=("Segoe UI", 12, "bold"))
        self.cm_title.pack(anchor="w")
        self.cm_canvas = tk.Canvas(self.cm_frame, height=190, bg=PANEL, highlightthickness=0)
        self.cm_canvas.pack(fill="both", expand=True, pady=6)
        self.model_notes = tk.StringVar(value="Click Train models to generate evaluation results.")
        tk.Label(self.cm_frame, textvariable=self.model_notes, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9), justify="left", wraplength=800).pack(anchor="w")

    def _build_predict_tab(self):
        ttk.Label(self.predict_tab, text="Predict attrition", style="Title.TLabel").pack(anchor="w")
        ttk.Label(self.predict_tab, text="Enter an employee profile. Train the models first if you have not already.",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(5, 15))
        body = ttk.Frame(self.predict_tab)
        body.pack(fill="both", expand=True)
        form = ttk.Frame(body, style="Panel.TFrame", padding=18)
        form.pack(side="left", fill="both", expand=True, padx=(0, 12))
        result_panel = ttk.Frame(body, style="Panel.TFrame", padding=18)
        result_panel.pack(side="left", fill="both", expand=True)
        self.fields = {}
        field_specs = [
            ("Age", "Age", "spin", (18, 65)),
            ("Department", "Department", "combo", list(DEPARTMENT_ROLES.keys())),
            ("JobRole", "Job role", "combo", []),
            ("MonthlyIncome", "Monthly income (₹)", "spin", (1000, 100000)),
            ("JobSatisfaction", "Job satisfaction (1–4)", "spin", (1, 4)),
            ("OverTime", "Works overtime?", "combo", ["No", "Yes"]),
            ("YearsAtCompany", "Years at company", "spin", (0, 47)),
            ("DistanceFromHome", "Distance from home (km)", "spin", (0, 100)),
        ]
        for row, (key, label, kind, opts) in enumerate(field_specs):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 14))
            if kind == "spin":
                widget = ttk.Spinbox(form, from_=opts[0], to=opts[1], width=20)
                widget.set(str({"Age": 35, "MonthlyIncome": 5500, "JobSatisfaction": 3,
                                "YearsAtCompany": 5, "DistanceFromHome": 10}[key]))
            else:
                widget = ttk.Combobox(form, values=opts, state="readonly", width=22)
                if key == "Department": widget.set("Research")
                elif key == "OverTime": widget.set("No")
            widget.grid(row=row, column=1, sticky="ew", pady=6)
            self.fields[key] = widget
        self.fields["Department"].bind("<<ComboboxSelected>>", self.update_roles)
        self.update_roles()
        ttk.Label(form, text="Model to use").grid(row=8, column=0, sticky="w", pady=8, padx=(0, 14))
        self.pred_model = ttk.Combobox(form, values=["Decision Tree", "Naive Bayes"], state="readonly", width=22)
        self.pred_model.set("Decision Tree")
        self.pred_model.grid(row=8, column=1, sticky="ew", pady=8)
        ttk.Button(form, text="Predict attrition", style="Accent.TButton", command=self.predict).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        form.columnconfigure(1, weight=1)
        ttk.Label(result_panel, text="Prediction result", style="PanelTitle.TLabel").pack(anchor="w")
        self.pred_badge = tk.Label(result_panel, text="Ready", bg="#e8edf7", fg=BLUE,
                                   font=("Segoe UI", 20, "bold"), padx=20, pady=18)
        self.pred_badge.pack(fill="x", pady=(16, 12))
        self.pred_detail = tk.StringVar(value="Complete the employee form and click Predict attrition.")
        tk.Label(result_panel, textvariable=self.pred_detail, bg=PANEL, fg=INK,
                 font=("Segoe UI", 10), wraplength=360, justify="left").pack(anchor="w", pady=6)
        tk.Label(result_panel, text="Important: synthetic-data prediction only. Do not use it to make real employment decisions.",
                 bg=PANEL, fg=RED, font=("Segoe UI", 9, "bold"), wraplength=360, justify="left").pack(anchor="w", pady=(20, 0))

    def _make_tree(self, parent, height=10):
        holder = ttk.Frame(parent)
        holder.pack(fill="both", expand=True)
        tree = ttk.Treeview(holder, show="headings", height=height)
        ybar = ttk.Scrollbar(holder, orient="vertical", command=tree.yview)
        xbar = ttk.Scrollbar(holder, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        holder.rowconfigure(0, weight=1); holder.columnconfigure(0, weight=1)
        return tree

    def _load_initial_data(self):
        try:
            self.refresh_dataset()
        except Exception as exc:
            messagebox.showerror("Dataset error", str(exc))

    def refresh_dataset(self):
        self.original_df = load_dataset()
        report = data_quality_report(self.original_df)
        self.cleaned_df = clean_data(self.original_df)
        self.dataset_summary.set(
            f"{len(self.original_df):,} rows × {len(self.original_df.columns)} columns  ·  "
            f"{report['total_missing']} missing values  ·  {report['duplicate_rows']} duplicate rows  ·  "
            f"{len(self.cleaned_df):,} rows after basic cleaning"
        )
        for item in self.data_tree.get_children(): self.data_tree.delete(item)
        columns = list(self.original_df.columns)
        self.data_tree.configure(columns=columns)
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=125, minwidth=90, anchor="center")
        preview = self.original_df.head(100).copy().astype(object).where(pd.notna(self.original_df.head(100)), "—")
        for row in preview.itertuples(index=False, name=None):
            self.data_tree.insert("", "end", values=tuple(row))
        self.card_rows.configure(text=f"{len(self.original_df):,}")
        self.card_missing.configure(text=str(report["total_missing"]))
        self.status_var.set("Dataset loaded · Local mode")

    def train(self):
        try:
            self.status_var.set("Training models…")
            self.update_idletasks()
            df = clean_data(load_dataset())
            X, y = split_features_target(df)
            X_train, X_test, y_train, y_test = split_data(X, y)
            self.models = train_models(X_train, y_train)
            self.results = compare_models(self.models, X_test, y_test, X_train, y_train)
            metrics_path = save_metrics_csv(self.results)
            plot_paths = save_confusion_matrix_plots(self.results)
            self.cleaned_df = df
            for item in self.metrics_tree.get_children(): self.metrics_tree.delete(item)
            for _, row in self.results.iterrows():
                self.metrics_tree.insert("", "end", iid=str(row["model"]), values=(
                    row["model"], f"{row['accuracy']:.3f}", f"{row['precision']:.3f}",
                    f"{row['recall']:.3f}", f"{row['f1_score']:.3f}"))
            best_idx = self.results["f1_score"].astype(float).idxmax()
            best = self.results.loc[best_idx]
            self.card_best.configure(text=f"{float(best['f1_score']):.3f}")
            self.model_notes.set(f"Best F1 on this test split: {best['model']}. Metrics saved to {metrics_path}. "
                                 f"Confusion-matrix plots saved: {len(plot_paths)}. Dataset is synthetic; scores are for demonstration only.")
            first = self.metrics_tree.get_children()
            if first:
                self.metrics_tree.selection_set(first[0])
                self.show_confusion()
            self.status_var.set("Models trained · Results saved locally")
            self.tabs.select(self.models_tab)
        except Exception as exc:
            self.status_var.set("Training failed")
            messagebox.showerror("Could not train models", str(exc))

    def show_confusion(self, _event=None):
        if self.results is None: return
        selection = self.metrics_tree.selection()
        if not selection: return
        model_name = selection[0]
        row = self.results[self.results["model"] == model_name].iloc[0]
        tn, fp, fn, tp = [int(row[k]) for k in ["true_negative", "false_positive", "false_negative", "true_positive"]]
        self.cm_title.configure(text=f"Confusion matrix · {model_name}")
        self.cm_canvas.delete("all")
        self.cm_canvas.update_idletasks()
        w = max(self.cm_canvas.winfo_width(), 500)
        cell = 86
        x0 = (w - cell * 2) // 2 + 35
        y0 = 36
        self.cm_canvas.create_text(x0 + cell, 12, text="Predicted", fill=MUTED, font=("Segoe UI", 9, "bold"))
        self.cm_canvas.create_text(x0 + cell / 2, y0 - 10, text="No", fill=INK, font=("Segoe UI", 9))
        self.cm_canvas.create_text(x0 + cell * 1.5, y0 - 10, text="Yes", fill=INK, font=("Segoe UI", 9))
        self.cm_canvas.create_text(x0 - 28, y0 + cell, text="Actual", angle=90, fill=MUTED, font=("Segoe UI", 9, "bold"))
        self.cm_canvas.create_text(x0 - 5, y0 + cell / 2, text="No", anchor="e", fill=INK, font=("Segoe UI", 9))
        self.cm_canvas.create_text(x0 - 5, y0 + cell * 1.5, text="Yes", anchor="e", fill=INK, font=("Segoe UI", 9))
        values = [[tn, fp], [fn, tp]]
        max_val = max(max(r) for r in values) or 1
        for i in range(2):
            for j in range(2):
                val = values[i][j]
                intensity = val / max_val
                fill = "#dce7ff" if intensity < .55 else "#a9c1ff"
                x1, y1 = x0 + j * cell, y0 + i * cell
                self.cm_canvas.create_rectangle(x1, y1, x1 + cell, y1 + cell, fill=fill, outline="white", width=3)
                self.cm_canvas.create_text(x1 + cell / 2, y1 + cell / 2, text=str(val), fill=INK,
                                           font=("Segoe UI", 16, "bold"))

    def update_roles(self, _event=None):
        dept = self.fields["Department"].get()
        roles = DEPARTMENT_ROLES.get(dept, [])
        self.fields["JobRole"]["values"] = roles
        if self.fields["JobRole"].get() not in roles:
            self.fields["JobRole"].set(roles[0] if roles else "")

    def predict(self):
        if self.models is None:
            messagebox.showinfo("Train models first", "Please train and compare the models first. The app will do that now.")
            self.train()
            if self.models is None: return
        try:
            info = {key: widget.get().strip() for key, widget in self.fields.items()}
            model_name = self.pred_model.get()
            result = predict_employee(self.models[model_name], info)
            saved = save_prediction_csv(result, model_name)
            pred = result["Prediction"]
            self.pred_badge.configure(text=f"Predicted attrition: {pred}",
                                      bg="#fdeaea" if pred == "Yes" else "#e1f5ed",
                                      fg=RED if pred == "Yes" else GREEN)
            probability = result.get("ProbabilityYes")
            prob_text = "Not available" if probability is None else f"{probability * 100:.1f}%"
            self.pred_detail.set(
                f"Model: {model_name}\nEstimated probability of Attrition = Yes: {prob_text}\n\n"
                f"Prediction saved to:\n{saved}\n\nThis is an estimate from synthetic training data, not a certainty."
            )
            self.status_var.set("Prediction saved locally")
        except (ValueError, KeyError) as exc:
            messagebox.showwarning("Check employee details", str(exc))
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))


def run_app():
    app = AttritionApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
