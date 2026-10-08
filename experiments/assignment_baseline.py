"""Train and use the two polynomial regression models for ML Assignment 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits


SEED = 42
FOLDS = 5
HOLDOUT_FRACTION = 0.20
FEATURES = {1: [f"x{i}" for i in range(1, 7)], 2: ["x1", "x2", "x3"]}


def candidate_specs(problem):
    """Use all supplied inputs across the permitted total-degree range.

    Under-determined ordinary-least-squares fits are skipped during evaluation.
    """
    return [(FEATURES[problem], d) for d in range(1, (10 if problem == 1 else 20) + 1)]


def make_model(degree):
    # Affine scaling preserves the class of polynomials of total degree <= d.
    # Each scaler is fitted inside each CV training fold, preventing leakage.
    return Pipeline([
        ("input_scale", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("term_scale", StandardScaler()),
        ("regression", LinearRegression()),
    ])


def read_dataset(path, expected_features, training=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    # Check before pandas can silently rename duplicate column headers.
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if len(set(header)) != len(header):
        raise ValueError(f"{path.name}: duplicate column names are not supported.")
    frame = pd.read_csv(path)
    required = list(expected_features) + (["y"] if training else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{path.name}: the dataset is empty.")
    for column in required:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{path.name}: {column} must be numeric.")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name}: {column} contains missing or infinite values.")
    # Test y, if present as a placeholder, and unrelated ID columns are ignored.
    # Test rows are never sorted or dropped.
    return frame


def detect_roll(data_dir, roll=None):
    data_dir = Path(data_dir)
    if roll is None:
        rolls = sorted({p.name[:-len("_train_var1.csv")]
                        for p in data_dir.glob("*_train_var1.csv")})
        if len(rolls) != 1:
            raise ValueError("Place one student's four CSVs in data/ or provide --roll YOUR_ROLL.")
        roll = rolls[0]
    if not re.fullmatch(r"[A-Za-z0-9-]+", roll):
        raise ValueError("The roll number must contain only letters, digits, or hyphens.")
    paths = {problem: {
        split: data_dir / f"{roll}_{split}_var{problem}.csv"
        for split in ("train", "test")
    } for problem in (1, 2)}
    missing = [str(p) for files in paths.values() for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Upload the four personalised datasets. Missing:\n" + "\n".join(missing))
    return roll, paths


def scores(y_true, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError("Model predictions contain non-finite values.")
    # R2 has no meaningful denominator for a constant target.
    r2 = None if np.ptp(np.asarray(y_true, dtype=float)) == 0 else float(r2_score(y_true, prediction))
    return {"mse": float(mean_squared_error(y_true, prediction)), "r2": r2}


def select_candidate(train, problem, cv):
    results = []
    smallest_fold = min(len(fit_idx) for fit_idx, _ in cv.split(train))
    y = train["y"]
    for features, degree in candidate_specs(problem):
        terms = math.comb(len(features) + degree, degree) - 1
        row = {"features": ",".join(features), "degree": degree,
               "terms": terms, "status": "ok"}
        # Require more observations than coefficients (including intercept).
        if terms + 1 >= smallest_fold:
            row["status"] = "skipped: too many coefficients for a CV training fold"
        else:
            try:
                outcome = cross_validate(
                    make_model(degree), train[features], y, cv=cv,
                    scoring={"mse": "neg_mean_squared_error"},
                    return_train_score=True, error_score="raise", n_jobs=1,
                )
                losses = -outcome["test_mse"]
                if not np.isfinite(losses).all():
                    raise ValueError("non-finite CV error")
                row.update({
                    "cv_mse": float(losses.mean()),
                    "cv_mse_std": float(losses.std(ddof=1)),
                    "cv_mse_se": float(losses.std(ddof=1) / math.sqrt(FOLDS)),
                    "train_mse": float(-outcome["train_mse"].mean()),
                })
                row.update({f"fold_{i + 1}_mse": float(value)
                            for i, value in enumerate(losses)})
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row["status"] = f"failed: {exc}"
        results.append(row)
    valid = [row for row in results if row["status"] == "ok"]
    if not valid:
        raise ValueError("No candidate could be fitted. Check dataset size and feature values.")
    best = min(valid, key=lambda row: row["cv_mse"])
    # A small numerical tolerance avoids choosing higher degrees because of
    # floating-point differences when the data follow an exact polynomial.
    tolerance = 1e-12 * max(float(np.var(y)), np.finfo(float).tiny)
    threshold = best["cv_mse"] + best["cv_mse_se"] + tolerance
    eligible = [row for row in valid if row["cv_mse"] <= threshold]
    chosen = min(eligible, key=lambda row: (row["terms"], row["degree"], row["features"]))
    for row in results:
        row["selected"] = row is chosen
    selection = {
        "rule": "fewest polynomial terms within one standard error of the minimum CV MSE",
        "minimum_cv_mse": best["cv_mse"],
        "one_se_threshold": threshold,
        "numerical_tolerance": tolerance,
    }
    return chosen, pd.DataFrame(results), selection


def diagnostic_plot(cv_results, chosen, y_true, prediction, path, problem):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(9.2, 6.0), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    cv_ax = fig.add_subplot(grid[0, :])
    valid = cv_results[cv_results["status"] == "ok"]
    floor = max(float(np.var(y_true)) * 1e-16, np.finfo(float).tiny)
    for feature_set, group in valid.groupby("features", sort=False):
        cv_ax.plot(group["degree"], np.maximum(group["cv_mse"], floor),
                   marker="o", markersize=3, label=feature_set)
    cv_ax.scatter(chosen["degree"], max(chosen["cv_mse"], floor),
                  marker="*", s=145, color="#a52a2a", zorder=5, label="Selected")
    cv_ax.set(title=f"var{problem}: cross-validation error by degree",
              xlabel="Total polynomial degree", ylabel="Mean CV MSE (log scale)", yscale="log")
    cv_ax.legend(fontsize=8, loc="best")
    cv_ax.grid(alpha=0.2)
    actual_ax = fig.add_subplot(grid[1, 0])
    actual_ax.scatter(y_true, prediction, s=13, alpha=0.65, color="#255e75")
    lo, hi = min(min(y_true), min(prediction)), max(max(y_true), max(prediction))
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    actual_ax.plot([lo, hi], [lo, hi], "--", color="#a52a2a", linewidth=1)
    actual_ax.set(title="Untouched holdout", xlabel="Actual y", ylabel="Predicted y")
    residual_ax = fig.add_subplot(grid[1, 1])
    residual_ax.scatter(prediction, np.asarray(y_true) - prediction,
                        s=13, alpha=0.65, color="#255e75")
    residual_ax.axhline(0, linestyle="--", color="#a52a2a", linewidth=1)
    residual_ax.set(title="Holdout residuals", xlabel="Predicted y", ylabel="Actual - predicted")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def save_predictions(prediction, path, expected_rows):
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (expected_rows,) or not np.isfinite(prediction).all():
        raise ValueError("Predictions must contain one finite value per test row.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": prediction}).to_csv(path, index=False)
    check = pd.read_csv(path)
    if list(check.columns) != ["y"] or len(check) != expected_rows:
        raise RuntimeError("Submission file format verification failed.")
    return path


def fit_problem(train, test, problem, roll, out_dir):
    out_dir = Path(out_dir)
    if len(train) < 30:
        raise ValueError(f"var{problem}: at least 30 training rows are required for this 5-fold workflow.")
    development, holdout = train_test_split(
        train, test_size=HOLDOUT_FRACTION, random_state=SEED, shuffle=True,
    )
    cv = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    chosen, comparisons, selection = select_candidate(development, problem, cv)
    features = chosen["features"].split(",")
    evaluation_model = make_model(chosen["degree"])
    evaluation_model.fit(development[features], development["y"])
    holdout_prediction = evaluation_model.predict(holdout[features])
    holdout_scores = scores(holdout["y"], holdout_prediction)
    baseline_scores = scores(holdout["y"], np.full(len(holdout), development["y"].mean()))

    # Degree and features are now fixed. Refit on ALL labelled rows only after
    # recording the holdout result. This final model produces the submission.
    final_model = make_model(chosen["degree"])
    final_model.fit(train[features], train["y"])
    prediction = final_model.predict(test[features])
    prediction_path = out_dir / f"{roll}_pred_var{problem}.csv"
    save_predictions(prediction, prediction_path, len(test))
    model_path = out_dir / f"var{problem}_model.joblib"
    joblib.dump({"pipeline": final_model, "features": features,
                 "degree": chosen["degree"], "problem": problem, "roll": roll}, model_path)
    comparisons.to_csv(out_dir / f"var{problem}_cv_results.csv", index=False)
    pd.DataFrame({"row_number_1_based": holdout.index.to_numpy() + 1,
                  "actual_y": holdout["y"].to_numpy(), "predicted_y": holdout_prediction,
                  "residual": holdout["y"].to_numpy() - holdout_prediction}).to_csv(
        out_dir / f"var{problem}_holdout_predictions.csv", index=False)
    diagnostic_plot(comparisons, chosen, holdout["y"].to_numpy(), holdout_prediction,
                    out_dir / f"var{problem}_diagnostics.png", problem)
    outside = ((test[features] < train[features].min()) |
               (test[features] > train[features].max())).any(axis=1)
    result = {
        "problem": problem, "features": features, "degree": chosen["degree"],
        "terms_excluding_intercept": chosen["terms"],
        "train_rows": len(train), "development_rows": len(development),
        "holdout_rows": len(holdout), "test_rows": len(test),
        "cv_mse": chosen["cv_mse"], "cv_mse_se": chosen["cv_mse_se"],
        "holdout": holdout_scores, "mean_baseline_holdout": baseline_scores,
        "selection": selection,
        "candidate_search": [{"features": f, "degree": d} for f, d in candidate_specs(problem)],
        "successful_candidates": int((comparisons["status"] == "ok").sum()),
        "skipped_or_failed_candidates": int((comparisons["status"] != "ok").sum()),
        "test_rows_outside_train_feature_ranges": int(outside.sum()),
        "final_fit_rows": len(train), "prediction_file": prediction_path.name,
        "model_file": model_path.name,
    }
    print(f"var{problem}: degree {chosen['degree']}, features {', '.join(features)}, "
          f"holdout MSE {holdout_scores['mse']:.6g}, R2 {holdout_scores['r2']}", flush=True)
    return result


def write_report(results, roll, student_name, out_dir):
    """Generate a three-page report using only this run's measured results."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    import reportlab
    from xml.sax.saxutils import escape

    out_dir = Path(out_dir)
    path = out_dir / f"{roll}_report.pdf"
    # Embed fonts distributed with ReportLab for consistent PDF rendering.
    font_dir = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("AssignmentSans", str(font_dir / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("AssignmentSans-Bold", str(font_dir / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("AssignmentSans", normal="AssignmentSans",
                                  bold="AssignmentSans-Bold", italic="AssignmentSans",
                                  boldItalic="AssignmentSans-Bold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], alignment=TA_LEFT,
                              fontName="AssignmentSans-Bold",
                              textColor=colors.HexColor("#8B1E2D"), fontSize=23, leading=27,
                              spaceAfter=13))
    styles.add(ParagraphStyle("ReportHeading", parent=styles["Heading2"],
                              fontName="AssignmentSans-Bold",
                              textColor=colors.HexColor("#8B1E2D"), fontSize=12,
                              leading=15, spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle("ReportBody", parent=styles["BodyText"], fontSize=10,
                              fontName="AssignmentSans",
                              leading=14, spaceAfter=8))
    story = []

    def paragraph(text, style="ReportBody"):
        story.append(Paragraph(text, styles[style]))

    def number(value):
        return "undefined (constant target)" if value is None else f"{value:.6g}"

    paragraph("Polynomial Regression", "ReportTitle")
    paragraph("ML Assignment 1 | " + escape(roll) +
              (" | " + escape(student_name) if student_name else ""))
    paragraph("Objective and data", "ReportHeading")
    paragraph("Two independent polynomial regression models predict the Net Power Score (var1) "
              "and Thermal Anomaly Score (var2). Each problem uses its own personalised train "
              "and test files. The target is y. IDs and any test target placeholder are excluded from fitting.")
    paragraph("Method", "ReportHeading")
    paragraph("For each problem, 20% of the labelled rows are reserved as an untouched random holdout. "
              "Five-fold shuffled cross-validation on the remaining 80% compares candidate feature sets "
              "and degrees. Both splits use seed 42. Each fold fits input standardisation, a full "
              "polynomial expansion, term standardisation, and ordinary least-squares regression in a "
              "single pipeline. All powers and interaction terms of total degree at most d are included. "
              "The intercept is fitted separately. No regularisation or non-polynomial estimator is used.")
    paragraph("Candidate search and degree selection", "ReportHeading")
    paragraph("For var1, the search covers degrees 1-10 using all six features. "
              "For var2, it covers degrees 1-20 using all three features. "
              "Candidates with as many "
              "coefficients as observations in a CV training fold, or more, are skipped.")
    paragraph("The selected candidate has the fewest polynomial terms among those within one standard "
              "error of the lowest mean CV MSE. Standard error is the sample standard deviation of the "
              "five fold MSEs divided by sqrt(5). This is a model-selection heuristic, not a confidence "
              "interval. A tolerance of 10^-12 times the development-target variance handles numerical "
              "ties. The holdout does not influence this choice.")
    paragraph("Evaluation and final fitting", "ReportHeading")
    paragraph("The selected model is fitted on the development split and evaluated once on the holdout "
              "using MSE and R-squared. A training-mean baseline provides context. The chosen degree "
              "and features are then fixed, and a fresh pipeline is trained on all labelled rows for "
              "test prediction. Holdout scores below describe the evaluation model; hidden test "
              "scores are unavailable. No observed data are removed or imputed.")
    paragraph("MSE = mean((y - prediction)^2). R-squared = 1 - SSE/SST. Lower MSE is better; "
              "R-squared closer to 1 is better. Negative R-squared is possible.")

    for result in results:
        problem = result["problem"]
        story.append(PageBreak())
        title = "Steam turbine optimisation" if problem == 1 else "Thermal reservoir mapping"
        paragraph(f"var{problem}: {title}", "ReportTitle")
        paragraph("Dataset and chosen model", "ReportHeading")
        paragraph(f"Labelled rows: {result['train_rows']:,}. Development: {result['development_rows']:,}. "
                  f"Holdout: {result['holdout_rows']:,}. Test: {result['test_rows']:,}.")
        paragraph(f"Selected degree: <b>{result['degree']}</b>. Features: "
                  f"<b>{escape(', '.join(result['features']))}</b>. "
                  f"Polynomial terms excluding intercept: {result['terms_excluding_intercept']}. "
                  f"Evaluated candidates: {result['successful_candidates']}. "
                  f"Skipped or failed: {result['skipped_or_failed_candidates']}.")
        paragraph("Measured performance", "ReportHeading")
        paragraph(f"Selected-model mean CV MSE: <b>{number(result['cv_mse'])}</b> "
                  f"(standard error {number(result['cv_mse_se'])}). "
                  f"Minimum candidate CV MSE: {number(result['selection']['minimum_cv_mse'])}. "
                  f"Selection threshold: {number(result['selection']['one_se_threshold'])}.")
        paragraph(f"Holdout MSE: <b>{number(result['holdout']['mse'])}</b>. "
                  f"Holdout R-squared: <b>{number(result['holdout']['r2'])}</b>. "
                  f"Training-mean baseline holdout MSE: {number(result['mean_baseline_holdout']['mse'])}.")
        story.append(Spacer(1, 4))
        story.append(Image(str(out_dir / f"var{problem}_diagnostics.png"),
                           width=6.65 * inch, height=4.34 * inch))
        paragraph("Interpretation and limitations", "ReportHeading")
        difference = result["mean_baseline_holdout"]["mse"] - result["holdout"]["mse"]
        paragraph(("The selected model improves on the mean baseline on this holdout. " if difference > 0
                   else "The selected model does not improve on the mean baseline on this holdout. ") +
                  "The charts show the cross-validation trade-off and holdout residuals; "
                  "they do not measure hidden test performance. " +
                  f"{result['test_rows_outside_train_feature_ranges']} test rows lie outside the "
                  "training range of at least one selected feature. This marginal-range check does not "
                  "establish whether a row lies inside the multivariate training distribution.")
        paragraph(f"Final model fit: {result['final_fit_rows']:,} labelled rows. Submission: "
                  f"{escape(result['prediction_file'])}, containing only the y column in original test-row order.")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("AssignmentSans", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(42, 25, f"ML Assignment 1 | {roll}")
        canvas.drawRightString(A4[0] - 42, 25, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=42, leftMargin=42,
                                 topMargin=38, bottomMargin=38,
                                 title=f"Polynomial Regression - {roll}", author=student_name)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def run_assignment(data_dir="data", roll=None, out_dir="outputs", student_name=""):
    roll, paths = detect_roll(data_dir, roll)
    # Validate both problems before generating any outputs.
    datasets = {problem: {
        split: read_dataset(path, FEATURES[problem], training=(split == "train"))
        for split, path in files.items()
    } for problem, files in paths.items()}
    out_dir = Path(out_dir)
    # A distinct run directory prevents old and new predictions being mixed.
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty. Choose a new --out directory for this run.")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with threadpool_limits(limits=1):
        for problem, data in datasets.items():
            print(f"Training var{problem}...", flush=True)
            results.append(fit_problem(data["train"], data["test"], problem, roll, out_dir))
    metadata = {
        "roll": roll, "student_name": student_name, "seed": SEED,
        "cv_folds": FOLDS, "holdout_fraction": HOLDOUT_FRACTION,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "matplotlib", "reportlab", "joblib")},
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for files in paths.values() for path in files.values()},
        "results": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    report = write_report(results, roll, student_name, out_dir)
    print(f"Saved predictions, models, diagnostics, metrics, and {report.name} in {out_dir}", flush=True)
    return metadata


def predict_saved(model_path, test_path, output_path):
    # Load only models you created or otherwise trust: joblib uses pickle.
    bundle = joblib.load(model_path)
    frame = read_dataset(test_path, bundle["features"])
    prediction = bundle["pipeline"].predict(frame[bundle["features"]])
    path = save_predictions(prediction, output_path, len(frame))
    print(f"Saved {len(frame)} predictions to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    train = actions.add_parser("train", help="Select, evaluate and fit both problems; create all outputs")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--roll", default="BT2024016", help="Student roll number (default: BT2024016)")
    train.add_argument("--name", default="", help="Optional student name printed in the PDF")
    train.add_argument("--out", default="outputs", help="A new or empty output directory")
    infer = actions.add_parser("predict", help="Predict using a saved model")
    infer.add_argument("--model", required=True)
    infer.add_argument("--test", required=True)
    infer.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "train":
        run_assignment(args.data_dir, args.roll, args.out, args.name)
    else:
        predict_saved(args.model, args.test, args.output)


if __name__ == "__main__":
    main()
