#!/usr/bin/env python3
"""Прогнозиране на качеството на вино чрез методи за класификация.

Reproducible pipeline for the UCI Wine Quality dataset (Cortez et al., 2009):
preprocessing, stratified split, feature selection, cross-validated
hyperparameter search, and comparison of Decision Tree, k-NN and Random Forest.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_FEATURES = 8
CLASS_ORDER = ["ниско", "средно", "високо"]
FEATURE_LABELS = {
    "fixed acidity": "фиксирана киселинност",
    "volatile acidity": "летлива киселинност",
    "citric acid": "лимонена киселина",
    "residual sugar": "остатъчна захар",
    "chlorides": "хлориди",
    "free sulfur dioxide": "свободен серен диоксид",
    "total sulfur dioxide": "общ серен диоксид",
    "density": "плътност",
    "pH": "pH",
    "sulphates": "сулфати",
    "alcohol": "алкохол",
    "is_red": "червено вино",
}


def quality_to_class(quality: int) -> str:
    """Map the 0–10 expert score to three ordinal classes.

    Original scores are heavily imbalanced (almost all wines are 5 or 6),
    so a 10-way classifier would be dominated by the majority grades.
    Three classes keep the problem supervised and multi-class while remaining
    learnable: low (≤4), medium (5–6), high (≥7).
    """
    if quality <= 4:
        return "ниско"
    if quality <= 6:
        return "средно"
    return "високо"


def load_wines() -> pd.DataFrame:
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red["is_red"] = 1
    white["is_red"] = 0
    frame = pd.concat([red, white], ignore_index=True)
    frame["quality_class"] = frame["quality"].map(quality_to_class)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"quality", "quality_class"}]


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["quality_class"].value_counts().reindex(CLASS_ORDER)
    original = frame["quality"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(original.index.astype(str), original.values, color="#3b6ea5")
    axes[0].set_title("Оригинална оценка за качество (0–10)")
    axes[0].set_xlabel("Оценка от експерти")
    axes[0].set_ylabel("Брой проби")

    axes[1].bar(counts.index, counts.values, color=["#b4533a", "#c4a35a", "#3d7a54"])
    axes[1].set_title("Целеви класове за класификация")
    axes[1].set_xlabel("Клас")
    axes[1].set_ylabel("Брой проби")
    savefig("class_distribution.png")

    return {k: int(v) for k, v in counts.items()}


def plot_correlation(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    corr = frame[features].corr(numeric_only=True)
    plt.figure(figsize=(10.5, 8.5))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=False,
        linewidths=0.3,
    )
    plt.title("Корелация между физикохимичните признаци")
    savefig("correlation_heatmap.png")
    return corr


def plot_key_features(frame: pd.DataFrame) -> None:
    keys = ["alcohol", "volatile acidity", "sulphates", "density"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, col in zip(axes, keys):
        sns.boxplot(
            data=frame,
            x="quality_class",
            y=col,
            order=CLASS_ORDER,
            ax=ax,
            palette=["#b4533a", "#c4a35a", "#3d7a54"],
            hue="quality_class",
            legend=False,
        )
        ax.set_title(FEATURE_LABELS[col])
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Разпределение на ключови признаци по клас", y=1.03)
    savefig("feature_boxplots.png")


def select_features(X_train: pd.DataFrame, y_train: np.ndarray, k: int) -> dict:
    mi = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    f_scores, _ = f_classif(X_train, y_train)
    ranking = pd.DataFrame(
        {
            "feature": X_train.columns,
            "feature_bg": [FEATURE_LABELS.get(c, c) for c in X_train.columns],
            "mutual_info": mi,
            "anova_f": f_scores,
        }
    ).sort_values("mutual_info", ascending=False)

    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.get_support()].tolist()

    plt.figure(figsize=(8.5, 4.8))
    plot_df = ranking.sort_values("mutual_info")
    plt.barh(plot_df["feature_bg"], plot_df["mutual_info"], color="#3b6ea5")
    plt.xlabel("Взаимна информация с класа")
    plt.title("Ранжиране на признаците (mutual information)")
    savefig("feature_ranking.png")

    return {
        "ranking": ranking.round(4).to_dict(orient="records"),
        "selected_kbest": selected,
        "k": k,
    }


def metrics_dict(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
    }


def fit_search(estimator, param_grid, X_train, y_train, cv) -> GridSearchCV:
    search = GridSearchCV(
        estimator,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(name, estimator, X_train, y_train, X_test, y_test, labels) -> dict:
    y_pred = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "name": name,
        "metrics": metrics_dict(y_test, y_pred),
        "confusion_matrix": cm.tolist(),
        "train_score_f1_macro": round(
            float(f1_score(y_train, estimator.predict(X_train), average="macro")), 4
        ),
        "classification_report": {
            cls: {
                k: round(float(v), 4)
                for k, v in report[cls].items()
                if k in {"precision", "recall", "f1-score", "support"}
            }
            for cls in CLASS_ORDER
        },
    }


def plot_confusion_matrices(results: list[dict], labels: list[str]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8))
    if len(results) == 1:
        axes = [axes]
    for ax, item in zip(axes, results):
        cm = np.array(item["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
        )
        ax.set_title(item["name"])
        ax.set_xlabel("Предсказан клас")
        ax.set_ylabel("Истински клас")
    fig.suptitle("Матрици на объркване (тестова извадка)", y=1.04)
    savefig("confusion_matrices.png")


def plot_model_comparison(rows: list[dict]) -> None:
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    pretty = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
    }
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    width = 0.18
    plt.figure(figsize=(10, 4.6))
    for i, metric in enumerate(metrics):
        values = [r["metrics"][metric] for r in rows]
        plt.bar(x + (i - 1.5) * width, values, width, label=pretty[metric])
    plt.xticks(x, names)
    plt.ylim(0, 1.05)
    plt.ylabel("Стойност")
    plt.title("Сравнение на моделите върху тестовата извадка")
    plt.legend(frameon=False, ncol=2)
    savefig("model_comparison.png")


def plot_forest_importance(model: RandomForestClassifier, columns: list[str]) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": columns,
            "feature_bg": [FEATURE_LABELS.get(c, c) for c in columns],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    plt.figure(figsize=(8.5, 4.8))
    plot_df = importance.sort_values("importance")
    plt.barh(plot_df["feature_bg"], plot_df["importance"], color="#3d7a54")
    plt.xlabel("Mean decrease in impurity")
    plt.title("Важност на признаците — Random Forest")
    savefig("random_forest_importance.png")
    return importance.round(4)


def dump_json(name: str, payload) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run() -> dict:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_wines()
    features = feature_columns(frame)
    class_counts = plot_class_distribution(frame)
    corr = plot_correlation(frame, features)
    plot_key_features(frame)

    X = frame[features]
    class_to_id = {name: idx for idx, name in enumerate(CLASS_ORDER)}
    y = frame["quality_class"].map(class_to_id).to_numpy()
    label_ids = list(range(len(CLASS_ORDER)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    selection = select_features(X_train, y_train, k=N_FEATURES)
    selected = selection["selected_kbest"]
    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    tree_search = fit_search(
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "max_depth": [4, 8, 12, None],
            "min_samples_split": [2, 10, 20],
            "criterion": ["gini", "entropy"],
        },
        X_train,
        y_train,
        cv,
    )
    knn_search = fit_search(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]
        ),
        {
            "clf__n_neighbors": [3, 5, 9, 15],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        X_train,
        y_train,
        cv,
    )
    forest_search = fit_search(
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", "log2"],
        },
        X_train,
        y_train,
        cv,
    )

    models = [
        ("Baseline (majority)", dummy),
        ("Decision Tree", tree_search.best_estimator_),
        ("k-NN", knn_search.best_estimator_),
        ("Random Forest", forest_search.best_estimator_),
    ]
    evaluations = [
        evaluate_model(name, est, X_train, y_train, X_test, y_test, label_ids)
        for name, est in models
    ]

    plot_confusion_matrices(
        [e for e in evaluations if not e["name"].startswith("Baseline")],
        labels=CLASS_ORDER,
    )
    plot_model_comparison(evaluations)

    forest: RandomForestClassifier = forest_search.best_estimator_
    importance = plot_forest_importance(forest, features)

    # Feature-selection ablation on the winning family (Random Forest defaults + best params).
    forest_sel = RandomForestClassifier(
        **{k: v for k, v in forest_search.best_params_.items()},
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    forest_sel.fit(X_train_sel, y_train)
    selected_eval = evaluate_model(
        "Random Forest (SelectKBest)",
        forest_sel,
        X_train_sel,
        y_train,
        X_test_sel,
        y_test,
        label_ids,
    )

    payload = {
        "dataset": {
            "source": "UCI Wine Quality (Cortez et al., 2009)",
            "n_red": int((frame["is_red"] == 1).sum()),
            "n_white": int((frame["is_red"] == 0).sum()),
            "n_total": int(len(frame)),
            "n_features": len(features),
            "missing_values": int(frame.isna().sum().sum()),
            "class_counts": class_counts,
            "original_quality_counts": {
                str(k): int(v) for k, v in frame["quality"].value_counts().sort_index().items()
            },
            "class_mapping": {
                "ниско": "quality ≤ 4",
                "средно": "quality 5–6",
                "високо": "quality ≥ 7",
            },
        },
        "split": {
            "test_size": TEST_SIZE,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "cv_folds": N_SPLITS,
            "cv_scoring": "f1_macro",
        },
        "feature_selection": selection,
        "best_params": {
            "Decision Tree": tree_search.best_params_,
            "k-NN": knn_search.best_params_,
            "Random Forest": forest_search.best_params_,
        },
        "cv_best_f1_macro": {
            "Decision Tree": round(float(tree_search.best_score_), 4),
            "k-NN": round(float(knn_search.best_score_), 4),
            "Random Forest": round(float(forest_search.best_score_), 4),
        },
        "test_results": evaluations,
        "selected_features_ablation": {
            "features": selected,
            "metrics": selected_eval["metrics"],
        },
        "top_correlated_pairs": (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .round(3)
            .rename_axis(["a", "b"])
            .reset_index(name="abs_corr")
            .to_dict(orient="records")
        ),
        "random_forest_importance": importance.to_dict(orient="records"),
    }
    dump_json("metrics.json", payload)
    pd.DataFrame(
        [
            {"model": e["name"], **e["metrics"], "cv_f1_macro": payload["cv_best_f1_macro"].get(e["name"])}
            for e in evaluations
        ]
    ).to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    importance.to_csv(RESULT_DIR / "feature_importance.csv", index=False)

    print("=== Wine quality classification ===")
    print(f"samples: {payload['dataset']['n_total']}  train: {payload['split']['n_train']}  test: {payload['split']['n_test']}")
    print("class counts:", class_counts)
    print("best params:", payload["best_params"])
    print("CV F1-macro:", payload["cv_best_f1_macro"])
    for e in evaluations:
        print(e["name"], e["metrics"])
    print("SelectKBest RF", selected_eval["metrics"])
    return payload


if __name__ == "__main__":
    run()
