# -*- coding: utf-8 -*-
"""
Учебный Python-скрипт для проекта:
Определение мошеннических банковских операций.

Скрипт основан на логике Colab-ноутбука:
EDA -> подготовка данных -> Logistic Regression -> Decision Tree -> Random Forest -> сравнение моделей.

Запуск:
    python antifraud_pipeline.py --input creditcard.csv --output output
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ----------------------------- Вспомогательные функции -----------------------------

def create_run_dirs(output_dir: Path) -> dict:
    """
    Создает отдельную папку для каждого запуска анализа.

    Структура результата:
        output/
          runs/
            run_YYYYMMDD_HHMMSS/
              plots/
              tables/
              texts/

    Такой подход нужен, чтобы результаты разных запусков не перемешивались,
    а Java-интерфейс мог показать все графики именно текущего запуска.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / f"run_{timestamp}"

    # Если пользователь запустил анализ несколько раз в одну секунду,
    # добавим короткий индекс, чтобы не перезаписать предыдущий запуск.
    if run_dir.exists():
        i = 2
        while (output_dir / "runs" / f"run_{timestamp}_{i}").exists():
            i += 1
        run_dir = output_dir / "runs" / f"run_{timestamp}_{i}"

    paths = {
        "output_base": output_dir,
        "runs": output_dir / "runs",
        "root": run_dir,
        "plots": run_dir / "plots",
        "tables": run_dir / "tables",
        "texts": run_dir / "texts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    # Файл-указатель на последний запуск. Его читает Java-интерфейс.
    (output_dir / "latest_run.txt").write_text(str(run_dir.resolve()), encoding="utf-8")
    return paths


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_dataset(df: pd.DataFrame) -> None:
    required = {"Time", "Amount", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "В датасете отсутствуют обязательные столбцы: " + ", ".join(sorted(missing))
        )
    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Столбец Class должен содержать только значения 0 и 1.")


def evaluate_model(model, x_test, y_test, model_name: str, plots_dir: Path) -> dict:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Classification report": classification_report(y_test, y_pred, digits=4, zero_division=0),
    }

    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"]).plot(values_format="d")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(plots_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close("all")

    return metrics


def explain_metric_row(row: pd.Series) -> str:
    return (
        f"{row['Model']}: Recall={row['Recall']:.3f}, Precision={row['Precision']:.3f}, "
        f"F1={row['F1-score']:.3f}, ROC-AUC={row['ROC-AUC']:.3f}. "
        f"Модель нашла {int(row['TP'])} мошеннических операций и пропустила {int(row['FN'])}. "
        f"Ложных тревог: {int(row['FP'])}."
    )


def choose_best_model(results: pd.DataFrame) -> str:
    # Для учебной антифрод-задачи приоритет: Recall, затем F1, затем ROC-AUC.
    ranked = results.sort_values(by=["Recall", "F1-score", "ROC-AUC"], ascending=False)
    return str(ranked.iloc[0]["Model"])


# ----------------------------- Основной pipeline -----------------------------

def run_pipeline(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_dir = Path(output_path)
    paths = create_run_dirs(output_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")

    print("=== Учебный антифрод-анализ ===", flush=True)
    print(f"Входной файл: {input_file}", flush=True)
    print(f"Базовая папка результатов: {output_dir.resolve()}", flush=True)
    print(f"Папка текущего запуска: {paths['root'].resolve()}", flush=True)
    print(f"RUN_DIR={paths['root'].resolve()}", flush=True)

    # 1. Загрузка и проверка
    df = pd.read_csv(input_file)
    original_shape = df.shape
    validate_dataset(df)

    missing_before = int(df.isna().sum().sum())
    df = df.dropna()
    missing_after = int(df.isna().sum().sum())

    # 2. EDA
    class_counts = df["Class"].value_counts().sort_index()
    class_percent = df["Class"].value_counts(normalize=True).sort_index() * 100
    fraud_count = int(class_counts.get(1, 0))
    normal_count = int(class_counts.get(0, 0))
    fraud_share = float(df["Class"].mean())

    class_distribution = pd.DataFrame({
        "Class": ["0 - обычная операция", "1 - мошенническая операция"],
        "Count": [normal_count, fraud_count],
        "Percent": [float(class_percent.get(0, 0)), float(class_percent.get(1, 0))],
    })
    class_distribution.to_csv(paths["tables"] / "class_distribution.csv", index=False, encoding="utf-8-sig")

    # Логарифмическая нормализация Amount: уменьшаем влияние очень крупных операций.
    # log1p безопасен для нулевых сумм: log1p(0) = 0.
    df["Amount_Log"] = np.log1p(df["Amount"])

    df.describe().T.to_csv(paths["tables"] / "numeric_description.csv", encoding="utf-8-sig")
    df.groupby("Class")["Amount"].describe().to_csv(paths["tables"] / "amount_by_class.csv", encoding="utf-8-sig")

    corr_matrix = df.corr(numeric_only=True)
    corr_matrix.to_csv(paths["tables"] / "correlation_matrix.csv", encoding="utf-8-sig")

    corr = corr_matrix["Class"].sort_values(ascending=False)
    corr.to_frame("Correlation_with_Class").to_csv(paths["tables"] / "correlation_with_class.csv", encoding="utf-8-sig")

    # Графики EDA
    plt.figure(figsize=(7, 5))
    plt.bar(["Обычные", "Мошеннические"], [normal_count, fraud_count])
    plt.yscale("log")
    plt.title("Распределение классов, логарифмическая шкала")
    plt.ylabel("Количество операций")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "01_class_distribution.png", dpi=150)
    plt.close()

    # Гистограммы всех числовых признаков
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols].hist(figsize=(16, 12), bins=30)
    plt.suptitle("Распределение числовых признаков", fontsize=16)
    plt.tight_layout()
    plt.savefig(paths["plots"] / "02_all_numeric_histograms.png", dpi=150)
    plt.close("all")

    plt.figure(figsize=(8, 5))
    plt.hist(df["Amount"], bins=60)
    plt.title("Распределение суммы операций до логарифмирования")
    plt.xlabel("Amount")
    plt.ylabel("Количество операций")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "03_amount_distribution_raw.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["Amount"], bins=60)
    axes[0].set_title("Amount до логарифмирования")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Количество")
    axes[1].hist(df["Amount_Log"], bins=60)
    axes[1].set_title("Amount после log1p")
    axes[1].set_xlabel("Amount_Log")
    axes[1].set_ylabel("Количество")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "04_amount_log_normalization.png", dpi=150)
    plt.close("all")

    plt.figure(figsize=(7, 5))
    df.boxplot(column="Amount", by="Class")
    plt.title("Сумма операций по классам")
    plt.suptitle("")
    plt.xlabel("Class: 0 — обычная, 1 — мошенническая")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "05_amount_boxplot_by_class.png", dpi=150)
    plt.close("all")

    df_eda = df.copy()
    df_eda["Hour"] = (df_eda["Time"] // 3600) % 24

    # Абсолютное распределение по часам.
    # Оно показывает общее количество операций, но из-за дисбаланса классов
    # мошеннические операции могут быть визуально почти незаметны.
    plt.figure(figsize=(8, 5))
    plt.hist(df_eda[df_eda["Class"] == 0]["Hour"], bins=24, alpha=0.7, label="Обычные")
    plt.hist(df_eda[df_eda["Class"] == 1]["Hour"], bins=24, alpha=0.7, label="Мошеннические")
    plt.title("Распределение операций по часам")
    plt.xlabel("Час")
    plt.ylabel("Количество операций")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["plots"] / "06_time_distribution_by_hour.png", dpi=150)
    plt.close()

    # Нормализованное распределение по часам.
    # Для каждого класса сумма значений по 24 часам равна 100%.
    # Это позволяет сравнивать форму распределения обычных и мошеннических операций,
    # а не абсолютные количества, которые сильно различаются из-за дисбаланса.
    hour_class_counts = (
        df_eda.groupby(["Hour", "Class"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(24), fill_value=0)
    )
    for class_value in [0, 1]:
        if class_value not in hour_class_counts.columns:
            hour_class_counts[class_value] = 0
    hour_class_counts = hour_class_counts[[0, 1]]
    hour_class_percent = hour_class_counts.div(hour_class_counts.sum(axis=0), axis=1).fillna(0) * 100
    hour_class_percent.columns = ["Обычные операции, %", "Мошеннические операции, %"]
    hour_class_percent.index.name = "Hour"
    hour_class_percent.to_csv(paths["tables"] / "hour_distribution_normalized.csv", encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.plot(hour_class_percent.index, hour_class_percent["Обычные операции, %"], marker="o", label="Обычные операции")
    plt.plot(hour_class_percent.index, hour_class_percent["Мошеннические операции, %"], marker="o", label="Мошеннические операции")
    plt.title("Нормализованное распределение операций по часам")
    plt.xlabel("Час")
    plt.ylabel("Доля операций внутри класса, %")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["plots"] / "06b_time_distribution_normalized.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Матрица корреляций")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "07_full_correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 10))
    corr.drop("Class").sort_values().plot(kind="barh")
    plt.title("Корреляция признаков с Class")
    plt.xlabel("Коэффициент корреляции")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "08_correlation_with_class.png", dpi=150)
    plt.close()

    # 3. Подготовка данных
    # Для обучения используем Amount_Log вместо исходного Amount.
    # Это снижает влияние редких экстремально крупных операций.
    X = df.drop(columns=["Class", "Amount"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    columns_to_scale = [col for col in ["Time", "Amount_Log"] if col in X_train_scaled.columns]
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    # 4. Обучение моделей
    print("Обучение Logistic Regression...", flush=True)
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)

    print("Обучение Decision Tree...", flush=True)
    decision_tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE)
    decision_tree.fit(X_train_scaled, y_train)

    print("Обучение Random Forest...", flush=True)
    random_forest = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    random_forest.fit(X_train_scaled, y_train)

    models = {
        "Logistic Regression": log_reg,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }

    metrics_list = []
    for name, model in models.items():
        metrics_list.append(evaluate_model(model, X_test_scaled, y_test, name, paths["plots"]))

    results = pd.DataFrame(metrics_list)
    results_for_csv = results.drop(columns=["Classification report"])
    results_for_csv.to_csv(paths["tables"] / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # График сравнения моделей
    plot_df = results.set_index("Model")[["Precision", "Recall", "F1-score", "ROC-AUC"]]
    plot_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Сравнение моделей")
    plt.ylabel("Значение метрики")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "09_model_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, name=name, ax=ax)
    plt.title("ROC-кривые моделей")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "10_roc_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for name, model in models.items():
        PrecisionRecallDisplay.from_estimator(model, X_test_scaled, y_test, name=name, ax=ax)
    plt.title("Precision-Recall-кривые моделей")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "11_precision_recall_curves.png", dpi=150)
    plt.close()

    # Важность признаков и дерево
    feature_importance = pd.DataFrame({
        "Feature": X_train_scaled.columns,
        "Importance": random_forest.feature_importances_,
    }).sort_values(by="Importance", ascending=False)
    feature_importance.to_csv(paths["tables"] / "feature_importance_random_forest.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 7))
    top = feature_importance.head(15)
    plt.barh(top["Feature"][::-1], top["Importance"][::-1])
    plt.title("Топ-15 важных признаков Random Forest")
    plt.xlabel("Важность")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "12_feature_importance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(20, 8))
    plot_tree(
        decision_tree,
        feature_names=X_train_scaled.columns,
        class_names=["Normal", "Fraud"],
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Визуализация дерева решений")
    plt.tight_layout()
    plt.savefig(paths["plots"] / "13_decision_tree.png", dpi=150)
    plt.close()

    best_model_name = choose_best_model(results)
    best_model = models[best_model_name]

    fraud_probability = best_model.predict_proba(X_test_scaled)[:, 1]
    fraud_prediction = best_model.predict(X_test_scaled)
    predictions = X_test.copy()
    predictions["Actual_Class"] = y_test.values
    predictions["Predicted_Class"] = fraud_prediction
    predictions["Fraud_Probability"] = fraud_probability
    predictions.sort_values(by="Fraud_Probability", ascending=False).to_csv(
        paths["tables"] / "fraud_predictions.csv", index=False, encoding="utf-8-sig"
    )

    # 5. Текстовые интерпретации для Java-интерфейса
    top_corr_pos = corr.drop("Class").sort_values(ascending=False).head(5)
    top_corr_neg = corr.drop("Class").sort_values(ascending=True).head(5)
    top_features = feature_importance.head(10)

    summary_text = f"""ОБЩИЙ ОБЗОР ДАТАСЕТА

Файл: {input_file.name}
Дата запуска анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Размер исходного датасета: {original_shape[0]:,} строк и {original_shape[1]:,} столбцов.
Размер после удаления пропусков: {df.shape[0]:,} строк и {df.shape[1]:,} столбцов.
Количество пропущенных значений до очистки: {missing_before}.
Количество пропущенных значений после очистки: {missing_after}.

Целевая переменная Class:
0 — обычная операция;
1 — мошенническая операция.

Обычных операций: {normal_count:,}.
Мошеннических операций: {fraud_count:,}.
Доля мошеннических операций: {pct(fraud_share)}.

Также добавлен признак Amount_Log = log(1 + Amount), который используется вместо исходного Amount при обучении моделей.

Главный вывод: датасет сильно несбалансирован. Мошеннических операций очень мало по сравнению с обычными. Поэтому нельзя оценивать качество модели только по Accuracy: модель может почти всегда отвечать «обычная операция» и формально получать высокую точность, но не решать задачу антифрода.
"""

    eda_text = f"""ИНТЕРПРЕТАЦИЯ EDA

1. Типы признаков
В датасете используются числовые признаки. Столбцы V1–V28 уже преобразованы методом PCA и анонимизированы авторами набора данных. Категориальных признаков нет, поэтому Encoding не требуется.

2. Пропуски
После очистки пропущенных значений не осталось. Это значит, что данные можно использовать для обучения моделей без дополнительного заполнения пропусков.

3. Распределение Class
Мошеннические операции составляют только {pct(fraud_share)}. Это типичная ситуация для антифрод-задач: редкие события нужно находить среди огромного количества нормальных операций.

4. Распределения числовых признаков
Построены гистограммы всех числовых признаков. Они помогают увидеть асимметрию, выбросы и общий характер распределений. Большинство V-признаков уже преобразованы PCA, а Amount имеет заметно скошенное распределение.

5. Amount и логарифмическая нормализация
Сумма операции сама по себе не позволяет надежно отделить мошенничество от обычных операций. Мошеннические операции могут встречаться как среди маленьких, так и среди крупных сумм. Для Amount выполнено преобразование Amount_Log = log(1 + Amount). Оно уменьшает влияние очень крупных операций и делает распределение более удобным для обучения моделей.

6. Time и нормализация распределения по часам
Признак Time помогает проверить, есть ли временные закономерности. Сначала построено абсолютное распределение операций по часам. Затем построено нормализованное распределение: для каждого класса сумма долей по 24 часам равна 100%.

Это важно, потому что обычных операций намного больше, чем мошеннических. Нормализация позволяет сравнивать не количество операций, а форму распределения: в какие часы внутри каждого класса операции встречаются чаще или реже. Если линия мошеннических операций заметно отличается от линии обычных, значит время может быть полезным признаком для модели.

7. Матрица корреляций
Построена полная матрица корреляций между числовыми признаками. Она показывает, насколько признаки линейно связаны друг с другом. Для признаков после PCA обычно ожидается сниженная взаимная корреляция, что полезно для моделей.

8. Корреляции с Class
Самые положительно связанные с Class признаки:
{top_corr_pos.to_string()}

Самые отрицательно связанные с Class признаки:
{top_corr_neg.to_string()}

Корреляции помогают понять направление связи, но не заменяют обучение модели. Для антифрода важны сложные комбинации признаков.
"""

    preprocessing_text = f"""ПОДГОТОВКА ДАННЫХ

1. Признаки и целевая переменная
X содержит все признаки операции, кроме Class.
y содержит ответ: 0 — обычная операция, 1 — мошенничество.

2. Разделение на train/test
Данные разделены на обучающую и тестовую выборки в пропорции 80/20.
Параметр stratify=y сохранит примерно одинаковую долю мошеннических операций в train и test.

Train: {X_train.shape[0]:,} строк.
Test: {X_test.shape[0]:,} строк.
Доля fraud в train: {pct(float(y_train.mean()))}.
Доля fraud в test: {pct(float(y_test.mean()))}.

3. Логарифмическая нормализация Amount
Исходный признак Amount был заменен на Amount_Log = log(1 + Amount). Это уменьшает влияние редких экстремально крупных операций и делает распределение суммы более сглаженным.

4. Масштабирование
Признаки Time и Amount_Log были стандартизированы через StandardScaler. Это особенно важно для Logistic Regression, потому что линейные модели чувствительны к масштабу признаков.
"""

    rows_explanation = "\n".join(["- " + explain_metric_row(row) for _, row in results.iterrows()])
    best_row = results[results["Model"] == best_model_name].iloc[0]
    models_text = f"""ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ МОДЕЛЕЙ

В проекте обучены три модели:
1. Logistic Regression — базовая линейная модель.
2. Decision Tree — интерпретируемая модель на правилах.
3. Random Forest — ансамбль деревьев, обычно более устойчивый, чем одно дерево.

Расшифровка метрик:
Accuracy — общая доля правильных ответов. В антифроде может быть обманчивой из-за дисбаланса.
Precision — сколько операций, отмеченных моделью как мошеннические, действительно мошеннические.
Recall — какую долю реальных мошеннических операций модель смогла найти.
F1-score — баланс между Precision и Recall.
ROC-AUC — общая способность модели различать обычные и мошеннические операции.

Результаты простыми словами:
{rows_explanation}

Лучшая модель по правилу Recall -> F1-score -> ROC-AUC: {best_model_name}.

Почему приоритет отдан Recall:
В антифрод-задаче опаснее пропустить мошенническую операцию, чем дополнительно проверить подозрительную нормальную операцию. Поэтому важнее найти как можно больше мошенничества, но при этом нужно следить за Precision, чтобы не создавать слишком много ложных тревог.

Краткий вывод по лучшей модели:
{best_model_name} обнаружила {int(best_row['TP'])} мошеннических операций из {int(best_row['TP'] + best_row['FN'])} в тестовой выборке. Пропущено {int(best_row['FN'])}. Ложных тревог: {int(best_row['FP'])}.
"""

    features_text = f"""ВАЖНОСТЬ ПРИЗНАКОВ

Random Forest позволяет оценить, какие признаки чаще использовались деревьями для разделения операций.

Топ-10 важных признаков:
{top_features.to_string(index=False)}

Важно: важность признака не означает причинно-следственную связь. Она показывает вклад признака в работу модели на данном датасете.
"""

    final_text = f"""ИТОГОВЫЙ ВЫВОД

В проекте реализован полный учебный pipeline для задачи определения мошеннических банковских операций:
1. загружен датасет Credit Card Fraud Detection;
2. проведен EDA;
3. выполнена очистка данных;
4. выполнена логарифмическая нормализация Amount;
5. построено нормализованное распределение операций по часам;
6. выполнено масштабирование Time и Amount_Log;
6. обучены Logistic Regression, Decision Tree и Random Forest;
7. модели сравнены по Precision, Recall, F1-score и ROC-AUC;
8. сохранены графики, таблицы и предсказания.

Главная особенность данных — сильный дисбаланс классов: мошеннических операций всего {pct(fraud_share)}.
Из-за этого Accuracy не является главной метрикой. Основной акцент нужно делать на Recall, Precision, F1-score и Precision-Recall-кривой.

Для демонстрации в учебном проекте лучшей моделью выбрана: {best_model_name}.
"""

    save_text(paths["texts"] / "01_summary.txt", summary_text)
    save_text(paths["texts"] / "02_eda_interpretation.txt", eda_text)
    save_text(paths["texts"] / "03_preprocessing.txt", preprocessing_text)
    save_text(paths["texts"] / "04_models_interpretation.txt", models_text)
    save_text(paths["texts"] / "05_feature_importance.txt", features_text)
    save_text(paths["texts"] / "06_final_conclusion.txt", final_text)
    save_text(paths["root"] / "human_readable_report.txt", "\n\n".join([summary_text, eda_text, preprocessing_text, models_text, features_text, final_text]))

    summary = {
        "input_file": str(input_file),
        "rows_original": int(original_shape[0]),
        "columns_original": int(original_shape[1]),
        "rows_after_cleaning": int(df.shape[0]),
        "columns_after_cleaning": int(df.shape[1]),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "normal_count": normal_count,
        "fraud_count": fraud_count,
        "fraud_share": fraud_share,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "best_model": best_model_name,
        "output_base_dir": str(output_dir.resolve()),
        "run_dir": str(paths["root"].resolve()),
        "plots_dir": str(paths["plots"].resolve()),
    }
    save_json(paths["root"] / "summary.json", summary)

    print("=== Анализ завершен успешно ===", flush=True)
    print(f"Лучшая модель: {best_model_name}", flush=True)
    print(f"Отчет: {paths['root'] / 'human_readable_report.txt'}", flush=True)
    print(f"Таблица сравнения моделей: {paths['tables'] / 'model_comparison.csv'}", flush=True)
    print(f"Папка текущего запуска: {paths['root']}", flush=True)
    print(f"Графики текущего запуска: {paths['plots']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Учебный антифрод pipeline")
    parser.add_argument("--input", required=True, help="Путь к CSV-файлу creditcard.csv")
    parser.add_argument("--output", default="output", help="Папка для сохранения результатов")
    args = parser.parse_args()

    try:
        run_pipeline(args.input, args.output)
    except Exception as exc:
        print("ОШИБКА ВЫПОЛНЕНИЯ PYTHON-СКРИПТА:", flush=True)
        print(str(exc), flush=True)
        raise
