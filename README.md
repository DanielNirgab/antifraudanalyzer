# Учебный проект: определение мошеннических банковских операций

Проект состоит из двух частей:

- `python/antifraud_pipeline.py` — Python-скрипт анализа датасета Credit Card Fraud Detection;
- `java-ui/src/DataAnalyzerApp.java` — простой Java Swing-интерфейс для запуска анализа и просмотра результатов.

## Что делает Python-скрипт

Скрипт выполняет полный учебный pipeline:

1. загружает выбранный CSV-файл;
2. проверяет структуру датасета;
3. выполняет EDA;
4. строит гистограммы всех числовых признаков;
5. строит матрицу корреляций;
6. выполняет логарифмическую нормализацию `Amount`: `Amount_Log = log(1 + Amount)`;
7. строит абсолютное и нормализованное распределение операций по часам;
8. обучает модели:
   - Logistic Regression;
   - Decision Tree;
   - Random Forest;
9. сохраняет метрики, предсказания, графики и текстовые интерпретации.

## Важное изменение: отдельная папка для каждого запуска

Теперь результаты каждого запуска не перезаписываются, а сохраняются в отдельную папку:

```text
output/
├── latest_run.txt
└── runs/
    └── run_YYYYMMDD_HHMMSS/
        ├── plots/
        ├── tables/
        ├── texts/
        ├── summary.json
        └── human_readable_report.txt
```

Файл `output/latest_run.txt` содержит путь к последнему запуску. Java-интерфейс читает этот файл и показывает результаты именно текущего запуска.

## Вкладка «Графики»

Во вкладке «Графики» Java-интерфейс автоматически сканирует папку:

```text
output/runs/run_YYYYMMDD_HHMMSS/plots/
```

и добавляет в выпадающий список все найденные изображения `.png`, `.jpg`, `.jpeg`.

Это значит, что если Python-скрипт создаст новый график и сохранит его в `plots`, он автоматически появится во вкладке «Графики».

## Установка зависимостей Python

```bash
pip install -r requirements.txt
```

## Ручной запуск Python

```bash
python python/antifraud_pipeline.py --input путь_к_creditcard.csv --output output
```

## Запуск Java-интерфейса

Из корня проекта:

```bash
javac java-ui/src/DataAnalyzerApp.java
java -cp java-ui/src DataAnalyzerApp
```

Далее:

1. нажмите «Проверить Python»;
2. выберите CSV-файл датасета;
3. нажмите «Запустить полный анализ»;
4. смотрите результаты во вкладках.

## Основные выходные файлы текущего запуска

```text
plots/01_class_distribution.png
plots/02_all_numeric_histograms.png
plots/03_amount_distribution_raw.png
plots/04_amount_log_normalization.png
plots/05_amount_boxplot_by_class.png
plots/06_time_distribution_by_hour.png
plots/06b_time_distribution_normalized.png
plots/07_full_correlation_heatmap.png
plots/08_correlation_with_class.png
plots/09_model_comparison.png
plots/10_roc_curves.png
plots/11_precision_recall_curves.png
plots/12_feature_importance.png
plots/13_decision_tree.png
plots/confusion_matrix_logistic_regression.png
plots/confusion_matrix_decision_tree.png
plots/confusion_matrix_random_forest.png
```
