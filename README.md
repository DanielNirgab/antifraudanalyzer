# Учебный проект: определение мошеннических банковских операций

Проект состоит из двух частей:

1. `python/antifraud_pipeline.py` — Python-скрипт анализа данных, обучения моделей и сохранения результатов.
2. `java-ui/src/DataAnalyzerApp.java` — простой Java Swing-интерфейс для выбора CSV-файла, запуска Python-скрипта и просмотра результатов по вкладкам.

## Что делает Python-скрипт

- загружает выбранный CSV-файл Credit Card Fraud Detection;
- проверяет наличие обязательных столбцов `Time`, `Amount`, `Class`;
- выполняет EDA;
- строит гистограммы всех числовых признаков;
- строит матрицу корреляций;
- выполняет логарифмическую нормализацию `Amount` через `Amount_Log = log(1 + Amount)`;
- обучает модели Logistic Regression, Decision Tree и Random Forest;
- сравнивает модели по Accuracy, Precision, Recall, F1-score и ROC-AUC;
- сохраняет таблицы, графики, предсказания и человекочитаемые пояснения.

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск Python отдельно

```bash
python python/antifraud_pipeline.py --input path/to/creditcard.csv --output output
```

## Запуск Java-интерфейса

```bash
javac java-ui/src/DataAnalyzerApp.java
java -cp java-ui/src DataAnalyzerApp
```

В интерфейсе:

1. проверьте Python;
2. выберите CSV-файл датасета;
3. нажмите «Запустить полный анализ»;
4. просмотрите результаты во вкладках.

## Основные результаты

После запуска результаты сохраняются в папку `output`:

- `output/texts/` — текстовые пояснения для интерфейса;
- `output/tables/` — CSV-таблицы;
- `output/plots/` — графики;
- `output/human_readable_report.txt` — общий текстовый отчет.
