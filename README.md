# Учебный интерфейс антифрод-анализа

Проект состоит из двух частей:

- `python/antifraud_pipeline.py` — Python-скрипт анализа датасета Credit Card Fraud Detection;
- `java-ui/src/DataAnalyzerApp.java` — простой Java Swing-интерфейс с вкладками для просмотра результатов.

## Установка Python-зависимостей

```bash
pip install -r requirements.txt
```

## Запуск Java-интерфейса

Из корня проекта:

```bash
javac java-ui/src/DataAnalyzerApp.java
java -cp java-ui/src DataAnalyzerApp
```

Далее:

1. нажмите «Проверить Python»;
2. нажмите «Выбрать CSV датасет» и выберите `creditcard.csv`;
3. нажмите «Запустить полный анализ»;
4. смотрите результаты во вкладках.

## Что создает Python-скрипт

В папке `output` появятся:

- `texts/*.txt` — человекочитаемые объяснения результатов;
- `tables/*.csv` — таблицы распределения классов, сравнения моделей, предсказания;
- `plots/*.png` — графики EDA, матрицы ошибок, ROC/PR-кривые, важность признаков;
- `human_readable_report.txt` — общий текстовый отчет.

## Отдельный запуск Python

```bash
python python/antifraud_pipeline.py --input path/to/creditcard.csv --output output
```
