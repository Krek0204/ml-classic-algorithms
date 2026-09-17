# ml-classic-algorithms

Практика реализации классических алгоритмов машинного обучения «с нуля» — без готовых моделей из библиотек, с опорой на математику и NumPy.

## Цель

Репозиторий для тренировки: разобрать, как устроены базовые ML-алгоритмы, и реализовать их самостоятельно. Каждый алгоритм — отдельный модуль с понятной структурой, краткими пояснениями и примерами на простых датасетах.

Это учебный проект, а не production-библиотека.

## Стек

| Компонент | Назначение |
|-----------|------------|
| **Python 3.10+** | Основной язык |
| **NumPy** | Линейная алгебра, векторизация |
| **Matplotlib** | Визуализация результатов и метрик |
| **scikit-learn** | Датасеты и сравнение с эталонной реализацией (не для обучения моделей) |
| **pytest** | Юнит-тесты |

## Алгоритмы

Список в порядке планируемой реализации. Первые пять — наиболее популярные классические методы; далее — расширение покрытия.

### Supervised learning

| # | Алгоритм | Задача | Статус |
|---|----------|--------|--------|
| 1 | Linear Regression | Регрессия | ⏳ planned |
| 2 | Logistic Regression | Классификация | ⏳ planned |
| 3 | Decision Tree | Классификация / регрессия | ⏳ planned |
| 4 | k-Nearest Neighbors (k-NN) | Классификация / регрессия | ⏳ planned |
| 5 | Support Vector Machine (SVM) | Классификация | ⏳ planned |
| 6 | Naive Bayes | Классификация | ⏳ planned |
| 7 | Random Forest | Классификация / регрессия | ⏳ planned |
| 8 | Gradient Boosting | Классификация / регрессия | ⏳ planned |

### Unsupervised learning

| # | Алгоритм | Задача | Статус |
|---|----------|--------|--------|
| 9 | K-Means | Кластеризация | ⏳ planned |
| 10 | PCA | Снижение размерности | ⏳ planned |

## Структура проекта

```text
ml-classic-algorithms/
├── algorithms/          # Реализации алгоритмов
├── examples/            # Ноутбуки / скрипты с демо
├── tests/               # Тесты
├── requirements.txt
└── README.md
```

Структура появится по мере добавления первых реализаций.

## Принципы реализации

- Алгоритмы пишутся вручную (NumPy), без `sklearn.fit` для обучения.
- scikit-learn допустим для загрузки данных и sanity-check метрик.
- Код читаемый: понятные имена, короткие docstring’и, без лишней абстракции.
- Для каждого алгоритма — тест и простой пример запуска.

## Быстрый старт

```bash
git clone https://github.com/<username>/ml-classic-algorithms.git
cd ml-classic-algorithms
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

После появления кода:

```bash
pytest
python examples/linear_regression_demo.py
```

## Roadmap

1. Базовая инфраструктура: `requirements.txt`, каркас пакетов, общий интерфейс `fit` / `predict`.
2. Первая пятёрка алгоритмов с тестами и примерами.
3. Метрики качества (MSE, accuracy, precision/recall, silhouette и т.д.).
4. Расширение списка (Random Forest, Gradient Boosting, K-Means, PCA).
5. Сравнительные бенчмарки «своя реализация vs sklearn».

## Лицензия

Apache License 2.0 — см. [LICENSE](LICENSE).
