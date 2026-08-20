# Прогнозиране на качеството на вино чрез методи за класификация

Курсов проект по извличане на знания от данни. По физикохимични измервания на португалско *Vinho Verde* се предсказва качественият клас на виното с класификационни модели.

```
физикохимични измервания
          ↓
   предварителна обработка
          ↓
   train / test split
          ↓
   избор на признаци
          ↓
   Decision Tree · k-NN · Random Forest
          ↓
      предсказване
          ↓
     оценка (accuracy, precision, recall, F1, confusion matrix)
          ↓
     сравнение на моделите
```

Данни: UCI [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) — Cortez et al. (2009). 1599 червени и 4898 бели проби, 11 химични признака + тип на виното, оценка от експерти 0–10.

## Как да се пусне

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/wine_quality.py
jupyter lab notebooks/wine_quality_classification.ipynb
```

`src/wine_quality.py` е възпроизводимият пайплайн (seed 42). Резултатите отиват в `results/`, графиките — в `figures/`. Бележникът повтаря същите стъпки с обяснения за защита/преглед.

## Целева променлива

Оригиналните оценки 0–10 са силно небалансирани (почти всички вина са 5 или 6). За класификация се ползват три класа:

| Клас   | Оценка | Брой |
| ------ | ------ | ---: |
| ниско  | ≤ 4    |  246 |
| средно | 5–6    | 4974 |
| високо | ≥ 7    | 1277 |

## Резултати (тестова извадка, 1300 проби)

| Модел                 | Accuracy | Precision | Recall | F1-macro | CV F1-macro |
| --------------------- | -------: | --------: | -----: | -------: | ----------: |
| Baseline (majority)   |    0.765 |     0.255 |  0.333 |    0.289 |           — |
| Decision Tree         |    0.795 |     0.593 |  0.572 |    0.579 |       0.547 |
| k-NN                  |  **0.816** |   **0.686** |  0.593 |  **0.618** |       0.585 |
| Random Forest         |    0.783 |     0.597 |  **0.650** |    0.616 |     **0.618** |

Precision / recall са macro-осреднени. Оптимизационната мярка при `GridSearchCV` е F1-macro, защото majority baseline вече има 76.5% accuracy.

- k-NN (`k=3`, Евклид, `weights=distance`) има най-висока тестова точност.
- Random Forest има най-добър CV резултат и най-добър recall за редките класове.
- Decision Tree с неограничена дълбочина преобучава (train F1 = 1.0).
- Най-информативният признак е **алкохолът**, следван от свободния серен диоксид и летливата киселинност.
- Намаляването до 8 признака чрез ANOVA `SelectKBest` почти не влошава Random Forest (0.616 → 0.611).

## Какво покрива спрямо курса

supervised learning · classification · train/test · cross-validation · feature selection · hyperparameter selection · decision trees · k-NN · Random Forest · accuracy / precision / recall / F1 · confusion matrices · comparison of algorithms

## Цитиране

P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis. Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4):547–553, 2009.
