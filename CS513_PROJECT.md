# CS 513 sample project — Catan initial placement analysis

## Scope and research question
Analyze the first two placement rounds (two settlements and two roads per player), before the first dice roll. What distinct profiles and theoretical resource advantages exist? No subsequent action, actual dice outcome, third settlement or winner enters the analysis.

## Data and reproducibility
100 four-player RandomPlayer games, 400 player positions, 1,900 land tiles. Raw JSON in `data/`; processed CSVs in `processed_data/`. Desert resource/number nulls are valid. Run from the repository root using Python with pandas, numpy, matplotlib, seaborn, scikit-learn and Jupyter. Existing Catanatron source and historical action-prediction scripts are separate from this workflow.

1. `01_data_extraction.ipynb` — extract the initial 16 actions and map.
2. `02_data_cleaning.ipynb` — validate every JSON against extracted data and board topology.
3. `03_feature_engineering.ipynb` — rebuild the existing 22-column feature dataset using `03_feature_engineering.py`.
4. `04_exploratory_data_analysis.ipynb` — statistics and plots.
5. `05_classification.ipynb` — clustering, analytical label, supervised comparison, group CV.
6. `06_model_comparison.ipynb` — summarize charts and results.

## Sample results
Computed on the supplied 400-row CSV dataset with scikit-learn 1.8.0, seed 42. The newly added notebooks are supplied as runnable source (their cells have not been executed on GitHub). Results may differ slightly across package versions.

Mean initial production pips: 12.91. Mean production by placement order: 13.07, 13.07, 12.95, 12.56. These are descriptive, not causal.

Clustering: maximum tested silhouette 0.1351 (K-Means, k=5); best Ward 0.0999 (k=4). Weak separation, so do not claim five distinct natural strategies.

Supervised proxy: probability of accumulating the cost of one road and one settlement (2 wood, 2 brick, 1 sheep, 1 wheat) within 20 hypothetical independent rolls, with no spending, trading, robber movement or bank shortages. Exact dynamic programming, not simulation. A probability >=0.10 is class 1. The horizon and threshold were chosen exploratorily after an 8-roll, 25% pilot had no positives. This is not an actual outcome or independent ground truth. 40 positives, 360 negatives; 357 zero probabilities. Mean probability 4.31%, maximum 75.85%. Grouped test set: 80 players (11 positive, 69 negative).

| Model | Test accuracy | Test balanced accuracy | Test F1 | Test ROC-AUC | 5-fold group CV mean F1 |
|---|---:|---:|---:|---:|---:|
| Majority baseline | .8625 | .5000 | .0000 | .5000 | .0000 |
| KNN | .8750 | .5455 | .1667 | .8821 | .4975 |
| Gaussian NB | .9125 | .7200 | .5882 | .8867 | .5684 |
| CART | .8750 | .5455 | .1667 | .6858 | .3058 |
| Random Forest | .9000 | .6364 | .4286 | 1.0000 | .8208 |
| Logistic regression | .8625 | .5382 | .1538 | .8353 | .3269 |

The holdout is small; Naive Bayes has the best held-out F1, while Random Forest has the best group-CV mean F1. No external validation or tuned-model performance is claimed. Exact probability calculation is preferable when complete board information is available. No model demonstrates winning advantage. Port types are excluded from the core model; add them in future work.

## References
- Catanatron: https://github.com/bcollazo/catanatron
- Bryan Collazo, “5 Ways NOT to Build a Catan AI”: https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
