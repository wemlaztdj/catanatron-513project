# Variable-road experiment — fresh 5,000-game test

This branch is based directly on main. The original tracked 100-game pilot data and processed files are untouched. Recovered experiment code is included, but notebook outputs are cleared. All new experiment notebooks use data_5000/, processed_data_5000/, and processed_data_variable_5000/. Generated datasets and charts are ignored by Git.

## Run locally

From the repository root, activate your existing Python environment.

    mkdir -p data_5000
    catanatron-play --num 5000 --players R,R,R,R --output data_5000/ --output-format json

Use an empty data_5000 directory when generating the dataset. If you already generated 5,000 games, put those files there instead of generating duplicates. Do not mix the tracked pilot data with the new data.

    python -m pip install pandas numpy scipy matplotlib seaborn scikit-learn notebook
    python -m unittest test_variable_road_model.py

Run notebooks 01, 02, 03, 04 (optional), 05_starting_advantage, 07_initial_vs_outcome, 08_variable_road_advantage. Notebook 05_classification and 06_model_comparison are optional, and should be run in that order after 05_starting_advantage.

For multiprocessing after 05_starting_advantage:

    python variable_road_model.py --data-dir processed_data_5000 --output-dir processed_data_variable_5000 --workers 4

Then replace the d=run(DATA,OUT,workers=1) line in notebook 08 with d=pd.read_csv(OUT/"starting_advantage.csv") to avoid recalculation. Notebook 08 can otherwise run the single-worker calculation directly.

## Method and limits

BFS searches from both initial road networks, respecting occupied roads, opponent buildings, distance rule and remaining road supply. For k additional roads, cost is (k+1,k+1,1,1,0). The exact fixed-production model includes 4:1 bank trading and a 40-personal-turn horizon. One-road results are reused; extra-road positions are recalculated. A genuine no-path case remains a separate technical status. Fixed class boundaries are 7 and 14 turns. Winner prediction compares both models on identical game-grouped folds.

Historical pilot: 376 of 400 players required one road, 24 required two, none had no path. All 24 newly eligible positions reached the 50% threshold in 40 turns in an independent calculation. These are not results for the fresh 5,000 games.

No new simulation, Python unit test or notebook execution was performed during branch recreation. Run locally for the new results.

## Notebook 05/06 troubleshooting
Select your installed Python environment (Python 3 kernel) and run from the repository root. 05a validates the 5,000-game CSVs before computing and prints progress every 250 players. Set REUSE_EXISTING=True only to reuse scores for the same game IDs. 05b validates the dataset and prints progress for each algorithm. 06 can generate five descriptive charts without 05b; classifier charts appear only when current classification outputs are present. The CLI defaults for advantage_model.py, advantage_analysis.py and advantage_charts.py now use the 5,000-game paths. No model assumptions or classification hyperparameters changed (parallel workers and exact-result caching only).
