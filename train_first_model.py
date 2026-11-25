
import pandas as pd
from catanatron.models.enums import ActionType
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

# 1. Load the dataset
df = pd.read_csv("data/main.csv.gz")

# 2. Select game-state features only
feature_cols = [c for c in df.columns if c.startswith("F_")]
X = df[feature_cols].fillna(0)

# 3. The target is the action category
y = df["ACTION_TYPE"].astype(int)

# Convert action IDs to readable names
action_types = list(ActionType)
action_names = y.map(lambda i: action_types[i].name)

print("Dataset shape:", X.shape)
print("\nAction type distribution:")
print(action_names.value_counts())

# 4. Preliminary train/test split
# This is only a pipeline test, NOT a final evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Compare two classifiers
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
}

for name, model in models.items():
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:",
          round(accuracy_score(y_test, predictions), 4))
    print("Balanced accuracy:",
          round(balanced_accuracy_score(y_test, predictions), 4))
    print("Macro F1:",
          round(f1_score(
              y_test, predictions, average="macro",
              zero_division=0
          ), 4))

    print(classification_report(
        y_test, predictions, zero_division=0
    ))
