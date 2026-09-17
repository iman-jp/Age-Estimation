import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor


train_df = pd.read_csv("logs/fusion_training_data_from_val.csv")
test_df = pd.read_csv("logs/fusion_training_data.csv")

print(f"Fusion train set (from val): {len(train_df)} rows")
print(f"Fusion test set (from test): {len(test_df)} rows")

feature_columns = [col for col in train_df.columns if col.startswith("pred_")]

X_train = train_df[feature_columns]
y_train = train_df["true_age"]
X_test = test_df[feature_columns]
y_test = test_df["true_age"]

base_mae = mean_absolute_error(y_test, X_test["pred_base"])
print(f"Base model alone: {base_mae:.4f}\n")

# --- Decision Tree ---
tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)
tree_mae = mean_absolute_error(y_test, tree_model.predict(X_test))
print(f"Decision Tree fusion MAE: {tree_mae:.4f}")
joblib.dump(tree_model, "checkpoints/fusion_model_tree.joblib")

for name, imp in sorted(zip(feature_columns, tree_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")

# --- Bayesian Ridge ---
bayes_model = BayesianRidge()
bayes_model.fit(X_train, y_train)
bayes_mae = mean_absolute_error(y_test, bayes_model.predict(X_test))
print(f"\nBayesian Ridge fusion MAE: {bayes_mae:.4f}")
joblib.dump(bayes_model, "checkpoints/fusion_model_bayesian.joblib")

for name, coef in sorted(zip(feature_columns, bayes_model.coef_), key=lambda x: -abs(x[1])):
    print(f"  {name}: {coef:.3f}")

# --- Random Forest ---
forest_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
forest_model.fit(X_train, y_train)
forest_mae = mean_absolute_error(y_test, forest_model.predict(X_test))
print(f"\nRandom Forest fusion MAE: {forest_mae:.4f}")
joblib.dump(forest_model, "checkpoints/fusion_model_forest.joblib")

for name, imp in sorted(zip(feature_columns, forest_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")