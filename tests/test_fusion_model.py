import pandas as pd
import joblib


test_df = pd.read_csv("logs/fusion_training_data.csv")
feature_columns = [col for col in test_df.columns if col.startswith("pred_")]

X_test = test_df[feature_columns]
y_test = test_df["true_age"]

tree_model = joblib.load("checkpoints/fusion_model_tree.joblib")
forest_model = joblib.load("checkpoints/fusion_model_forest.joblib")
bayes_model = joblib.load("checkpoints/fusion_model_bayesian.joblib")

# just the fusion results this time, not all 9 individual model columns again
results = pd.DataFrame()
results["image_id"] = test_df["image_id"]
results["true_age"] = y_test

results["fusion_tree"] = tree_model.predict(X_test)
results["fusion_forest"] = forest_model.predict(X_test)
results["fusion_bayesian"] = bayes_model.predict(X_test)

results["error_tree"] = (results["fusion_tree"] - y_test).abs()
results["error_forest"] = (results["fusion_forest"] - y_test).abs()
results["error_bayesian"] = (results["fusion_bayesian"] - y_test).abs()

results.to_csv("logs/fusion_test_results.csv", index=False)

print(f"logged {len(results)} images to logs/fusion_test_results.csv\n")
print(f"decision tree:   MAE = {results['error_tree'].mean():.4f}")
print(f"random forest:   MAE = {results['error_forest'].mean():.4f}")
print(f"bayesian ridge:  MAE = {results['error_bayesian'].mean():.4f}")