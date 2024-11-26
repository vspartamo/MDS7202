import optuna
import mlflow
import mlflow.sklearn
import pickle
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Definir directorio base
base_dir = "Lab_12"
os.makedirs(base_dir, exist_ok=True)

data_path = os.path.join(base_dir, "water_potability.csv")
df = pd.read_csv(data_path)

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y, random_state=3)

def objective(trial):
    """Función objetivo para Optuna"""
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_uniform("gamma", 0, 5),
    }

    model = XGBClassifier(**params, random_state=3, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred)

    with mlflow.start_run(run_name=f"XGBoost_lr_{params['learning_rate']:.2f}"):
        mlflow.log_params(params)
        mlflow.log_metric("valid_f1", f1)

    return f1

def optimize_model():
    experiment_name = "Water_Potability_Optimization"
    mlflow.set_experiment(experiment_name)

    mlflow_callback = MLflowCallback(metric_name="valid_f1", tracking_uri=mlflow.get_tracking_uri())

    # Crear carpetas para guardar artefactos
    plots_dir = os.path.join(base_dir, "plots")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize", study_name="XGBoost Optimization")
    study.optimize(objective, n_trials=50, callbacks=[mlflow_callback])
    optuna.visualization.plot_optimization_history(study).write_image(os.path.join(plots_dir, "optimization_history.png"))
    optuna.visualization.plot_param_importances(study).write_image(os.path.join(plots_dir, "param_importances.png"))

    # Obtener el mejor modelo
    best_model = XGBClassifier(**study.best_params, random_state=3, use_label_encoder=False, eval_metric="logloss")
    best_model.fit(X_train, y_train)

    with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    with mlflow.start_run(run_name="Best_Model"):
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        mlflow.log_params(study.best_params)
        mlflow.log_artifact(os.path.join(plots_dir, "optimization_history.png"))
        mlflow.log_artifact(os.path.join(plots_dir, "param_importances.png"))

if __name__ == "__main__":
    optimize_model()
