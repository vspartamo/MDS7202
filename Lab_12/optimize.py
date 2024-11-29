import optuna
import mlflow
import mlflow.sklearn
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import json
from xgboost import XGBClassifier
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)

base_dir = "docs"

df = pd.read_csv("water_potability.csv")
df.dropna(inplace=True)

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=3
)


def objective(trial: optuna.Trial):
    """Función objetivo para Optuna"""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }

    model = XGBClassifier(**params, random_state=3, eval_metric="logloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred)

    exp_id = mlflow.create_experiment(f"Experiment with lr={params['learning_rate']}")
    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"XGBoost_lr_{params['learning_rate']}"
    ):
        mlflow.log_params(params)
        mlflow.log_metric("valid_f1", f1)
    mlflow.end_run()
    return f1


def optimize_model():
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize", study_name="XGBoost Optimization")
    study.optimize(objective, n_trials=50)
    print(f"Best trial: {study.best_trial.value}")

    best_model_experiment_id = mlflow.create_experiment("Best_Model")
    with mlflow.start_run(
        experiment_id=best_model_experiment_id, run_name="Best_Model"
    ):

        opti_ax = plot_optimization_history(study)
        mlflow.log_figure(
            figure=opti_ax.figure, artifact_file="optimization_history.png"
        )
        opti_ax.figure.savefig(os.path.join(plots_dir, "optimization_history.png"))

        param_importances_ax = plot_param_importances(study)
        mlflow.log_figure(
            figure=param_importances_ax.figure, artifact_file="param_importance.png"
        )
        param_importances_ax.figure.savefig(
            os.path.join(plots_dir, "param_importance.png")
        )

        best_model = XGBClassifier(
            **study.best_params, random_state=3, eval_metric="logloss"
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_valid)
        f1 = f1_score(y_valid, y_pred)

        mlflow.log_metric("valid_f1", f1)
        mlflow.log_params(study.best_params)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)

        with open(os.path.join(base_dir, "library_versions.json"), "w") as f:
            json.dump({"optuna": optuna.__version__, "mlflow": mlflow.__version__}, f)

        mlflow.log_params(study.best_params)
