import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import optuna
import mlflow
import pickle
from typing import Union, List, Callable


def optimize_hyperparameters(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    get_optuna_params: Callable,
    experiment_name: str,
    n_trials: int = 50,
    direction: str = "maximize",
):
    def objective(trial: optuna.Trial):
        run_name = f"trial_{trial.number}_optimization"
        with mlflow.start_run(run_name=run_name, nested=True):
            optuna_params = get_optuna_params(trial)
            model.set_params(**optuna_params)
            mlflow.log_params(optuna_params)
            
            model.fit(X_train, y_train)

            # Optimizamos sobre el AUC ROC
            y_proba_pred = model.predict_proba(X_train)[:, 1]
            roc_auc = roc_auc_score(y_train, y_proba_pred)
            mlflow.log_metric("roc_auc", roc_auc)
            
            return roc_auc

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="optuna_study"):
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        mlflow.end_run()

    # Log best parameters in a separate run
    with mlflow.start_run(run_name="best_params", nested=False):
        mlflow.log_params(study.best_params)

    return study.best_params


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
        "roc_auc": (
            roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
        ),
    }
    print(f"Métricas del modelo:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"Classification report:\n{classification_report(y_true, y_pred)}")
    return metrics


def log_metrics(metrics: dict, prefix: str):
    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()})


def save_model(pipeline: Pipeline, save_model_path: str):
    with open(save_model_path, "wb") as f:
        pickle.dump(pipeline, f)


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray],
    numeric_features: Union[pd.Index, List[str]],
    categorical_features: Union[pd.Index, List[str]],
    experiment_name: str,
    save_model_path: str = None,
    scaler=None,
    use_pca=False,
    pca_components=50,
    optimize: bool = False,
    get_optuna_params: Callable = None,
    n_trials: int = 50,
):
    mlflow.set_experiment(experiment_name)

    # Create preprocessor
    preprocessor = create_preprocessor(
        numeric_features, categorical_features, scaler, use_pca, pca_components
    )

    # Optimize hyperparameters if required
    if optimize and get_optuna_params:
        best_params = optimize_hyperparameters(
            model, X_train, y_train, get_optuna_params, experiment_name, n_trials
        )
        mlflow.end_run()
        model.set_params(**best_params)

    # Create pipeline
    pipeline = create_pipeline(model, preprocessor)

    # Train the pipeline
    with mlflow.start_run(run_name="best_params_training"):
        pipeline.fit(X_train, y_train)

        # Log metrics
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)

        if hasattr(model, "predict_proba"):
            y_train_proba_pred = pipeline.predict_proba(X_train)[:, 1]
            y_val_proba_pred = pipeline.predict_proba(X_val)[:, 1]
        else:
            y_train_proba_pred = None
            y_val_proba_pred = None

        print("Evaluación del modelo en el conjunto de entrenamiento:")
        train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba_pred)
        print("Evaluación del modelo en el conjunto de validación:")
        val_metrics = evaluate_model(y_val, y_val_pred, y_val_proba_pred)

        log_metrics(train_metrics, "train")
        log_metrics(val_metrics, "val")

        if save_model_path:
            # Save model and preprocessor
            save_model(pipeline, save_model_path)

    return pipeline


def retrain_model(
    pipeline_or_path: Union[str, Pipeline],
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    fun_to_update_model: Callable,
    save_model_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    optimize: bool = False,
    get_optuna_params: Callable = None,
    n_trials: int = 50,
):
    if isinstance(pipeline_or_path, str):
        with open(pipeline_or_path, "rb") as f:
            pipeline = pickle.load(f)
    else:
        pipeline = pipeline_or_path

    preprocessor = pipeline.named_steps["preprocessor"]
    model: BaseEstimator = pipeline.named_steps["classifier"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_transformed = preprocessor.transform(X_train)

    if optimize and get_optuna_params:
        best_params = optimize_hyperparameters(
            model,
            X_train_transformed,
            y_train,
            get_optuna_params,
            "retraining_optimization",
            n_trials,
        )
        model.set_params(**best_params)

    model = fun_to_update_model(model, X_train_transformed, y_train)

    pipeline = create_pipeline(model, preprocessor)
    save_model(pipeline, save_model_path)

    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = (
        model.predict_proba(X_test_transformed)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    print("Evaluación en el conjunto de prueba:")
    evaluate_model(y_test, y_pred, y_pred_proba)

    return pipeline