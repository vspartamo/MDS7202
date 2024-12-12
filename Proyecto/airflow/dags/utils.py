import pandas as pd
from sklearn.pipeline import Pipeline
from scipy.stats import ks_2samp, mannwhitneyu
from scipy.stats import cramervonmises_2samp
import pickle
from typing import Callable, List, Union
import mlflow
import optuna
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler


# FunctionTransformer para dropear columnas
def drop_columns(X, features_to_drop):
    return X.drop(columns=features_to_drop, errors="ignore")


# FunctionTransformer para verificar y ordenar columnas
def verify_and_order_columns(X, features_order):
    missing_columns = [col for col in features_order if col not in X.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    return X[features_order]


def create_preprocessor(numeric_features, scaler, use_pca, pca_components):
    """Crea el preprocesador basado en las características numéricas,
    incluyendo los FunctionTransformers para dropear y ordenar columnas."""

    FEATURES_TO_DROP = [
        "borrow_block_number",
        "wallet_address",
        "borrow_timestamp",
        "first_tx_timestamp",
        "last_tx_timestamp",
        "risky_first_tx_timestamp",
        "risky_last_tx_timestamp",
        "unique_borrow_protocol_count",
        "unique_lending_protocol_count",
    ]

    FEATURES_ORDER = [
        "wallet_age"
        "incoming_tx_count"
        "outgoing_tx_count"
        "net_incoming_tx_count"
        "total_gas_paid_eth"
        "avg_gas_paid_per_tx_eth"
        "risky_tx_count"
        "risky_unique_contract_count"
        "risky_first_last_tx_timestamp_diff"
        "risky_sum_outgoing_amount_eth"
        "outgoing_tx_sum_eth"
        "incoming_tx_sum_eth"
        "outgoing_tx_avg_eth"
        "incoming_tx_avg_eth"
        "max_eth_ever"
        "min_eth_ever"
        "total_balance_eth"
        "risk_factor"
        "total_collateral_eth"
        "total_collateral_avg_eth"
        "total_available_borrows_eth"
        "total_available_borrows_avg_eth"
        "avg_weighted_risk_factor"
        "risk_factor_above_threshold_daily_count"
        "avg_risk_factor"
        "max_risk_factor"
        "borrow_amount_sum_eth"
        "borrow_amount_avg_eth"
        "borrow_count"
        "repay_amount_sum_eth"
        "repay_amount_avg_eth"
        "repay_count"
        "borrow_repay_diff_eth"
        "deposit_count"
        "deposit_amount_sum_eth"
        "time_since_first_deposit"
        "withdraw_amount_sum_eth"
        "withdraw_deposit_diff_if_positive_eth"
        "liquidation_count"
        "time_since_last_liquidated"
        "liquidation_amount_sum_eth"
        "market_adx"
        "market_adxr"
        "market_apo"
        "market_aroonosc"
        "market_aroonup"
        "market_atr"
        "market_cci"
        "market_cmo"
        "market_correl"
        "market_dx"
        "market_fastk"
        "market_fastd"
        "market_ht_trendmode"
        "market_linearreg_slope"
        "market_macd_macdext"
        "market_macd_macdfix"
        "market_macd"
        "market_macdsignal_macdext"
        "market_macdsignal_macdfix"
        "market_macdsignal"
        "market_max_drawdown_365d"
        "market_natr"
        "market_plus_di"
        "market_plus_dm"
        "market_ppo"
        "market_rocp"
        "market_rocr"
    ]

    if scaler is None:
        scaler = MinMaxScaler()

    # Dropear columnas
    drop_transformer = FunctionTransformer(
        drop_columns, kw_args={"features_to_drop": FEATURES_TO_DROP}
    )

    # Verificar y ordenar columnas
    order_transformer = FunctionTransformer(
        verify_and_order_columns, kw_args={"features_order": FEATURES_ORDER}
    )

    # Crear pipeline para las características numéricas
    numeric_transformer_steps = [("scaler", scaler)]
    if use_pca:
        numeric_transformer_steps.append(("pca", PCA(n_components=pca_components)))
    numeric_transformer = Pipeline(steps=numeric_transformer_steps)

    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("dropper", drop_transformer, numeric_features),
            ("num", numeric_transformer, numeric_features),
            ("orderer", order_transformer, numeric_features),
        ]
    )
    return preprocessor


def create_pipeline(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
):
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


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

    evaluate_model(y_test, y_pred, y_pred_proba)

    return pipeline


def preprocess_data(df: pd.DataFrame, preprocessor: Pipeline) -> pd.DataFrame:
    """
    Prepares the data to be used in the model.

    Args
    -----
    data: dict
        The data to prepare.

    Returns
    -----
    pd.DataFrame
        The data prepared.
    """
    df = preprocessor.transform(df)
    return df


def detect_drift(
    train_data, production_data, features, target_column, method="ks", alpha=0.05
):
    for feature in features:
        train_feature = train_data[feature]
        prod_feature = production_data[feature]

        if method == "ks":
            # Prueba Kolmogorov-Smirnov
            stat, p_value = ks_2samp(train_feature, prod_feature)
        elif method == "mw":
            # Prueba Mann-Whitney U
            stat, p_value = mannwhitneyu(
                train_feature, prod_feature, alternative="two-sided"
            )
        elif method == "cv":
            # Prueba Cramér-von Mises
            stat, p_value = cramervonmises_2samp(train_feature, prod_feature)

        drift_detected = p_value < alpha

    # Detección de target drift
    target_train = train_data[target_column]
    target_prod = production_data[target_column]

    if method == "ks":
        target_stat, target_p_value = ks_2samp(target_train, target_prod)
    elif method == "mw":
        target_stat, target_p_value = mannwhitneyu(
            target_train, target_prod, alternative="two-sided"
        )
    elif method == "cv":
        target_stat, target_p_value = cramervonmises_2samp(target_train, target_prod)

    target_drift_detected = target_p_value < alpha

    return drift_detected, target_drift_detected
