from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler


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
        "wallet_age",
        "incoming_tx_count",
        "outgoing_tx_count",
        "net_incoming_tx_count",
        "total_gas_paid_eth",
        "avg_gas_paid_per_tx_eth",
        "risky_tx_count",
        "risky_unique_contract_count",
        "risky_first_last_tx_timestamp_diff",
        "risky_sum_outgoing_amount_eth",
        "outgoing_tx_sum_eth",
        "incoming_tx_sum_eth",
        "outgoing_tx_avg_eth",
        "incoming_tx_avg_eth",
        "max_eth_ever",
        "min_eth_ever",
        "total_balance_eth",
        "risk_factor",
        "total_collateral_eth",
        "total_collateral_avg_eth",
        "total_available_borrows_eth",
        "total_available_borrows_avg_eth",
        "avg_weighted_risk_factor",
        "risk_factor_above_threshold_daily_count",
        "avg_risk_factor",
        "max_risk_factor",
        "borrow_amount_sum_eth",
        "borrow_amount_avg_eth",
        "borrow_count",
        "repay_amount_sum_eth",
        "repay_amount_avg_eth",
        "repay_count",
        "borrow_repay_diff_eth",
        "deposit_count",
        "deposit_amount_sum_eth",
        "time_since_first_deposit",
        "withdraw_amount_sum_eth",
        "withdraw_deposit_diff_if_positive_eth",
        "liquidation_count",
        "time_since_last_liquidated",
        "liquidation_amount_sum_eth",
        "market_adx",
        "market_adxr",
        "market_apo",
        "market_aroonosc",
        "market_aroonup",
        "market_atr",
        "market_cci",
        "market_cmo",
        "market_correl",
        "market_dx",
        "market_fastk",
        "market_fastd",
        "market_ht_trendmode",
        "market_linearreg_slope",
        "market_macd_macdext",
        "market_macd_macdfix",
        "market_macd",
        "market_macdsignal_macdext",
        "market_macdsignal_macdfix",
        "market_macdsignal",
        "market_max_drawdown_365d",
        "market_natr",
        "market_plus_di",
        "market_plus_dm",
        "market_ppo",
        "market_rocp",
        "market_rocr",
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
