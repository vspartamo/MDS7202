from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.pipeline import Pipeline
import uvicorn
import os
import pandas as pd
from sklearn.compose import ColumnTransformer


COLUMNS_TO_DROP = [
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

model_path = "models/best_model.pkl"
app = FastAPI()

print("Directorio de trabajo actual:", os.getcwd())
print("Ruta absoluta:", os.path.abspath("."))


def prepare_data(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
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
    # 1. sacar columnas
    df = df.drop(COLUMNS_TO_DROP, axis=1)
    # 2. Pasar por preprocesador del pipeline
    df = preprocessor.transform(df)
    return df


class ClientInfo(BaseModel):
    wallet_age: float
    incoming_tx_count: int
    outgoing_tx_count: int
    net_incoming_tx_count: int
    total_gas_paid_eth: float
    avg_gas_paid_per_tx_eth: float
    risky_tx_count: int
    risky_unique_contract_count: int
    risky_first_last_tx_timestamp_diff: int
    risky_sum_outgoing_amount_eth: float
    outgoing_tx_sum_eth: float
    incoming_tx_sum_eth: float
    outgoing_tx_avg_eth: float
    incoming_tx_avg_eth: float
    max_eth_ever: float
    min_eth_ever: float
    total_balance_eth: float
    risk_factor: float
    total_collateral_eth: float
    total_collateral_avg_eth: float
    total_available_borrows_eth: float
    total_available_borrows_avg_eth: float
    avg_weighted_risk_factor: float
    risk_factor_above_threshold_daily_count: float
    avg_risk_factor: float
    max_risk_factor: float
    borrow_amount_sum_eth: float
    borrow_amount_avg_eth: float
    borrow_count: int
    repay_amount_sum_eth: float
    repay_amount_avg_eth: float
    repay_count: int
    borrow_repay_diff_eth: float
    deposit_count: int
    deposit_amount_sum_eth: float
    time_since_first_deposit: float
    withdraw_amount_sum_eth: float
    withdraw_deposit_diff_if_positive_eth: float
    liquidation_count: int
    time_since_last_liquidated: float
    liquidation_amount_sum_eth: float
    market_adx: float
    market_adxr: float
    market_apo: float
    market_aroonosc: float
    market_aroonup: float
    market_atr: float
    market_cci: float
    market_cmo: float
    market_correl: float
    market_dx: float
    market_fastk: float
    market_fastd: float
    market_ht_trendmode: int
    market_linearreg_slope: float
    market_macd_macdext: float
    market_macd_macdfix: float
    market_macd: float
    market_macdsignal_macdext: float
    market_macdsignal_macdfix: float
    market_macdsignal: float
    market_max_drawdown_365d: float
    market_natr: float
    market_plus_di: float
    market_plus_dm: float
    market_ppo: float
    market_rocp: float
    market_rocr: float


# Post
@app.post("/predict")
def predict_morosity(client_data: ClientInfo):
    """
    Predicts the morosity of a sample using the best model.

    Args
    -----
    sample: ClientInfo
        The sample to predict.

    Returns
    -----
    dict
        The prediction of the morosity.
    """
    try:
        with open(model_path, "rb") as file:
            pipeline: Pipeline = pickle.load(file)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise

    data_df = pd.DataFrame([client_data.model_dump()])

    processed_sample = prepare_data(data_df, pipeline.named_steps["preprocessor"])

    prediction = pipeline.predict(processed_sample)[0]

    return {"morosity": int(prediction)}


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=7888)
    except KeyboardInterrupt:
        print("Server stopped. Goodbye! :)")
