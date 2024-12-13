from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd

model_path = "models/best_model.pkl"
app = FastAPI()

ORDERED_COLUMNS = [
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
    # 1. Load the model
    try:
        with open(model_path, "rb") as file:
            pipeline: Pipeline = pickle.load(file)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise

    # 2. Prepare the data with de model dump
    data_df = pd.DataFrame(client_data.model_dump(), index=[0])
    data_df_clean = data_df.drop(FEATURES_TO_DROP, axis=1, errors="ignore")
    processed_sample = pipeline.named_steps["preprocessor"].transform(data_df_clean)

    # 3. Create dataframe with processed sample and names of columns
    processed_sample_df = pd.DataFrame(
        processed_sample,
        columns=ORDERED_COLUMNS,
    )
    # 4. Make the prediction
    prediction = pipeline.predict(processed_sample_df)
    # 5. Return the prediction
    return {"morosity": int(prediction)}


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=7888)
    except KeyboardInterrupt:
        print("Server stopped. Goodbye! :)")
