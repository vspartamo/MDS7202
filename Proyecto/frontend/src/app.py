import gradio as gr
import pandas as pd
import requests


def predict_from_csv(csv_file):
    """
    Procesa predicciones desde un archivo CSV
    """
    try:
        df = pd.read_csv(csv_file.name)
        predictions = []

        # Verificar que el CSV tenga todas las columnas necesarias
        required_columns = [
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

        if not all(col in df.columns for col in required_columns):
            return "Error: El CSV no contiene todas las columnas requeridas"

        # Realizar predicciones para cada fila
        for _, row in df.iterrows():
            payload = row.to_dict()
            response = requests.post("http://backend:7888/predict", json=payload)
            predictions.append(response.json()["morosity"])

        return pd.DataFrame({"Datos": df.index, "Predicción de Morosidad": predictions})

    except Exception as e:
        return f"Error al procesar el archivo: {str(e)}"


def predict_manual(
    wallet_age,
    incoming_tx_count,
    outgoing_tx_count,
    net_incoming_tx_count,
    total_gas_paid_eth,
    avg_gas_paid_per_tx_eth,
    risky_tx_count,
    risky_unique_contract_count,
    risky_first_last_tx_timestamp_diff,
    risky_sum_outgoing_amount_eth,
    outgoing_tx_sum_eth,
    incoming_tx_sum_eth,
    outgoing_tx_avg_eth,
    incoming_tx_avg_eth,
    max_eth_ever,
    min_eth_ever,
    total_balance_eth,
    risk_factor,
    total_collateral_eth,
    total_collateral_avg_eth,
    total_available_borrows_eth,
    total_available_borrows_avg_eth,
    avg_weighted_risk_factor,
    risk_factor_above_threshold_daily_count,
    avg_risk_factor,
    max_risk_factor,
    borrow_amount_sum_eth,
    borrow_amount_avg_eth,
    borrow_count,
    repay_amount_sum_eth,
    repay_amount_avg_eth,
    repay_count,
    borrow_repay_diff_eth,
    deposit_count,
    deposit_amount_sum_eth,
    time_since_first_deposit,
    withdraw_amount_sum_eth,
    withdraw_deposit_diff_if_positive_eth,
    liquidation_count,
    time_since_last_liquidated,
    liquidation_amount_sum_eth,
    market_adx,
    market_adxr,
    market_apo,
    market_aroonosc,
    market_aroonup,
    market_atr,
    market_cci,
    market_cmo,
    market_correl,
    market_dx,
    market_fastk,
    market_fastd,
    market_ht_trendmode,
    market_linearreg_slope,
    market_macd_macdext,
    market_macd_macdfix,
    market_macd,
    market_macdsignal_macdext,
    market_macdsignal_macdfix,
    market_macdsignal,
    market_max_drawdown_365d,
    market_natr,
    market_plus_di,
    market_plus_dm,
    market_ppo,
    market_rocp,
    market_rocr,
):
    """
    Realiza predicción con datos ingresados manualmente
    """
    try:
        payload = {
            "wallet_age": wallet_age,
            "incoming_tx_count": incoming_tx_count,
            "outgoing_tx_count": outgoing_tx_count,
            "net_incoming_tx_count": net_incoming_tx_count,
            "total_gas_paid_eth": total_gas_paid_eth,
            "avg_gas_paid_per_tx_eth": avg_gas_paid_per_tx_eth,
            "risky_tx_count": risky_tx_count,
            "risky_unique_contract_count": risky_unique_contract_count,
            "risky_first_last_tx_timestamp_diff": risky_first_last_tx_timestamp_diff,
            "risky_sum_outgoing_amount_eth": risky_sum_outgoing_amount_eth,
            "outgoing_tx_sum_eth": outgoing_tx_sum_eth,
            "incoming_tx_sum_eth": incoming_tx_sum_eth,
            "outgoing_tx_avg_eth": outgoing_tx_avg_eth,
            "incoming_tx_avg_eth": incoming_tx_avg_eth,
            "max_eth_ever": max_eth_ever,
            "min_eth_ever": min_eth_ever,
            "total_balance_eth": total_balance_eth,
            "risk_factor": risk_factor,
            "total_collateral_eth": total_collateral_eth,
            "total_collateral_avg_eth": total_collateral_avg_eth,
            "total_available_borrows_eth": total_available_borrows_eth,
            "total_available_borrows_avg_eth": total_available_borrows_avg_eth,
            "avg_weighted_risk_factor": avg_weighted_risk_factor,
            "risk_factor_above_threshold_daily_count": risk_factor_above_threshold_daily_count,
            "avg_risk_factor": avg_risk_factor,
            "max_risk_factor": max_risk_factor,
            "borrow_amount_sum_eth": borrow_amount_sum_eth,
            "borrow_amount_avg_eth": borrow_amount_avg_eth,
            "borrow_count": borrow_count,
            "repay_amount_sum_eth": repay_amount_sum_eth,
            "repay_amount_avg_eth": repay_amount_avg_eth,
            "repay_count": repay_count,
            "borrow_repay_diff_eth": borrow_repay_diff_eth,
            "deposit_count": deposit_count,
            "deposit_amount_sum_eth": deposit_amount_sum_eth,
            "time_since_first_deposit": time_since_first_deposit,
            "withdraw_amount_sum_eth": withdraw_amount_sum_eth,
            "withdraw_deposit_diff_if_positive_eth": withdraw_deposit_diff_if_positive_eth,
            "liquidation_count": liquidation_count,
            "time_since_last_liquidated": time_since_last_liquidated,
            "liquidation_amount_sum_eth": liquidation_amount_sum_eth,
            "market_adx": market_adx,
            "market_adxr": market_adxr,
            "market_apo": market_apo,
            "market_aroonosc": market_aroonosc,
            "market_aroonup": market_aroonup,
            "market_atr": market_atr,
            "market_cci": market_cci,
            "market_cmo": market_cmo,
            "market_correl": market_correl,
            "market_dx": market_dx,
            "market_fastk": market_fastk,
            "market_fastd": market_fastd,
            "market_ht_trendmode": market_ht_trendmode,
            "market_linearreg_slope": market_linearreg_slope,
            "market_macd_macdext": market_macd_macdext,
            "market_macd_macdfix": market_macd_macdfix,
            "market_macd": market_macd,
            "market_macdsignal_macdext": market_macdsignal_macdext,
            "market_macdsignal_macdfix": market_macdsignal_macdfix,
            "market_macdsignal": market_macdsignal,
            "market_max_drawdown_365d": market_max_drawdown_365d,
            "market_natr": market_natr,
            "market_plus_di": market_plus_di,
            "market_plus_dm": market_plus_dm,
            "market_ppo": market_ppo,
            "market_rocp": market_rocp,
            "market_rocr": market_rocr,
        }

        # Añadir timeout y manejo de respuesta
        response = requests.post(
            "http://backend:7888/predict", json=payload, timeout=10
        )

        return response.json()["morosity"]

    except Exception as e:
        return f"Error en la predicción: {str(e)}"


# Definir la interfaz de Gradio
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Predictor de Morosidad")

        with gr.Tab("Predicción Manual"):
            with gr.Row():
                wallet_age = gr.Number(label="Edad de Billetera")
                incoming_tx_count = gr.Number(label="Número de Transacciones Entrantes")
                outgoing_tx_count = gr.Number(label="Número de Transacciones Salientes")
                net_incoming_tx_count = gr.Number(
                    label="Diferencia de Transacciones Entrantes"
                )
                total_gas_paid_eth = gr.Number(label="Total de Gas Pagado en ETH")
                avg_gas_paid_per_tx_eth = gr.Number(
                    label="Gas Promedio Pagado por Transacción en ETH"
                )
                risky_tx_count = gr.Number(label="Número de Transacciones Riesgosas")
                risky_unique_contract_count = gr.Number(
                    label="Número de Contratos Únicos Riesgosos"
                )
                risky_first_last_tx_timestamp_diff = gr.Number(
                    label="Diferencia de Tiempo entre la primera y última transacción riesgosa"
                )
                risky_sum_outgoing_amount_eth = gr.Number(
                    label="Suma de Montos Salientes Riesgosos en ETH"
                )
                outgoing_tx_sum_eth = gr.Number(label="Suma de Montos Salientes en ETH")
                incoming_tx_sum_eth = gr.Number(label="Suma de Montos Entrantes en ETH")
                outgoing_tx_avg_eth = gr.Number(
                    label="Promedio de Montos Salientes en ETH"
                )
                incoming_tx_avg_eth = gr.Number(
                    label="Promedio de Montos Entrantes en ETH"
                )
                max_eth_ever = gr.Number(label="Máximo ETH alguna vez")
                min_eth_ever = gr.Number(label="Mínimo ETH alguna vez")
                total_balance_eth = gr.Number(label="Balance Total en ETH")
                risk_factor = gr.Number(label="Factor de Riesgo")
                total_collateral_eth = gr.Number(label="Total de Colateral en ETH")
                total_collateral_avg_eth = gr.Number(
                    label="Promedio de Colateral en ETH"
                )
                total_available_borrows_eth = gr.Number(
                    label="Total de Préstamos Disponibles en ETH"
                )
                total_available_borrows_avg_eth = gr.Number(
                    label="Promedio de Préstamos Disponibles en ETH"
                )
                avg_weighted_risk_factor = gr.Number(
                    label="Promedio Ponderado del Factor de Riesgo"
                )
                risk_factor_above_threshold_daily_count = gr.Number(
                    label="Conteo Diario de Factor de Riesgo por Encima del Umbral"
                )
                avg_risk_factor = gr.Number(label="Factor de Riesgo Promedio")
                max_risk_factor = gr.Number(label="Factor de Riesgo Máximo")
                borrow_amount_sum_eth = gr.Number(
                    label="Suma de Montos Prestados en ETH"
                )
                borrow_amount_avg_eth = gr.Number(
                    label="Promedio de Montos Prestados en ETH"
                )
                borrow_count = gr.Number(label="Conteo de Préstamos")
                repay_amount_sum_eth = gr.Number(label="Suma de Montos Pagados en ETH")
                repay_amount_avg_eth = gr.Number(
                    label="Promedio de Montos Pagados en ETH"
                )
                repay_count = gr.Number(label="Conteo de Pagos")
                borrow_repay_diff_eth = gr.Number(
                    label="Diferencia de Préstamos y Pagos en ETH"
                )
                deposit_count = gr.Number(label="Conteo de Depósitos")
                deposit_amount_sum_eth = gr.Number(
                    label="Suma de Montos Depositados en ETH"
                )
                time_since_first_deposit = gr.Number(
                    label="Tiempo desde el Primer Depósito"
                )
                withdraw_amount_sum_eth = gr.Number(
                    label="Suma de Montos Retirados en ETH"
                )
                withdraw_deposit_diff_if_positive_eth = gr.Number(
                    label="Diferencia de Retiros y Depósitos si es Positiva en ETH"
                )
                liquidation_count = gr.Number(label="Conteo de Liquidaciones")
                time_since_last_liquidated = gr.Number(
                    label="Tiempo desde la Última Liquidación"
                )
                liquidation_amount_sum_eth = gr.Number(
                    label="Suma de Montos Liquidados en ETH"
                )
                market_adx = gr.Number(label="ADX de Mercado")
                market_adxr = gr.Number(label="ADXR de Mercado")
                market_apo = gr.Number(label="APO de Mercado")
                market_aroonosc = gr.Number(label="Aroon Oscilador de Mercado")
                market_aroonup = gr.Number(label="Aroon Up de Mercado")
                market_atr = gr.Number(label="ATR de Mercado")
                market_cci = gr.Number(label="CCI de Mercado")
                market_cmo = gr.Number(label="CMO de Mercado")
                market_correl = gr.Number(label="Correlación de Mercado")
                market_dx = gr.Number(label="DX de Mercado")
                market_fastk = gr.Number(label="FastK de Mercado")
                market_fastd = gr.Number(label="FastD de Mercado")
                market_ht_trendmode = gr.Number(label="HT Trend Mode de Mercado")
                market_linearreg_slope = gr.Number(
                    label="Linear Regression Slope de Mercado"
                )
                market_macd_macdext = gr.Number(label="MACD de Mercado")
                market_macd_macdfix = gr.Number(label="MACD Fix de Mercado")
                market_macd = gr.Number(label="MACD de Mercado")
                market_macdsignal_macdext = gr.Number(label="MACD Signal de Mercado")
                market_macdsignal_macdfix = gr.Number(
                    label="MACD Signal Fix de Mercado"
                )
                market_macdsignal = gr.Number(label="MACD Signal de Mercado")
                market_max_drawdown_365d = gr.Number(
                    label="Máximo Drawdown de Mercado en 365 días"
                )
                market_natr = gr.Number(label="NATR de Mercado")
                market_plus_di = gr.Number(label="Plus DI de Mercado")
                market_plus_dm = gr.Number(label="Plus DM de Mercado")
                market_ppo = gr.Number(label="PPO de Mercado")
                market_rocp = gr.Number(label="ROCP de Mercado")
                market_rocr = gr.Number(label="ROCR de Mercado")

            predict_button = gr.Button("Predecir Morosidad")
            manual_output = gr.Textbox(label="Resultado")

            predict_button.click(
                fn=predict_manual,
                inputs=[
                    wallet_age,
                    incoming_tx_count,
                    outgoing_tx_count,
                    net_incoming_tx_count,
                    total_gas_paid_eth,
                    avg_gas_paid_per_tx_eth,
                    risky_tx_count,
                    risky_unique_contract_count,
                    risky_first_last_tx_timestamp_diff,
                    risky_sum_outgoing_amount_eth,
                    outgoing_tx_sum_eth,
                    incoming_tx_sum_eth,
                    outgoing_tx_avg_eth,
                    incoming_tx_avg_eth,
                    max_eth_ever,
                    min_eth_ever,
                    total_balance_eth,
                    risk_factor,
                    total_collateral_eth,
                    total_collateral_avg_eth,
                    total_available_borrows_eth,
                    total_available_borrows_avg_eth,
                    avg_weighted_risk_factor,
                    risk_factor_above_threshold_daily_count,
                    avg_risk_factor,
                    max_risk_factor,
                    borrow_amount_sum_eth,
                    borrow_amount_avg_eth,
                    borrow_count,
                    repay_amount_sum_eth,
                    repay_amount_avg_eth,
                    repay_count,
                    borrow_repay_diff_eth,
                    deposit_count,
                    deposit_amount_sum_eth,
                    time_since_first_deposit,
                    withdraw_amount_sum_eth,
                    withdraw_deposit_diff_if_positive_eth,
                    liquidation_count,
                    time_since_last_liquidated,
                    liquidation_amount_sum_eth,
                    market_adx,
                    market_adxr,
                    market_apo,
                    market_aroonosc,
                    market_aroonup,
                    market_atr,
                    market_cci,
                    market_cmo,
                    market_correl,
                    market_dx,
                    market_fastk,
                    market_fastd,
                    market_ht_trendmode,
                    market_linearreg_slope,
                    market_macd_macdext,
                    market_macd_macdfix,
                    market_macd,
                    market_macdsignal_macdext,
                    market_macdsignal_macdfix,
                    market_macdsignal,
                    market_max_drawdown_365d,
                    market_natr,
                    market_plus_di,
                    market_plus_dm,
                    market_ppo,
                    market_rocp,
                    market_rocr,
                ],
                outputs=manual_output,
            )

        with gr.Tab("Predicción por CSV"):
            csv_input = gr.File(
                label="Cargar Archivo CSV", file_types=["csv"], file_count="single"
            )
            predict_csv_button = gr.Button("Predecir desde CSV")
            csv_output = gr.Dataframe(label="Resultados")

            predict_csv_button.click(
                fn=predict_from_csv, inputs=csv_input, outputs=csv_output
            )

    return demo


# Iniciar la aplicación
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
