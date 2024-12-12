import pandas as pd
import shap
import pickle

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from utils import preprocess_data, detect_drift, retrain_model


def interpretability_analysis():
    """Análisis de interpretabilidad"""
    # Cargar modelo reentrenado
    with open("/opt/airflow/models/retrained_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Cargar datos de prueba
    df = pd.read_csv("/opt/airflow/data/cleaned_data.csv")
    X = df.drop("target", axis=1)

    # Análisis SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Guardar gráficos de SHAP
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False,
        plot_size=(10, 6),
        filename="/opt/airflow/data/shap_summary.png",
    )


def extract_data(dataset_url, save_path):
    import requests

    response = requests.get(dataset_url)
    with open(save_path, "wb") as file:
        file.write(response.content)
    print(f"Dataset saved to {save_path}")


def track_interpretability():
    print("Tracking model interpretability...")
    # Placeholder for interpretability tracking logic
    
def select_branch_from_drift(drift_value):
    """Selecciona la rama a seguir basado en los valores de drift"""
    if drift_value is True:
        return "retrain_model"
    return "skip_retrain"


# Default arguments for the DAG
args = {
    "owner": "Team NPR",
    "retries": 1,
}

# Define DAG
with DAG(
    dag_id="ml_pipeline_example",
    default_args=args,
    description="A ML pipeline example with Airflow",
    start_date=days_ago(1),
    schedule_interval="@weekly",
    catchup=False,
) as dag:

    start_step = EmptyOperator(task_id="Start")

    extract_data_step = PythonOperator(
        task_id="extract_data_1",
        python_callable=extract_data,
        op_kwargs={
            "dataset_url": "https://example.com/data1.csv",
            "save_path": "/path/to/data1.csv",
        },
    )

    clean_data = PythonOperator(
        task_id="clean_data_1",
        python_callable=preprocess_data,
        op_kwargs={"data_path": "/path/to/data1.csv"},
    )

    # Analyze data drift
    analyze_drift_step = BranchPythonOperator(
        task_id="analyze_data_drift",
        python_callable=analyze_data_drift,
        op_kwargs={"data_paths": ["/path/to/data1.csv", "/path/to/data2.csv"]},
    )

    retrain_step = PythonOperator(
        task_id="retrain_model", python_callable=retrain_model
    )

    skip_retrain_step = EmptyOperator(task_id="skip_retrain")

    # Track interpretability
    interpretability_step = PythonOperator(
        task_id="track_interpretability", python_callable=track_interpretability
    )

    end_step = EmptyOperator(task_id="End")

    # Define task dependencies
    start_step >> extract_data_step >> 
    extract_step_1 >> clean_step_1
    extract_step_2 >> clean_step_2
    [clean_step_1, clean_step_2] >> analyze_drift_step
    analyze_drift_step >> [retrain_step, skip_retrain_step]
    [retrain_step, skip_retrain_step] >> interpretability_step
    interpretability_step >> end_step
