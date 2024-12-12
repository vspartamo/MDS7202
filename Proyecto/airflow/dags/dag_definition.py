import pandas as pd
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator


def retrain_model():
    """Reentrenar modelo si se detecta drift"""
    with open("/opt/airflow/data/drift_flag.txt", "r") as f:
        drift_detected = f.read().strip() == "True"

    if drift_detected:
        # Cargar datos
        df = pd.read_csv("/opt/airflow/data/cleaned_data.csv")

        # Preparar datos
        X = df.drop("target", axis=1)
        y = df["target"]

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrenar modelo
        model = RandomForestClassifier()
        model.fit(X_train_scaled, y_train)

        # Evaluar modelo
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)

        # Guardar modelo
        with open("/opt/airflow/models/retrained_model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Guardar reporte
        with open("/opt/airflow/models/model_performance.txt", "w") as f:
            f.write(report)


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


def clean_data(data_path):
    import pandas as pd

    df = pd.read_csv(data_path)
    df.dropna(inplace=True)  # Simple cleaning step
    df.to_csv(data_path, index=False)
    print(f"Cleaned data saved to {data_path}")


def analyze_data_drift(data_paths):
    print("Analyzing data drift...")
    # Placeholder for drift analysis logic
    return "retrain_model" if True else "skip_retrain"  # Simulated condition


def retrain_model():
    print("Retraining model...")
    # Placeholder for model retraining logic


def track_interpretability():
    print("Tracking model interpretability...")
    # Placeholder for interpretability tracking logic


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
    schedule_interval="@daily",
    catchup=False,
) as dag:

    start_task = EmptyOperator(task_id="Start")

    # Extract data tasks
    extract_task_1 = PythonOperator(
        task_id="extract_data_1",
        python_callable=extract_data,
        op_kwargs={
            "dataset_url": "https://example.com/data1.csv",
            "save_path": "/path/to/data1.csv",
        },
    )

    extract_task_2 = PythonOperator(
        task_id="extract_data_2",
        python_callable=extract_data,
        op_kwargs={
            "dataset_url": "https://example.com/data2.csv",
            "save_path": "/path/to/data2.csv",
        },
    )

    # Clean data tasks
    clean_task_1 = PythonOperator(
        task_id="clean_data_1",
        python_callable=clean_data,
        op_kwargs={"data_path": "/path/to/data1.csv"},
    )

    clean_task_2 = PythonOperator(
        task_id="clean_data_2",
        python_callable=clean_data,
        op_kwargs={"data_path": "/path/to/data2.csv"},
    )

    # Analyze data drift
    analyze_drift_task = BranchPythonOperator(
        task_id="analyze_data_drift",
        python_callable=analyze_data_drift,
        op_kwargs={"data_paths": ["/path/to/data1.csv", "/path/to/data2.csv"]},
    )

    retrain_task = PythonOperator(
        task_id="retrain_model", python_callable=retrain_model
    )

    skip_retrain_task = EmptyOperator(task_id="skip_retrain")

    # Track interpretability
    interpretability_task = PythonOperator(
        task_id="track_interpretability", python_callable=track_interpretability
    )

    end_task = EmptyOperator(task_id="End")

    # Define task dependencies
    start_task >> [extract_task_1, extract_task_2]
    extract_task_1 >> clean_task_1
    extract_task_2 >> clean_task_2
    [clean_task_1, clean_task_2] >> analyze_drift_task
    analyze_drift_task >> [retrain_task, skip_retrain_task]
    [retrain_task, skip_retrain_task] >> interpretability_task
    interpretability_task >> end_task
