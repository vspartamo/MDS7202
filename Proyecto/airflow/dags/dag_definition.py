from datetime import timedelta
import pandas as pd
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator


def clean_and_transform_data():
    """Limpieza y transformación de datos"""
    df = pd.read_csv("/opt/airflow/data/raw_data.csv")

    # Ejemplo de limpieza - ajustar según sus datos
    df.dropna(inplace=True)
    df = df[df["feature_column"] > 0]

    df.to_csv("/opt/airflow/data/cleaned_data.csv", index=False)


def detect_data_drift():
    """Detectar desviaciones en los datos"""
    original_data = pd.read_csv("/opt/airflow/data/original_dataset.csv")
    new_data = pd.read_csv("/opt/airflow/data/cleaned_data.csv")

    # Ejemplo simple de detección de drift - personalizar según necesidad
    drift_detected = False
    for column in original_data.columns:
        original_stats = original_data[column].describe()
        new_stats = new_data[column].describe()

        # Ejemplo de criterio de drift
        if (
            abs(original_stats["mean"] - new_stats["mean"])
            > original_stats["mean"] * 0.2
        ):
            drift_detected = True
            break

    # Guardar bandera de drift
    with open("/opt/airflow/data/drift_flag.txt", "w") as f:
        f.write(str(drift_detected))


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


args = {
    "owner": "Team NPR",
}

with DAG(
    dag_id="Proyecto MDS7202 - Pipeline",
    default_args=args,
    schedule_interval="@weekly",
    catchup=True,
) as dag:

    t1 = BashOperator(
        task_id="download_data",
        bash_command="curl -o "
        "/root/airflow/data_1_{{ ds }}.csv "
        "https://gitlab.com/imezadelajara/datos_clase_7_mds7202/-/raw/main/airflow_class/data_1.csv",
    )

    t2 = PythonOperator(task_id="clean_data", python_callable=clean_and_transform_data)

    t3 = PythonOperator(task_id="detect_drift", python_callable=detect_data_drift)

    t4 = PythonOperator(task_id="retrain_model", python_callable=retrain_model)

    t5 = PythonOperator(
        task_id="interpretability_analysis", python_callable=interpretability_analysis
    )

    # Definir dependencias
    t1 >> t2 >> t3 >> t4 >> t5
