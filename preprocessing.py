import pandas as pd
import numpy as np
from typing import Union, List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

DATA_PATH = "data/"
COLUMNS_TO_DROP = [
    'borrow_block_number',
    'wallet_address',
    'borrow_timestamp',
    'first_tx_timestamp',
    'last_tx_timestamp',
    'risky_first_tx_timestamp',
    'risky_last_tx_timestamp',
    'unique_borrow_protocol_count',
    'unique_lending_protocol_count',
]

# Función para cargar los datos
def load_data(data_path: str) -> pd.DataFrame:
    """Carga y concatena los datos de entrada y salida desde archivos parquet."""
    X_t0 = pd.read_parquet(f"{data_path}/X_t0.parquet")
    y_t0 = pd.read_parquet(f"{data_path}/y_t0.parquet")
    return pd.concat([X_t0, y_t0], axis=1)

# Función para crear el preprocesador
def create_preprocessor(
    numeric_features: Union[pd.Index, List[str]],
    categorical_features: Union[pd.Index, List[str]],
    scaler=None,
    use_pca=False,
    pca_components=50,
):
    """Crea un preprocesador para transformar características numéricas y categóricas."""
    if scaler is None:
        scaler = StandardScaler()

    # Define transformaciones para características numéricas
    numeric_transformer_steps = [("scaler", scaler)]
    if use_pca:
        numeric_transformer_steps.append(("pca", PCA(n_components=pca_components)))
    numeric_transformer = Pipeline(steps=numeric_transformer_steps)

    # Define transformaciones para características categóricas
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Combina transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor

def create_pipeline(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
):
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])




########################
df_t0 = load_data(DATA_PATH)
df_t0_cleaned = df_t0.drop(columns=COLUMNS_TO_DROP, inplace=False)

numeric_features = df_t0_cleaned.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df_t0_cleaned.select_dtypes(include=['object']).columns

X = df_t0_cleaned.drop(columns='target')
y = df_t0_cleaned['target']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=42
)

preprocessor = create_preprocessor(
    numeric_features=numeric_features.drop('target'),
    categorical_features=categorical_features,
    use_pca=False,  # usar pca?
    pca_components=50 
)

print(f"Tamaño de entrenamiento: {X_train.shape}")
print(f"Tamaño de validación: {X_val.shape}")
print(f"Tamaño de prueba: {X_test.shape}")
