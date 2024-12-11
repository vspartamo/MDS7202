import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path: str) -> pd.DataFrame:
    X_t0 = pd.read_parquet(f"{data_path}/X_t0.parquet")
    y_t0 = pd.read_parquet(f"{data_path}/y_t0.parquet")
    return pd.concat([X_t0, y_t0], axis=1)

def clean_data(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    return df.drop(columns=columns_to_drop, inplace=False)

def split_data(X, y, train_size=0.7, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test