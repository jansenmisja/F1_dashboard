import os
from datetime import datetime
import numpy as np
import pandas as pd

from utils.helpers import backup_file

# Absoluut pad naar je centrale modellog
LOG_PATH = "/content/drive/MyDrive/F1_ML_Project/log/model_log.csv"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log_model_run(
    actie: str,
    beschrijving: str,
    train_mae: float,
    train_rmse: float,
    train_r2: float,
    cv_mae: float,
    cv_rmse: float,
    cv_r2: float,
    cv_splits: int,
    test_mae: float = np.nan,
    test_rmse: float = np.nan,
    test_r2: float = np.nan,
    opmerkingen: str = "",
) -> None:
    """
    Log één model-run naar model_log.csv (append-only).

    Kolommen:
    ['Datum','Actie','Beschrijving',
     'Test_MAE','Test_RMSE','Test_R2',
     'CV_MAE','CV_RMSE','CV_R2',
     'Opmerkingen',
     'Train_MAE','Train_RMSE','Train_R2',
     'CV_splits']
    """

    row = pd.DataFrame(
        [{
            "Datum": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Actie": actie,
            "Beschrijving": beschrijving,
            "Test_MAE": float(test_mae) if not pd.isna(test_mae) else np.nan,
            "Test_RMSE": float(test_rmse) if not pd.isna(test_rmse) else np.nan,
            "Test_R2": float(test_r2) if not pd.isna(test_r2) else np.nan,
            "CV_MAE": float(cv_mae),
            "CV_RMSE": float(cv_rmse),
            "CV_R2": float(cv_r2),
            "Opmerkingen": opmerkingen,
            "Train_MAE": float(train_mae),
            "Train_RMSE": float(train_rmse),
            "Train_R2": float(train_r2),
            "CV_splits": int(cv_splits),
        }]
    )

    # Backup + append-only schrijven
    if os.path.exists(LOG_PATH):
        backup_file(LOG_PATH)
        header = False
    else:
        header = True

    row.to_csv(LOG_PATH, mode="a", header=header, index=False)
    print(f"[LOG] Model-run gelogd naar: {LOG_PATH}")
