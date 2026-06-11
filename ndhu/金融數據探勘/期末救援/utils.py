from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as ExcelImage
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# 使用者指定的模型特徵欄位。
# 注意：欄位順序會被保留，後續 StandardScaler 與 MLPClassifier 都使用此順序。
FEATURE_COLUMNS = [
    "3M_Mu",
    "6M_Mu",
    "3M_Sigma",
    "6M_Sigma",
    "3M_Downsiderisk",
    "6M_Downsiderisk",
    "IV_C0",
    "IV_C1",
    "IV_C2",
    "IV_C3",
    "IV_C4",
    "IV_P0",
    "IV_P1",
    "IV_P2",
    "IV_P3",
    "IV_P4",
    "Std_C0",
    "Std_C1",
    "Std_C2",
    "Std_C3",
    "Std_C4",
    "Std_P0",
    "Std_P1",
    "Std_P2",
    "Std_P3",
    "Std_P4",
]


# 原始 Excel 第 2 個工作表中，Target1 的實際欄名較長。
# 程式保留多個候選名稱，讓報告中寫 Target1 時也能正常執行。
TARGET_CANDIDATES = [
    "Target1",
    "Target1 (>=1.015 OR<=0.985)",
    "Target1 (>=1.015 OR <=0.985)",
    "Target1 (>=1.015 OR <= 0.985)",
]


# 不同工作表可能因 Excel 合併或命名習慣產生欄名差異。
# 這裡把等價欄位統一成模型使用的標準欄名。
COLUMN_ALIASES = {
    "3M_DownsideRisk": "3M_Downsiderisk",
    "6M_DownsideRisk": "6M_Downsiderisk",
    "6M_Mu_1": "6M_Mu",
    "6M_Sigma_1": "6M_Sigma",
}


# ===== 神經網路架構與訓練參數 =====
#
# sklearn 的 MLPClassifier 不需要手動指定輸入層與輸出層：
# 1. 輸入層神經元數 = X 的特徵數，本研究為 len(FEATURE_COLUMNS) = 26。
# 2. 輸出層神經元數 = y 的類別數，本研究為 Target1 的 0/1 二元分類，所以是 2 類。
# 3. 隱藏層神經元數則由 hidden_layer_sizes 明確指定。
INPUT_LAYER_SIZE = len(FEATURE_COLUMNS)
OUTPUT_LAYER_SIZE = 2
HIDDEN_LAYER_SIZES = (32, 16)
ACTIVATION = "relu"
SOLVER = "adam"
ALPHA = 0.0001
LEARNING_RATE_INIT = 0.001
MAX_ITER = 1000
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.15
N_ITER_NO_CHANGE = 30


def find_target_column(df: pd.DataFrame) -> str:
    """尋找 Target1 欄位名稱。"""
    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"找不到 Target1 欄位，現有欄位為：{list(df.columns)}")


def load_sheet_data(excel_path: Path | str, sheet_name: int | str = 1) -> pd.DataFrame:
    """讀取指定 Excel 工作表，並依日期順序保留原始時間序列。

    pandas 的 sheet_name 若使用整數，會從 0 開始計算：
    sheet_name=0 代表第 1 個工作表，sheet_name=1 代表第 2 個工作表。
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"找不到資料檔：{excel_path}")

    workbook = pd.ExcelFile(excel_path)
    if isinstance(sheet_name, int) and (sheet_name < 0 or sheet_name >= len(workbook.sheet_names)):
        raise ValueError(
            f"指定的 sheet_name={sheet_name} 不存在。"
            f"此檔案共有 {len(workbook.sheet_names)} 個工作表：{workbook.sheet_names}"
        )

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 若有 Date 欄位，先依 Date 排序，避免 Excel 內容不是時間遞增造成時間切分錯誤。
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_second_sheet_data(excel_path: Path | str) -> pd.DataFrame:
    """讀取 Excel 第 2 個工作表；保留此函式是為了相容舊版 notebook。"""
    return load_sheet_data(excel_path, sheet_name=1)


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """取出特徵與 Target1，並移除缺漏值列。"""
    df = df.rename(columns={old: new for old, new in COLUMN_ALIASES.items() if old in df.columns})
    target_column = find_target_column(df)

    missing_features = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        raise ValueError(f"資料缺少以下特徵欄位：{missing_features}")

    model_df = df[FEATURE_COLUMNS + [target_column]].copy()

    # 將所有模型欄位轉成數值；無法轉換者會變 NaN，後續一起移除。
    for column in FEATURE_COLUMNS + [target_column]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    # 因為前幾列可能沒有 3M/6M 或 lag 特徵，這裡只移除模型必要欄位有缺漏的列。
    model_df = model_df.dropna(subset=FEATURE_COLUMNS + [target_column]).reset_index(drop=True)

    X = model_df[FEATURE_COLUMNS]
    y = model_df[target_column].astype(int)
    return X, y


def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """嚴格時間序列切分：前 80% 訓練、後 20% 測試，不做 random split。"""
    if len(X) != len(y):
        raise ValueError("X 與 y 的資料筆數不一致。")
    if len(X) < 10:
        raise ValueError("有效資料筆數太少，無法進行訓練與測試。")

    split_index = int(len(X) * train_ratio)
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("train_ratio 造成訓練集或測試集為空。")

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()
    return X_train, X_test, y_train, y_test


def scale_without_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """StandardScaler 只在訓練集 fit，再 transform 訓練集與測試集，避免 Data Leakage。"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 轉回 DataFrame，保留欄名，方便除錯與報告追蹤。
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    return X_train_scaled_df, X_test_scaled_df, scaler


def build_mlp_classifier(seed: int) -> MLPClassifier:
    """建立 sklearn MLPClassifier 二元分類模型。"""
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation=ACTIVATION,
        solver=SOLVER,
        alpha=ALPHA,
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
        random_state=seed,
    )


def get_network_architecture_info(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """回傳本研究使用的神經網路架構資訊，方便 notebook 與報告呈現。"""
    return {
        "input_layer_size": int(X.shape[1]),
        "input_features": list(X.columns),
        "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
        "output_layer_size": int(y.nunique()),
        "output_classes": sorted(y.unique().tolist()),
        "activation": ACTIVATION,
        "solver": SOLVER,
        "alpha": ALPHA,
        "learning_rate_init": LEARNING_RATE_INIT,
        "max_iter": MAX_ITER,
        "early_stopping": EARLY_STOPPING,
    }


def evaluate_thresholds_for_seed(
    seed: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float],
) -> list[dict[str, float]]:
    """單一 seed 下訓練 MLP，並掃描不同 threshold 的績效。"""
    model = build_mlp_classifier(seed)
    model.fit(X_train, y_train)

    # predict_proba 第 2 欄代表預測為 1 的機率。
    positive_probability = model.predict_proba(X_test)[:, 1]

    seed_results: list[dict[str, float]] = []
    y_test_array = y_test.to_numpy()

    for threshold in thresholds:
        # 題目指定：模型輸出的機率「大於」Threshold 則預測為 1。
        y_pred = (positive_probability > threshold).astype(int)
        true_positive_count = int(((y_pred == 1) & (y_test_array == 1)).sum())

        seed_results.append(
            {
                "Seed": seed,
                "Threshold": threshold,
                "Predicted_1_Count": int(y_pred.sum()),
                "TruePositive_Count": true_positive_count,
                # zero_division=0 可避免沒有預測 1 時 precision 無法計算。
                "Precision": precision_score(y_test_array, y_pred, zero_division=0),
                "Recall": recall_score(y_test_array, y_pred, zero_division=0),
                "F1": f1_score(y_test_array, y_pred, zero_division=0),
            }
        )

    return seed_results


def run_repeated_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seeds: range,
    thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """執行 30 個 seed 的重複實驗，並彙總每個 threshold 的平均績效。"""
    all_results: list[dict[str, float]] = []

    for seed in seeds:
        seed_results = evaluate_thresholds_for_seed(
            seed=seed,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            thresholds=thresholds,
        )
        all_results.extend(seed_results)

    detail_df = pd.DataFrame(all_results)

    # 每個 Threshold 對 30 次實驗取平均，符合題目要求。
    summary_df = (
        detail_df.groupby("Threshold", as_index=False)
        .agg(
            Avg_Pred1=("Predicted_1_Count", "mean"),
            Avg_TP=("TruePositive_Count", "mean"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            F1=("F1", "mean"),
        )
        .sort_values("Threshold")
        .reset_index(drop=True)
    )

    # 讓輸出表格易讀，但不改變計算邏輯。
    rounded_columns = ["Avg_Pred1", "Avg_TP", "Precision", "Recall", "F1"]
    summary_df[rounded_columns] = summary_df[rounded_columns].round(4)

    return summary_df, detail_df


def get_dataset_info(y_train: pd.Series, y_test: pd.Series) -> dict[str, int]:
    """整理訓練集與測試集的 Target1 分布，供輸出工作簿記錄。"""
    return {
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_target_1": int(np.sum(y_train == 1)),
        "train_target_0": int(np.sum(y_train == 0)),
        "test_target_1": int(np.sum(y_test == 1)),
        "test_target_0": int(np.sum(y_test == 0)),
    }


def build_target_stats(y: pd.Series, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """建立 Target1 的 0/1 次數統計表。"""
    rows = []
    for split_name, split_y in [
        ("All", y),
        ("Train", y_train),
        ("Test", y_test),
    ]:
        total_count = len(split_y)
        for target_value in [0, 1]:
            count = int(np.sum(split_y == target_value))
            ratio = count / total_count if total_count > 0 else 0
            rows.append(
                {
                    "Dataset": split_name,
                    "Target1": target_value,
                    "Count": count,
                    "Ratio": round(ratio, 4),
                }
            )
    return pd.DataFrame(rows)


def build_feature_stats(X: pd.DataFrame) -> pd.DataFrame:
    """建立模型輸入特徵的敘述統計表。"""
    stats_df = X.describe().T.reset_index()
    stats_df = stats_df.rename(columns={"index": "Feature"})

    # 保留報告常用欄位，避免輸出表太雜。
    stats_df = stats_df[
        ["Feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ].copy()

    numeric_columns = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    stats_df[numeric_columns] = stats_df[numeric_columns].round(6)
    return stats_df


def plot_threshold_metrics(summary_df: pd.DataFrame, output_path: Path | str) -> Path:
    """繪製 Threshold 對 Precision、Recall、F1 的折線圖。"""
    output_path = Path(output_path)

    plt.figure(figsize=(9, 5.2))
    plt.plot(summary_df["Threshold"], summary_df["Precision"], marker="o", label="Precision")
    plt.plot(summary_df["Threshold"], summary_df["Recall"], marker="s", label="Recall")
    plt.plot(summary_df["Threshold"], summary_df["F1"], marker="^", label="F1-score")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision / Recall / F1-score")
    plt.xticks(summary_df["Threshold"])
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.show()
    plt.close()

    return output_path


def export_summary_workbook(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    dataset_info: dict[str, int],
    chart_path: Path | str,
    output_path: Path | str,
    target_stats_df: pd.DataFrame | None = None,
    feature_stats_df: pd.DataFrame | None = None,
) -> Path:
    """輸出 ThresholdSummary.xlsx，包含彙總表、30 次明細、資料切分資訊與圖表。"""
    chart_path = Path(chart_path)
    output_path = Path(output_path)

    info_df = pd.DataFrame(
        [
            {"Item": "Train rows", "Value": dataset_info["train_rows"]},
            {"Item": "Test rows", "Value": dataset_info["test_rows"]},
            {"Item": "Train Target1=1", "Value": dataset_info["train_target_1"]},
            {"Item": "Train Target1=0", "Value": dataset_info["train_target_0"]},
            {"Item": "Test Target1=1", "Value": dataset_info["test_target_1"]},
            {"Item": "Test Target1=0", "Value": dataset_info["test_target_0"]},
            {"Item": "Split rule", "Value": "前 80% 訓練、後 20% 測試"},
            {"Item": "Scaler rule", "Value": "StandardScaler 只 fit 訓練集"},
        ]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        detail_df.to_excel(writer, sheet_name="SeedDetails", index=False)
        info_df.to_excel(writer, sheet_name="DatasetInfo", index=False)
        if target_stats_df is not None:
            target_stats_df.to_excel(writer, sheet_name="TargetStats", index=False)
        if feature_stats_df is not None:
            feature_stats_df.to_excel(writer, sheet_name="FeatureStats", index=False)

    # 將 matplotlib 圖片插入 Summary 工作表，方便報告直接引用。
    workbook = load_workbook(output_path)
    worksheet = workbook["Summary"]
    image = ExcelImage(chart_path)
    image.anchor = "H2"
    worksheet.add_image(image)

    # 調整 Summary 欄寬，讓主要表格可讀。
    for column_cells in worksheet.columns:
        max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 12), 22)

    workbook.save(output_path)
    return output_path
