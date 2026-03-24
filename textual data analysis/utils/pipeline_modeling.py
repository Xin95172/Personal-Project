import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Algorithms
from algos.svm import SVMClassifier
from algos.random_forest import RFClassifier
from algos.MNIR import load_or_fit_feature_splits, summarize_predictions

def evaluate_all_models(
    dtm_bow_path="../lexicon_resources/dtm_csr_BoW.npz",
    dtm_tfidf_path="../lexicon_resources/dtm_csr_TF_IDF.npz",
    verdict_results_path="../artifacts/reports/verdict_results.xlsx"
):
    """
    執行所有的模型訓練與驗證 (儀表板專用版)，
    目前內部以 SVM 與 Random Forest 為例，您可以依照需求擴充加入 NN 或 NB。
    
    Returns:
        metrics_df: 包含各個模型表現的 dataframe
    """
    print("1. 載入特徵矩陣 (Features) 與判決標籤 (Labels)...")
    if not os.path.exists(dtm_bow_path):
        print(f"找不到檔案 {dtm_bow_path}，請確認是否已經生成 DTM。")
        return pd.DataFrame()
        
    dtm_csr_BoW = sp.load_npz(dtm_bow_path)
    verdict_results = pd.read_excel(verdict_results_path, index_col=0).to_numpy()

    print("2. 拆分訓練、驗證與測試集 (Train/Val/Test Split)...")
    x_train_bow, x_temp_bow, y_train_bow, y_temp_bow = train_test_split(
        dtm_csr_BoW, verdict_results, test_size=0.3, random_state=42
    )
    x_val_bow, x_test_bow, y_val_bow, y_test_bow = train_test_split(
        x_temp_bow, y_temp_bow, test_size=0.66666, random_state=42
    )

    y_train_bow = np.asarray(y_train_bow).ravel()
    y_val_bow = np.asarray(y_val_bow).ravel()
    y_test_bow = np.asarray(y_test_bow).ravel()

    print(f"訓練集大小: {x_train_bow.shape}, 測試集大小: {x_test_bow.shape}")

    print("3. 透過 MNIR 轉換特徵 (Feature Extraction)...")
    feature_res = load_or_fit_feature_splits(
        X_train=x_train_bow,
        y_train=y_train_bow,
        X_val=x_val_bow,
        X_test=x_test_bow,
        train_name="mnir_z_train_bow.npy",
        val_name="mnir_z_val_bow.npy",
        test_name="mnir_z_test_bow.npy",
        model_name="mnir_mnlm_bow.rds",
    )
    
    z_train_bow = feature_res["z_train"]
    z_val_bow = feature_res["z_val"]
    z_test_bow = feature_res["z_test"]
    
    print("4. 訓練並驗證機器學習模型 (在此示範 SVM)...")
    
    # Simple SVM grid validation
    svm_grid = sorted([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    best_score = -np.inf
    best_params = None
    
    results_list = []
    
    print("開始 Grid Search SVM...")
    for c in svm_grid:
        clf = SVMClassifier(C=c, max_iter=10000)
        clf.fit(z_train_bow, y_train_bow)
        y_pred = clf.predict(z_val_bow)
        
        acc = accuracy_score(y_val_bow, y_pred)
        macro_f1 = f1_score(y_val_bow, y_pred, average="macro")
        
        results_list.append({"Model": f"SVM (C={c})", "Accuracy": acc, "Macro F1": macro_f1})
        
        if macro_f1 > best_score:
            best_score = macro_f1
            best_params = {'C': c, 'max_iter': 10000}
            
    print(f"最佳 SVM 參數為: {best_params}，驗證集 Macro F1: {round(best_score, 4)}")
    
    print("5. 進行最終模型測試 (Test Set)...")
    final_clf = SVMClassifier(**best_params)
    final_clf.fit(z_train_bow, y_train_bow)
    y_test_pred = final_clf.predict(z_test_bow)
    
    test_acc = accuracy_score(y_test_bow, y_test_pred)
    test_macro_f1 = f1_score(y_test_bow, y_test_pred, average="macro")
    
    metrics_df = pd.DataFrame(results_list)
    print(f"✅ 建模管線完成！最終測試集準確率: {test_acc:.4f}")
    
    return metrics_df

if __name__ == "__main__":
    df_metrics = evaluate_all_models()
    print(df_metrics)
