import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from itertools import product
import time, re

columns = ['IV_C0', 'Std_C0', 'IV_P0', 'Std_P0',
           'IV_C1', 'Std_C1', 'IV_P1', 'Std_P1',
           'IV_C2', 'Std_C2', 'IV_P2', 'Std_P2',
           '3M_Mu', '3M_Sigma', '6M_Mu', '6M_Sigma',
           'Target1', 'Target2']

D = pd.read_excel("Final.xlsx", usecols = columns)
x = D[columns[0:16]]
Target1 = D['Target1']
Target2 = D['Target2']

category = 2
dim = 16

param_grid = {
    'units': [20, 40, 60, 80],
    'epochs': [1000, 2000, 3000],
    'batch_size': [50, 100, 150, 200]
}

def create_model(input_dim, category, units):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=units, activation='relu', input_dim = input_dim))
    model.add(tf.keras.layers.Dense(units=units, activation='relu'))
    model.add(tf.keras.layers.Dense(units=category, activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}時{minutes}分{seconds}秒"

def calculate_prediction_stats(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    predicted_ones = np.sum(y_pred_binary)
    correct_ones = np.sum((y_pred_binary == 1) & (y_true == 1))
    return predicted_ones, correct_ones

def grid_search_cv(X, y, param_grid, num_trials=3):
    for i in range(31):
        start_time = time.time()
        param_combinations = list(product(param_grid['units'], param_grid['epochs'], param_grid['batch_size']))
        best_acc1 = 0
        best_params = None
        best_total_acc = 0
        best_guess1_trial = 0
        best_temp11 = 0

        kf = KFold(n_splits = num_trials, shuffle = True, random_state = 42)
        print(f"開始進行格點搜尋，共有 {len(param_combinations)} 種參數組合，進行 {num_trials} 次交叉驗證")

        for idx, (units, epochs, batch_size) in enumerate(param_combinations, 1):
            fold_scores = []
            fold_prediction_stats = []
            print(f"\n測試第 {idx}/{len(param_combinations)} 種參數組合: units={units}, epochs={epochs}, batch_size={batch_size}")

            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=category)
                y_val_categorical = tf.keras.utils.to_categorical(y_val, num_classes=category)

                model = create_model(input_dim = dim, category = category, units = units)
                model.fit(X_train, y_train_categorical, epochs = epochs, batch_size = batch_size, verbose = 0)

                y_pred = model.predict(X_val)
                predicted_ones, correct_ones = calculate_prediction_stats(y_val, np.argmax(y_pred, axis = 1))
                acc1 = correct_ones / predicted_ones if predicted_ones > 0 else 0

                fold_scores.append(acc1)
                fold_prediction_stats.append((predicted_ones, correct_ones))

                print(f"第 {fold} 次測試 - 猜1次數: {predicted_ones}, 猜1正確次數: {correct_ones}, acc1: {acc1:.4f}")

            mean_acc1 = np.mean(fold_scores)
            total_predicted_ones = np.mean([stats[0] for stats in fold_prediction_stats])
            total_correct_ones = np.mean([stats[1] for stats in fold_prediction_stats])

            total_acc = total_correct_ones / total_predicted_ones if total_predicted_ones > 0 else 0

            if mean_acc1 > best_acc1:
                best_acc1 = mean_acc1
                best_params = {'units': units, 'epochs': epochs, 'batch_size': batch_size}
                best_total_acc = total_acc
                best_guess1_trial = total_predicted_ones
                best_temp11 = total_correct_ones

            print(f"參數組合平均結果: 猜1次數: {total_predicted_ones:.2f}, 正確次數: {total_correct_ones:.2f}, 平均 acc1: {mean_acc1:.4f}")

        total_time = time.time() - start_time
        print(f"\n最佳結果：猜1次數: {best_guess1_trial:.2f}, 正確次數: {best_temp11:.2f}, 最佳 acc1: {best_acc1:.4f}")
        print(f"總執行時間: {format_time(total_time)}")

        print(f"\n最佳參數組合: units={best_params['units']}, epochs={best_params['epochs']}, batch_size={best_params['batch_size']}")
        print(f"最佳平均acc1={best_acc1:.6f}, 對應的平均總體準確率={best_total_acc:.6f}")
        print(f"最佳平均acc1時猜1的次數={best_guess1_trial}, 平均正確次數={best_temp11:.2f}")

        result = [best_acc1, best_total_acc, best_guess1_trial, best_temp11, best_params['units'], best_params['epochs'], best_params['batch_size'], format_time(total_time)]
        result_list.append(result)
        print(f'進行第{i}次')
    
    results_df = pd.DataFrame(result_list, columns = ['最佳單次acc1', '最佳平均總體準確率', '最佳猜1次數', '最佳正確次數', 'units', 'epochs', 'batch_size', '總執行時間'])

    return results_df

for target, name in zip([Target1, Target2], ['Target1', 'Target2']):
    result_list = []
    results_df = grid_search_cv(x, target, param_grid)
    clean_name = re.sub(r'[\/:*?"<>|]', '', name)
    results_df.to_excel("最佳參數組合結果.xlsx", sheet_name=clean_name, index = False)
