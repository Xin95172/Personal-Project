import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from itertools import product
import time, os

def create_model(input_dim, category, units):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=units, activation='relu', input_dim=input_dim))
    model.add(tf.keras.layers.Dense(units=units, activation='relu'))
    model.add(tf.keras.layers.Dense(units=category, activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}時{minutes}分{seconds}秒"

def grid_search_cv(X, y, param_grid, num_trials=2):
    param_combinations = list(product(param_grid['units'], param_grid['epochs'], param_grid['batch_size']))
    best_acc1 = 0
    best_params = None

    kf = KFold(n_splits=num_trials, shuffle=True, random_state=42)
    for units, epochs, batch_size in param_combinations:
        fold_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=category)
            y_val_categorical = tf.keras.utils.to_categorical(y_val, num_classes=category)

            model = create_model(input_dim=dim, category=category, units=units)
            model.fit(X_train, y_train_categorical, epochs=epochs, batch_size=batch_size, verbose=0)

            y_pred = model.predict(X_val)
            acc1 = np.mean(np.argmax(y_pred, axis=1) == y_val)
            fold_scores.append(acc1)

        mean_acc1 = np.mean(fold_scores)

        if mean_acc1 > best_acc1:
            best_acc1 = mean_acc1
            best_params = {'units': units, 'epochs': epochs, 'batch_size': batch_size}

    return best_acc1, best_params

def repeat_grid_search(X, y, param_grid, repeat_times=2):
    results = []

    for _ in range(repeat_times):
        start_time = time.time()
        best_acc1, best_params = grid_search_cv(X, y, param_grid)
        total_time = time.time() - start_time
        results.append({
            '最佳 acc1': best_acc1,
            'units': best_params['units'],
            'epochs': best_params['epochs'],
            'batch_size': best_params['batch_size'],
            '總時間': total_time
        })

    return pd.DataFrame(results)

folder_path = ""

columns = ['IV_C0', 'Std_C0', 'IV_P0', 'Std_P0',
           'IV_C1', 'Std_C1', 'IV_P1', 'Std_P1',
           'IV_C2', 'Std_C2', 'IV_P2', 'Std_P2',
           '3M_Mu', '3M_Sigma', '6M_Mu', '6M_Sigma', 'Target1', 'Target2']

category = 2
dim = 16
param_grid = {
    'units': [40, 60],
    'epochs': [1000, 2000],
    'batch_size': [100, 150]
}

with pd.ExcelWriter("格點搜尋總結果.xlsx") as writer:
    for filename in os.listdir(folder_path):
        if filename.endswith("5269.xlsx") or filename.endswith("Final_Chen.xlsx"):
            file_path = os.path.join(folder_path, filename)
            print(f"正在處理檔案: {filename}")

            D = pd.read_excel(file_path, usecols=columns)
            x = D[columns[0:16]]
            y_target1 = D['Target1']
            y_target2 = D['Target2']

            print("開始 Target1 的格點搜尋...")
            target1_results = repeat_grid_search(x, y_target1, param_grid)

            print("開始 Target2 的格點搜尋...")
            target2_results = repeat_grid_search(x, y_target2, param_grid)

            sheet_name1 = f"{filename}_Target1"
            sheet_name2 = f"{filename}_Target2"
            target1_results.to_excel(writer, sheet_name=sheet_name1[:31], index=False)
            target2_results.to_excel(writer, sheet_name=sheet_name2[:31], index=False)

print("所有結果已成功保存至 '格點搜尋總結果.xlsx'。")