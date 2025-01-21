import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import itertools
import time

# 加載數據
columns = ['IV_C0', 'Std_C0', 'IV_P0', 'Std_P0',
           'IV_C1', 'Std_C1', 'IV_P1', 'Std_P1',
           'IV_C2', 'Std_C2', 'IV_P2', 'Std_P2',
           'IV_C3', 'Std_C3', 'IV_P3', 'Std_P3', 'Target']

D = pd.read_excel("/Users/xinc./Documents/GitHub/desktop-tutorial/ndhu/HW/FIN_52750 ImplyVolatility 練習.xlsx", usecols=columns)

x_train, x_test, y_train, y_test = train_test_split(D[columns[0:16]], 
                                                    D['Target'], 
                                                    test_size=0.2)
category = 2  # 標的漲跌有幾種Label答案Y
dim = 16      # 以幾種Feature特徵X，來預測標的漲跌

# One-hot編碼標籤
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=category)
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=category)

# 參數網格定義
param_grid = {
    'units': [20, 40, 60],
    'epochs': [1000, 2000, 3000],
    'batch_size': [50, 100, 150, 200]
}

# 最佳參數的存儲
best_acc1 = 0
best_params = {}
best_temp11 = 0
best_guess1_total = 0
best_total_acc = 0
best_guess1_trial = 0  # 用來記錄最佳平均acc1對應的某一次試驗的猜1總次數

# 格點搜尋，總共會執行 20 次交叉驗證
num_trials = 30

# 計算程序執行時間
start_time = time.time()

# 創建模型架構在外部
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(dim,)))  # 使用Input來指定輸入形狀
model.add(tf.keras.layers.Dense(units=40, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=40, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=category, activation=tf.nn.softmax))

# 編譯模型
model.compile(optimizer='adam', 
              loss=tf.keras.losses.categorical_crossentropy, 
              metrics=['accuracy'])

# 保存初始權重
initial_weights = model.get_weights()

# 執行格點搜尋
for units, epochs, batch_size in itertools.product(param_grid['units'], param_grid['epochs'], param_grid['batch_size']):
    acc1_total = 0
    temp11_total = 0
    guess1_total = 0
    temp10_total = 0
    total_acc_total = 0  # 用於儲存總體準確率的總和
    
    for trial in range(num_trials):
        # 重置模型權重為初始值
        model.set_weights(initial_weights)
        
        # 訓練模型
        model.fit(x_train, y_train2, epochs=epochs, batch_size=batch_size, verbose=0)

        # 測試模型，並計算總體準確率
        score = model.evaluate(x_test, y_test2, verbose=0)
        total_acc = score[1]  # 總體準確率
        total_acc_total += total_acc  # 累積總體準確率
        
        # 預測結果
        predict2 = model.predict(x_test)
        predict2 = np.argmax(predict2, axis=1)
        test2 = np.argmax(y_test2, axis=1)

        # 計算猜1的總次數及正確次數
        temp11 = np.sum((predict2 == 1) & (test2 == 1))  # 猜1且正確的次數
        temp10 = np.sum((predict2 == 1) & (test2 == 0))  # 猜1但錯誤的次數
        guess1_total = temp11 + temp10  # 總的猜1次數
        
        # 計算猜1的平均正確率 (acc1)
        acc1 = temp11 / guess1_total if guess1_total > 0 else 0
        acc1_total += acc1
        temp11_total += temp11
        temp10_total += temp10

        print(f"試驗 {trial + 1}: 猜1的次數: {guess1_total}, 猜1正確次數: {temp11}, acc1: {acc1:.6f}, 總體準確率: {total_acc:.6f}")

    avg_acc1 = acc1_total / num_trials
    avg_total_acc = total_acc_total / num_trials  # 計算平均總體準確率
    avg_temp11_total = temp11_total / num_trials  # 計算平均正確次數
    
    print(f"格點搜尋: units={units}, epochs={epochs}, batch_size={batch_size}, 平均acc1={avg_acc1:.6f}, 平均總體準確率={avg_total_acc:.6f}")
    
    # 更新最佳結果
    if avg_acc1 > best_acc1:
        best_acc1 = avg_acc1
        best_total_acc = avg_total_acc
        best_params = {'units': units, 'epochs': epochs, 'batch_size': batch_size}
        best_temp11 = avg_temp11_total
        best_guess1_total = guess1_total  # 記錄當前猜1總次數 (某一次試驗的結果)
        best_guess1_trial = guess1_total  # 記錄當次 trial 的猜1次數

# 顯示最佳結果
print(f"\n最佳參數組合: units={best_params['units']}, epochs={best_params['epochs']}, batch_size={best_params['batch_size']}")
print(f"最佳平均acc1={best_acc1:.6f}, 對應的平均總體準確率={best_total_acc:.6f}")
print(f"最佳平均acc1時猜1的次數={best_guess1_trial}, 平均正確次數={best_temp11:.2f}")

# 計算並顯示總共花費的時間
end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"總共花費時間: {int(hours)}時 {int(minutes)}分 {int(seconds)}秒")