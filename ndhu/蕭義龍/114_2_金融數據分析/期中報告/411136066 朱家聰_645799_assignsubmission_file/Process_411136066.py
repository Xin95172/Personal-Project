import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from google.colab import drive
import os

# 1. 掛載雲端硬碟
drive.mount('/content/drive')

# 2. 設定資料夾路徑
base_path = '/content/drive/My Drive/金融資料分析/'

files = {
    'Markowitz 3 Stocks': os.path.join(base_path, '3家公司.csv'),
    'Black Model 4 Stocks': os.path.join(base_path, '4家公司.csv')
}

# 3. 清理資料
def clean_price_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # 移除日期欄
    for col in df.columns:
        if 'date' in col.lower() or '日期' in col:
            df = df.drop(columns=[col])

    # 轉成數值，處理逗號、空白、字串問題
    df = df.apply(lambda x: pd.to_numeric(
        x.astype(str).str.replace(',', '').str.strip(),
        errors='coerce'
    ))

    df = df.dropna()
    return df

# 4. 計算效率前緣
def solve_frontier(path):
    if not os.path.exists(path):
        print(f'❌ 找不到檔案：{path}')
        return None, None, None, None

    df = clean_price_data(path)

    returns = df.pct_change(fill_method=None).dropna()

    mean_rets = returns.mean() * 252
    cov_mat = returns.cov() * 252
    n = len(df.columns)

    def get_return(w):
        return np.sum(mean_rets * w)

    def get_vol(w):
        return np.sqrt(w.T @ cov_mat @ w)

    def neg_sharpe(w):
        return -get_return(w) / get_vol(w)

    cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # 允許放空，效率前緣會比較完整
    bounds = None

    # 最小變異投資組合
    mvp = minimize(
        get_vol,
        [1/n] * n,
        method='SLSQP',
        constraints=cons_sum,
        bounds=bounds
    )

    # 最大 Sharpe 投資組合
    msr = minimize(
        neg_sharpe,
        [1/n] * n,
        method='SLSQP',
        constraints=cons_sum,
        bounds=bounds
    )

    # 掃描報酬率範圍
    r_min, r_max = mean_rets.min(), mean_rets.max()
    padding = (r_max - r_min) * 0.8
    target_rets = np.linspace(r_min - padding, r_max + padding, 200)

    vols = []

    for tr in target_rets:
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, tr=tr: np.sum(mean_rets * w) - tr}
        )

        res = minimize(
            get_vol,
            [1/n] * n,
            method='SLSQP',
            constraints=cons,
            bounds=bounds
        )

        vols.append(res.fun if res.success else np.nan)

    result_df = pd.DataFrame({
        'Expected Return': [get_return(mvp.x), get_return(msr.x)],
        'Risk': [get_vol(mvp.x), get_vol(msr.x)],
        'Sharpe Ratio': [
            get_return(mvp.x) / get_vol(mvp.x),
            get_return(msr.x) / get_vol(msr.x)
        ]
    }, index=['Minimum Variance Portfolio', 'Maximum Sharpe Portfolio'])

    weights_df = pd.DataFrame(
        [mvp.x, msr.x],
        columns=df.columns,
        index=['MVP Weight', 'MSR Weight']
    )

    return np.array(vols), np.array(target_rets), result_df, weights_df

# 5. 畫效率前緣圖
plt.figure(figsize=(12, 7))

summary = {}
weights_summary = {}

for name, path in files.items():
    vols, rets, result_df, weights_df = solve_frontier(path)

    if vols is not None:
        summary[name] = result_df
        weights_summary[name] = weights_df

        color = 'red' if '3' in name else 'blue'
        linestyle = '--' if '3' in name else '-'

        plt.plot(
            vols,
            rets,
            label=name,
            color=color,
            linestyle=linestyle,
            linewidth=2
        )

        # 標示 MVP
        mvp_risk = result_df.loc['Minimum Variance Portfolio', 'Risk']
        mvp_ret = result_df.loc['Minimum Variance Portfolio', 'Expected Return']

        plt.scatter(
            mvp_risk,
            mvp_ret,
            color=color,
            s=100,
            edgecolors='black',
            zorder=5
        )

        plt.annotate(
            f'MVP\nReturn={mvp_ret:.2%}\nRisk={mvp_risk:.2%}',
            (mvp_risk, mvp_ret),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9
        )

plt.title('Efficient Frontier Comparison: 3 Stocks vs 4 Stocks', fontsize=16)
plt.xlabel('Annualized Volatility / Risk', fontsize=12)
plt.ylabel('Expected Return', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# 只限制顯示範圍，不切掉資料
plt.xlim(0.20, 0.25)
plt.ylim(0.30, 0.60)

plt.show()

# 6. 輸出績效表與權重表
for name in summary:
    print(f'\n===== {name} 投資組合績效 =====')
    display(summary[name].style.format({
        'Expected Return': '{:.2%}',
        'Risk': '{:.2%}',
        'Sharpe Ratio': '{:.2f}'
    }))

    print(f'\n===== {name} 權重配置 =====')
    display(weights_summary[name].style.format('{:.2%}'))

# 7. 統整比較表
comparison_df = pd.concat(summary, axis=0)

print('\n===== 三家公司與四家公司投資組合比較總表 =====')
display(comparison_df.style.format({
    'Expected Return': '{:.2%}',
    'Risk': '{:.2%}',
    'Sharpe Ratio': '{:.2f}'
}))

# 8. 判斷第四家公司是否改善投資組合
three_msr = summary['Markowitz 3 Stocks'].loc['Maximum Sharpe Portfolio']
four_msr = summary['Black Model 4 Stocks'].loc['Maximum Sharpe Portfolio']

print('\n===== 第四家公司加入後的判斷 =====')

if four_msr['Sharpe Ratio'] > three_msr['Sharpe Ratio']:
    print('✅ 加入第四家公司後，Maximum Sharpe Ratio 提高，代表投資組合效率改善。')
    print('✅ 因此第四家公司可以視為較佳的新增投資標的。')
else:
    print('⚠️ 加入第四家公司後，Maximum Sharpe Ratio 沒有提高。')
    print('⚠️ 代表第四家公司對投資組合效率改善有限。')