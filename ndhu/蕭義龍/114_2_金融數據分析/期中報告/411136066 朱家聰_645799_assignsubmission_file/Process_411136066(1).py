import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

# =========================
# 1. 讀取資料
# =========================
filename = '4家公司.csv'

try:
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
    except:
        df = pd.read_csv(filename, encoding='big5')

    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        '2330收盤價': '2330',
        '2603收盤價': '2603',
        '2881 收盤價': '2881',
        '2881收盤價': '2881',
        '2359 收盤價': '2359',
        '2359收盤價': '2359'
    })

    assets_all = ['2330', '2603', '2881', '2359']

    for col in assets_all:
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    price_data = df[assets_all].dropna()

    returns = price_data.pct_change().dropna()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

except FileNotFoundError:
    print(f"錯誤：找不到檔案 {filename}，請確認是否已上傳到 Colab。")
    raise


# =========================
# 2. 投資組合函數
# =========================
rf = 0.01

def portfolio_performance(weights, mean_returns, cov_matrix):
    p_ret = np.sum(mean_returns * weights)
    p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_std, p_ret

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
    p_std, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    if p_std == 0:
        return 999
    return -(p_ret - rf) / p_std

def portfolio_volatility(weights, mean_returns, cov_matrix):
    p_std, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return p_std

def get_tangency_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    init_guess = np.ones(num_assets) / num_assets

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))

    result = sco.minimize(
        neg_sharpe_ratio,
        init_guess,
        args=(mean_returns, cov_matrix, rf),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result

def get_min_variance_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    init_guess = np.ones(num_assets) / num_assets

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))

    result = sco.minimize(
        portfolio_volatility,
        init_guess,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result


# =========================
# 3. 效率前緣函數
# =========================
def efficient_frontier(mean_returns, cov_matrix, points=100):
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), points)
    risks = []

    num_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))

    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target}
        )

        result = sco.minimize(
            portfolio_volatility,
            np.ones(num_assets) / num_assets,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            risks.append(result.fun)
        else:
            risks.append(np.nan)

    return np.array(risks), target_returns


# =========================
# 4. 三家公司與四家公司分析
# =========================
assets_3 = ['2330', '2603', '2881']
assets_4 = ['2330', '2603', '2881', '2359']

mean_3 = mean_returns[assets_3]
cov_3 = cov_matrix.loc[assets_3, assets_3]

mean_4 = mean_returns[assets_4]
cov_4 = cov_matrix.loc[assets_4, assets_4]

# 切點投資組合
opt_tan_3 = get_tangency_portfolio(mean_3, cov_3)
opt_tan_4 = get_tangency_portfolio(mean_4, cov_4)

std_tan_3, ret_tan_3 = portfolio_performance(opt_tan_3.x, mean_3, cov_3)
std_tan_4, ret_tan_4 = portfolio_performance(opt_tan_4.x, mean_4, cov_4)

sharpe_3 = (ret_tan_3 - rf) / std_tan_3
sharpe_4 = (ret_tan_4 - rf) / std_tan_4

# 最小變異投資組合
opt_min_3 = get_min_variance_portfolio(mean_3, cov_3)
opt_min_4 = get_min_variance_portfolio(mean_4, cov_4)

std_min_3, ret_min_3 = portfolio_performance(opt_min_3.x, mean_3, cov_3)
std_min_4, ret_min_4 = portfolio_performance(opt_min_4.x, mean_4, cov_4)

sharpe_min_3 = (ret_min_3 - rf) / std_min_3
sharpe_min_4 = (ret_min_4 - rf) / std_min_4

# 效率前緣
risk_3, target_3 = efficient_frontier(mean_3, cov_3)
risk_4, target_4 = efficient_frontier(mean_4, cov_4)


# =========================
# 5. 印出比較表
# =========================
summary = pd.DataFrame({
    '投資組合': [
        '三家公司-切點投資組合',
        '四家公司-切點投資組合',
        '三家公司-最小變異投資組合',
        '四家公司-最小變異投資組合'
    ],
    '年化報酬率': [ret_tan_3, ret_tan_4, ret_min_3, ret_min_4],
    '年化風險': [std_tan_3, std_tan_4, std_min_3, std_min_4],
    'Sharpe Ratio': [sharpe_3, sharpe_4, sharpe_min_3, sharpe_min_4]
})

print("========== 投資組合績效比較 ==========")
print(summary.round(4))

weights_3 = pd.DataFrame({
    '股票': assets_3,
    '三家公司切點權重': opt_tan_3.x
})

weights_4 = pd.DataFrame({
    '股票': assets_4,
    '四家公司切點權重': opt_tan_4.x
})

print("\n========== 三家公司切點投資組合權重 ==========")
print(weights_3.round(4))

print("\n========== 四家公司切點投資組合權重 ==========")
print(weights_4.round(4))


# =========================
# 6. 效率前緣圖
# =========================
plt.figure(figsize=(10, 6))

plt.plot(risk_3, target_3, label='3 Assets Efficient Frontier', linewidth=2)
plt.plot(risk_4, target_4, label='4 Assets Efficient Frontier (+2359)', linewidth=2)

plt.scatter(std_tan_3, ret_tan_3, marker='*', s=250, label='3 Assets Tangency')
plt.scatter(std_tan_4, ret_tan_4, marker='*', s=250, label='4 Assets Tangency')

plt.scatter(std_min_3, ret_min_3, marker='o', s=120, label='3 Assets Min Variance')
plt.scatter(std_min_4, ret_min_4, marker='o', s=120, label='4 Assets Min Variance')

plt.xlabel('Annualized Risk / Volatility')
plt.ylabel('Annualized Expected Return')
plt.title('Efficient Frontier Comparison: 3 Assets vs 4 Assets')
plt.legend()
plt.grid(True)

# 你之前要求圖不要太誇張，可以保留這個範圍
plt.xlim(0, 0.25)
plt.ylim(0.3, 0.6)

plt.show()


# =========================
# 7. 績效長條圖
# =========================
categories = ['Expected Return', 'Risk', 'Sharpe Ratio']
value_3 = [ret_tan_3, std_tan_3, sharpe_3]
value_4 = [ret_tan_4, std_tan_4, sharpe_4]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(x - width/2, value_3, width, label='3 Assets')
plt.bar(x + width/2, value_4, width, label='4 Assets (+2359)')

plt.xticks(x, categories)
plt.ylabel('Value')
plt.title('Tangency Portfolio Performance Comparison')
plt.legend()
plt.grid(axis='y')

for i, v in enumerate(value_3):
    plt.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')

for i, v in enumerate(value_4):
    plt.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')

plt.show()


# =========================
# 8. 權重圓餅圖
# =========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.pie(opt_tan_3.x, labels=assets_3, autopct='%1.1f%%', startangle=90)
plt.title('3 Assets Tangency Portfolio Weights')

plt.subplot(1, 2, 2)
plt.pie(opt_tan_4.x, labels=assets_4, autopct='%1.1f%%', startangle=90)
plt.title('4 Assets Tangency Portfolio Weights')

plt.tight_layout()
plt.show()


# =========================
# 9. 自動輸出分析文字
# =========================
print("\n========== 投資分析結論 ==========")

if sharpe_4 > sharpe_3:
    print("加入第四家公司 2359 所羅門後，切點投資組合的 Sharpe Ratio 提高，代表在承擔相同風險下，投資組合可以獲得較好的風險調整後報酬。")
else:
    print("加入第四家公司 2359 所羅門後，切點投資組合的 Sharpe Ratio 沒有提高，代表風險調整後績效未明顯改善。")

if std_min_4 < std_min_3:
    print("四家公司最小變異投資組合的年化風險低於三家公司，表示 2359 所羅門具有分散風險的效果。")
else:
    print("四家公司最小變異投資組合的年化風險沒有低於三家公司，表示 2359 所羅門的分散風險效果有限。")

if ret_tan_4 > ret_tan_3 and std_tan_4 < std_tan_3:
    print("整體而言，2359 所羅門加入後達到報酬提高且風險下降，是非常理想的第四家標的。")
elif ret_tan_4 > ret_tan_3:
    print("整體而言，2359 所羅門加入後提高了預期報酬，但仍需觀察風險是否同步上升。")
elif std_tan_4 < std_tan_3:
    print("整體而言，2359 所羅門加入後主要效果是降低風險，較偏向分散投資功能。")
else:
    print("整體而言，2359 所羅門加入後沒有明顯改善報酬或風險，需重新評估是否為最佳第四家標的。")