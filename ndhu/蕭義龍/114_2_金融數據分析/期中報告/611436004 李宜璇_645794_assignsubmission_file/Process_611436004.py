import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

# ===== 資料抓取 =====
assets_3 = ['AAPL', 'JPM', 'XOM']
assets_4 = ['AAPL', 'JPM', 'XOM', 'TLT']

raw = yf.download(assets_4, start='2020-01-01', end='2024-12-31', interval='1mo', auto_adjust=True)['Close']
returns = raw.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

print("資料抓取完成！")
print(f"月平均報酬率：\n{mean_returns.round(4)}")

# ===== 效率前緣函數 =====
def portfolio_stats(weights, mean_ret, cov_mat):
    port_return = np.dot(weights, mean_ret)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    return port_return, port_std

def min_variance(target_return, mean_ret, cov_mat):
    n = len(mean_ret)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, mean_ret) - target_return}
    ]
    bounds = [(0, 1)] * n
    result = minimize(
        lambda w: np.dot(w.T, np.dot(cov_mat, w)),
        x0=np.ones(n)/n, bounds=bounds, constraints=constraints
    )
    return result

def get_frontier(assets):
    mean_ret = mean_returns[assets]
    cov_mat = cov_matrix.loc[assets, assets]
    min_r, max_r = mean_ret.min(), mean_ret.max()
    target_returns = np.linspace(min_r, max_r, 300)
    frontier_std, frontier_ret = [], []
    for tr in target_returns:
        res = min_variance(tr, mean_ret, cov_mat)
        if res.success:
            _, std = portfolio_stats(res.x, mean_ret, cov_mat)
            frontier_std.append(std)
            frontier_ret.append(tr)
    return np.array(frontier_std), np.array(frontier_ret)

std_3, ret_3 = get_frontier(assets_3)
std_4, ret_4 = get_frontier(assets_4)

# ===== 效率前緣圖 =====
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(std_4, ret_4, 'r-', label='TLT 債券ETF（四家）', linewidth=2.5)
ax.plot(std_3, ret_3, 'b--', label='三家基準', linewidth=2, alpha=0.8)
ax.set_xlabel('月標準差（風險）', fontsize=12)
ax.set_ylabel('月平均報酬率', fontsize=12)
ax.set_title('三家 vs 四家有效前緣比較', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('efficient_frontier_組員A.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== MVP計算 =====
def get_mvp(mean_ret, cov_mat):
    n = len(mean_ret)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    result = minimize(
        lambda w: np.dot(w.T, np.dot(cov_mat, w)),
        x0=np.ones(n)/n, bounds=bounds, constraints=constraints
    )
    w = result.x
    ret = np.dot(w, mean_ret)
    std = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    return w, ret, std

w3, ret3, std3 = get_mvp(mean_returns[assets_3], cov_matrix.loc[assets_3, assets_3])
w4, ret4, std4 = get_mvp(mean_returns[assets_4], cov_matrix.loc[assets_4, assets_4])

std_drop = (std3 - std4) / std3 * 100
ret_chg = (ret4 - ret3) / abs(ret3) * 100

print("\n=== 三家基準 MVP ===")
print(f"月報酬率：{ret3:.4f}，月標準差：{std3:.4f}")
print(f"\n=== 四家（TLT 債券ETF）MVP ===")
print(f"月報酬率：{ret4:.4f}（{ret_chg:+.2f}%）")
print(f"月標準差：{std4:.4f}（風險下降：{std_drop:.2f}%）")
print(f"權重：{dict(zip(assets_4, w4.round(4)))}")