import pandas as pd
from scipy.stats.mstats import winsorize as _winsorize


# TESG 等級 → 數字
GRADE_MAP = {
    'A+': 7, 'A': 6, 'B+': 5, 'B': 4, 'B-': 3, 'C': 2, 'C-': 1
}

def detect_grade_cols(df: pd.DataFrame) -> list[str]:
    """自動偵測值為等級（A+, A, B+, ...）的欄位"""
    grade_keys = set(GRADE_MAP.keys())
    result = []
    for col in df.columns:
        if df[col].dtype == object:
            unique_vals = set(df[col].dropna().unique())
            # 該欄位的所有非空值都在 GRADE_MAP 裡 → 判定為 grade 欄位
            if unique_vals and unique_vals.issubset(grade_keys):
                result.append(col)
    return result

def detect_rank_cols(df: pd.DataFrame) -> list[str]:
    """自動偵測值為 '5/111' 格式的排名欄位"""
    result = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            # 檢查是否符合 "數字/數字" 的格式
            if len(sample) > 0 and sample.str.match(r'^\d+/\d+$').all():
                result.append(col)
    return result

def convert_grade_to_numeric(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """將 TESG 等級欄位轉為數值（A+=7, ..., C-=1）
    若未指定 cols，自動偵測
    """
    df = df.copy()
    if cols is None:
        cols = detect_grade_cols(df)
    for col in cols:
        df[col] = df[col].map(GRADE_MAP)
    return df

def convert_rank_to_pct(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """將 '5/111' 格式的排名欄位轉為百分比（0~1）
    若未指定 cols，自動偵測
    """
    df = df.copy()
    if cols is None:
        cols = detect_rank_cols(df)
    for col in cols:
        split = df[col].astype(str).str.split('/', expand=True)
        numerator = pd.to_numeric(split[0], errors='coerce')
        denominator = pd.to_numeric(split[1], errors='coerce')
        df[col] = numerator / denominator
    return df

def convert_data(df: pd.DataFrame) -> pd.DataFrame:
    """一次完成所有資料轉換（grade → 數字、rank → 百分比）"""
    df = df.copy()
    df = convert_grade_to_numeric(df)
    df = convert_rank_to_pct(df)
    return df

def winsorize_cols(df: pd.DataFrame, cols: list[str], limits: tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
    """對指定欄位做 winsorize，自動跳過 NaN"""
    df = df.copy()
    for col in cols:
        mask = df[col].notna()
        df.loc[mask, col] = _winsorize(df.loc[mask, col], limits=limits)
    return df


# ============================
# 回歸
# ============================

def run_regression(
    df: pd.DataFrame,
    y: str,
    x_vars: list[str],
    fe_vars: list[str] | None = None,
    cluster_var: str | None = None,
    model_type: str = 'ols',
):
    """執行回歸分析

    Parameters
    ----------
    y : 應變數欄位名
    x_vars : 解釋變數 + 控制變數欄位名
    fe_vars : 固定效果欄位名（會用 C() 產生 dummy）
    cluster_var : cluster standard error 的分群欄位
    model_type : 'ols' 或 'probit'
    """
    import statsmodels.formula.api as smf

    # 先 dropna，避免 cluster groups 長度不匹配
    used_cols = [y] + x_vars
    if fe_vars:
        used_cols += fe_vars
    if cluster_var:
        used_cols += [cluster_var]
    df_clean = df[used_cols].dropna()

    # 建構 formula
    rhs = ' + '.join(x_vars)
    if fe_vars:
        rhs += ' + ' + ' + '.join([f'C({v})' for v in fe_vars])
    formula = f'{y} ~ {rhs}'

    # 選模型
    if model_type == 'ols':
        model = smf.ols(formula, data=df_clean)
    elif model_type == 'probit':
        model = smf.probit(formula, data=df_clean)
    else:
        raise ValueError(f"不支援的 model_type: {model_type}，請用 'ols' 或 'probit'")

    # fit（支援 clustered SE）
    fit_kwds = {'disp': 0}
    if cluster_var:
        fit_kwds['cov_type'] = 'cluster'
        fit_kwds['cov_kwds'] = {'groups': df_clean[cluster_var]}

    result = model.fit(**fit_kwds)

    return result


# ============================
# 差異性檢定
# ============================

def _stars(p: float) -> str:
    """根據 p-value 回傳顯著星號"""
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    return ''


def difference_test_table(
    df: pd.DataFrame,
    test_vars: list[str],
    groups: dict[str, pd.Series],
    diffs: list[tuple[str, str]] | None = None,
    decimals: int = 3,
) -> pd.DataFrame:
    """通用差異性檢定表

    Parameters
    ----------
    test_vars : 要檢定的變數清單
    groups : {組別名稱: 布林遮罩}，例如 {'有議合': df['engagement_t'] == 1}
    diffs : 要做差異檢定的配對，例如 [('有議合', '無議合')]
            若未指定，不產生差異欄
    decimals : 小數點後顯示位數
    """
    from scipy.stats import ttest_ind, ranksums

    # 依 mask 篩選子集
    subsets = {name: df[mask] for name, mask in groups.items()}

    rows = []
    for var in test_vars:
        row = {}
        data = {}
        for name, sub in subsets.items():
            s = sub[var].dropna()
            data[name] = s
            row[(name, 'mean')] = round(s.mean(), decimals)
            row[(name, 'median')] = round(s.median(), decimals)

        if diffs:
            for name_a, name_b in diffs:
                a, b = data[name_a], data[name_b]
                _, p_t = ttest_ind(a, b, equal_var=False)
                _, p_r = ranksums(a, b)
                label = f'{name_a}-{name_b}'
                
                mean_diff = a.mean() - b.mean()
                median_diff = a.median() - b.median()
                
                row[(label, 'mean')] = f'{mean_diff:.{decimals}f}{_stars(p_t)}'
                row[(label, 'mean_p')] = round(p_t, decimals)
                row[(label, 'median')] = f'{median_diff:.{decimals}f}{_stars(p_r)}'
                row[(label, 'median_p')] = round(p_r, decimals)

        rows.append(row)

    result = pd.DataFrame(rows, index=test_vars)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


def corr_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """產生帶有顯著性星號的相關係數矩陣 (Correlation Matrix)
    左下角為 Pearson 相關係數，右上角為 Spearman 等級相關係數。

    Parameters
    ----------
    df : 資料集
    cols : 要計算相關係數的變數清單
    """
    import numpy as np
    from scipy.stats import t
    
    # 確保資料沒有 NaN
    df_clean = df[cols].dropna()
    n = len(df_clean)
    
    # 1. 向量化計算相關係數 (Pandas 底層有高度優化，極快)
    r_pearson = df_clean.corr(method='pearson').values
    r_spearman = df_clean.corr(method='spearman').values
    
    # 2. 向量化計算 P-value (利用 numpy 矩陣運算計算 t 統計量)
    # 避免對角線 r=1 導致分母為 0，限制數值上限
    r_p_safe = np.clip(r_pearson, -0.999999, 0.999999)
    t_pearson = r_p_safe * np.sqrt((n - 2) / (1 - r_p_safe**2))
    p_pearson = t.sf(np.abs(t_pearson), n - 2) * 2
    
    r_s_safe = np.clip(r_spearman, -0.999999, 0.999999)
    t_spearman = r_s_safe * np.sqrt((n - 2) / (1 - r_s_safe**2))
    p_spearman = t.sf(np.abs(t_spearman), n - 2) * 2
    
    # 3. 填入 DataFrame
    corr = pd.DataFrame(index=cols, columns=cols)
    
    # 這裡的迴圈只負責「字串排版」，不需要做繁重的統計運算，瞬間就能跑完
    for i in range(len(cols)):
        for j in range(len(cols)):
            col_i, col_j = cols[i], cols[j]
            if i == j:
                corr.loc[col_i, col_j] = "1.000"
            elif i > j:
                corr.loc[col_i, col_j] = f"{r_pearson[i, j]:.3f}{_stars(p_pearson[i, j])}"
            else:
                corr.loc[col_i, col_j] = f"{r_spearman[i, j]:.3f}{_stars(p_spearman[i, j])}"
                
    return corr



# ============================
# 視覺化
# ============================

def plot_desc_bar(
    df: pd.DataFrame,
    cols: list[str],
    demean: bool = False,
    title: str = '',
    figsize: tuple[int, int] = (14, 6),
    save_path: str | None = None,
    ax=None,
):
    """畫指定變數的 mean 長條圖

    Parameters
    ----------
    df : 原始資料（會自動算 mean）
    cols : 要畫的欄位名稱
    demean : 若 True，每個變數先減去這組變數的平均值，突顯相對起伏
    title : 圖表標題
    save_path : 存檔路徑（.png），None 則不存檔
    ax : 指定 matplotlib Axes 物件 (畫子圖時使用)
    """
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    means = df[cols].mean()
    if demean:
        # 減去這組變數 mean 的平均值 (scalar)
        avg_of_means = means.mean()
        means = means - avg_of_means

    if ax is None:
        fig, current_ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        current_ax = ax
        show_plot = False

    colors = ['#3498db'] # 預設藍色
    if demean:
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in means.values]
    
    current_ax.bar(range(len(means)), means.values, color=colors)
    
    if demean:
        current_ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        current_ax.set_ylabel('Deviation from Group Average Mean')
    else:
        current_ax.set_ylabel('Mean')
    current_ax.set_title(title)
    
    current_ax.set_xticks(range(len(means)))
    current_ax.set_xticklabels(means.index, rotation=45, ha='right')

    if show_plot:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig
    else:
        return current_ax

def plot_group_bar(
    table: pd.DataFrame,
    group_names: list[str] | None = None,
    title: str = '',
    figsize: tuple[int, int] = (14, 6),
    save_path: str | None = None,
):
    """從差異檢定表畫分組長條圖（只取 mean 欄）

    Parameters
    ----------
    table : difference_test_table 回傳的 MultiIndex DataFrame
    group_names : 要畫的組別名稱（第一層 column），若不指定則取所有非差異欄
    title : 圖表標題
    figsize : 圖表大小
    save_path : 存檔路徑（.png），None 則不存檔
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 取第一層 column 中不含 '-' 的組別（排除差異欄）
    if group_names is None:
        group_names = [name for name in table.columns.get_level_values(0).unique()
                       if '-' not in name]

    # 取 mean 欄位
    plot_data = table.loc[:, [(g, 'mean') for g in group_names]]
    plot_data.columns = group_names
    plot_data = plot_data.apply(pd.to_numeric, errors='coerce')

    # 畫圖
    x = np.arange(len(plot_data.index))
    width = 0.8 / len(group_names)

    fig, ax = plt.subplots(figsize=figsize)
    for i, g in enumerate(group_names):
        ax.bar(x + i * width, plot_data[g], width, label=g)

    ax.set_xticks(x + width * (len(group_names) - 1) / 2)
    ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return fig