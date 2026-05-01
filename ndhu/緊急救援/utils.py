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
) -> pd.DataFrame:
    """通用差異性檢定表

    Parameters
    ----------
    test_vars : 要檢定的變數清單
    groups : {組別名稱: 布林遮罩}，例如 {'有議合': df['engagement_t'] == 1}
    diffs : 要做差異檢定的配對，例如 [('有議合', '無議合')]
            若未指定，不產生差異欄
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
            row[(name, 'mean')] = s.mean()
            row[(name, 'median')] = s.median()

        if diffs:
            for name_a, name_b in diffs:
                a, b = data[name_a], data[name_b]
                _, p_t = ttest_ind(a, b, equal_var=False)
                _, p_r = ranksums(a, b)
                label = f'{name_a}-{name_b}'
                row[(label, 'mean')] = f'{a.mean() - b.mean():.4f}{_stars(p_t)}'
                row[(label, 'median')] = f'{a.median() - b.median():.4f}{_stars(p_r)}'

        rows.append(row)

    result = pd.DataFrame(rows, index=test_vars)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result