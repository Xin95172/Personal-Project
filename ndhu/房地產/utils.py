import math
import statistics
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime


NUMERIC_COLUMNS = [
    "成交總價_含車位_萬元",
    "建坪_含車位",
    "樓層",
    "總樓層",
    "屋齡",
    "屋齡平方",
    "有車位虛擬變數",
    "中高樓層虛擬變數",
    "到東華附小距離(google)",
    "到球崙公園距離(google)",
    "到門諾距離(google)",
]


def read_xlsx_records(path, sheet_index=0):
    """Read one worksheet in an .xlsx file without third-party dependencies."""
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with zipfile.ZipFile(path) as z:
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared_strings.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))

        sheet = list(workbook.find("a:sheets", ns))[sheet_index]
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = relmap[rel_id].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target

        root = ET.fromstring(z.read(target))
        rows = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            values = []
            expected_col = 1
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "A1")
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                col_num = 0
                for ch in col_letters:
                    col_num = col_num * 26 + ord(ch.upper()) - ord("A") + 1
                while expected_col < col_num:
                    values.append("")
                    expected_col += 1

                value_node = cell.find("a:v", ns)
                value = "" if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value != "":
                    value = shared_strings[int(value)]
                values.append(value)
                expected_col += 1
            rows.append(values)

    headers = rows[0]
    records = []
    for row in rows[1:]:
        padded = row + [""] * (len(headers) - len(row))
        records.append(dict(zip(headers, padded)))
    return records


def to_float(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(",", "").strip())


def parse_date_fraction(value):
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value)[:10])
    return dt.year + (dt.month - 1) / 12 + (dt.day - 1) / 365.25


def prepare_model_rows(records, numeric_columns=NUMERIC_COLUMNS):
    model_rows = []
    for row in records:
        clean = dict(row)
        for col in numeric_columns:
            clean[col] = to_float(clean.get(col))
        clean["成交年分數"] = parse_date_fraction(clean.get("成交日期"))
        clean["ln_price"] = math.log(clean["成交總價_含車位_萬元"])
        model_rows.append(clean)
    return model_rows


def describe_columns(rows, columns):
    summary = []
    for col in columns:
        vals = [row[col] for row in rows]
        summary.append(
            {
                "欄位": col,
                "min": min(vals),
                "max": max(vals),
                "unique": len(set(vals)),
            }
        )
    return summary


def add_centered_square(rows, col, centered_col=None, squared_col=None):
    centered_col = centered_col or f"{col}_center"
    squared_col = squared_col or f"{centered_col}平方"
    mean_value = statistics.mean(row[col] for row in rows)
    new_rows = []
    for row in rows:
        new_row = dict(row)
        centered = new_row[col] - mean_value
        new_row[centered_col] = centered
        new_row[squared_col] = centered**2
        new_rows.append(new_row)
    return new_rows, mean_value


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def matvec(a, v):
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def inverse(matrix):
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Design matrix is singular; remove collinear variables.")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        aug[col] = [x / pivot_value for x in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def normal_p_value(t_stat):
    return math.erfc(abs(t_stat) / math.sqrt(2))


def chi2_p_value(stat, df):
    try:
        from scipy.stats import chi2

        return chi2.sf(stat, df)
    except ImportError:
        if df == 2:
            return math.exp(-stat / 2)
        return None


def ols(rows, y_col, x_cols):
    y = [row[y_col] for row in rows]
    x = [[1.0] + [row[col] for col in x_cols] for row in rows]
    xt = transpose(x)
    xtx_inv = inverse(matmul(xt, x))
    beta = matvec(xtx_inv, matvec(xt, y))
    fitted = [sum(b * value for b, value in zip(beta, row)) for row in x]
    residuals = [actual - pred for actual, pred in zip(y, fitted)]
    n = len(y)
    k = len(beta)
    sse = sum(e ** 2 for e in residuals)
    sst = sum((actual - statistics.mean(y)) ** 2 for actual in y)
    sigma2 = sse / (n - k)
    std_err = [math.sqrt(sigma2 * xtx_inv[i][i]) for i in range(k)]
    t_stats = [b / se for b, se in zip(beta, std_err)]
    return {
        "names": ["截距"] + x_cols,
        "beta": beta,
        "std_err": std_err,
        "t_stats": t_stats,
        "p_values": [normal_p_value(t) for t in t_stats],
        "fitted": fitted,
        "residuals": residuals,
        "n": n,
        "r2": 1 - sse / sst,
        "adj_r2": 1 - (sse / (n - k)) / (sst / (n - 1)),
        "rmse": math.sqrt(sse / n),
    }


def variance_inflation_factors(rows, x_cols):
    table = []
    for col in x_cols:
        other_cols = [item for item in x_cols if item != col]
        if not other_cols:
            vif = 1.0
            r2 = 0.0
        else:
            aux = ols(rows, col, other_cols)
            r2 = aux["r2"]
            vif = math.inf if r2 >= 1 else 1 / (1 - r2)
        table.append({"變數": col, "R2": r2, "VIF": vif})
    return table


def print_vif_table(vif_table):
    print(f"{'變數':<24}{'R^2':>12}{'VIF':>12}{'判讀':>16}")
    print("-" * 64)
    for item in vif_table:
        vif = item["VIF"]
        if vif >= 10:
            note = "嚴重"
        elif vif >= 5:
            note = "偏高"
        else:
            note = "可接受"
        vif_text = "inf" if math.isinf(vif) else f"{vif:.3f}"
        print(f"{item['變數']:<24}{item['R2']:>12.4f}{vif_text:>12}{note:>16}")


def residual_summary(result):
    residuals = result["residuals"]
    n = len(residuals)
    mean = statistics.mean(residuals)
    std = statistics.stdev(residuals) if n > 1 else 0.0
    centered = [e - mean for e in residuals]
    m2 = sum(e**2 for e in centered) / n
    m3 = sum(e**3 for e in centered) / n
    m4 = sum(e**4 for e in centered) / n
    skew = m3 / (m2**1.5) if m2 > 0 else 0.0
    kurtosis = m4 / (m2**2) if m2 > 0 else 0.0
    jarque_bera = n / 6 * (skew**2 + ((kurtosis - 3) ** 2) / 4)
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": min(residuals),
        "q1": statistics.quantiles(residuals, n=4)[0],
        "median": statistics.median(residuals),
        "q3": statistics.quantiles(residuals, n=4)[2],
        "max": max(residuals),
        "skew": skew,
        "kurtosis": kurtosis,
        "jarque_bera": jarque_bera,
        "jarque_bera_p": chi2_p_value(jarque_bera, 2),
    }


def breusch_pagan_test(rows, result, x_cols):
    aux_rows = []
    for row, residual in zip(rows, result["residuals"]):
        aux_row = dict(row)
        aux_row["_resid_sq"] = residual**2
        aux_rows.append(aux_row)
    aux = ols(aux_rows, "_resid_sq", x_cols)
    lm_stat = result["n"] * aux["r2"]
    df = len(x_cols)
    return {
        "lm_stat": lm_stat,
        "df": df,
        "p_value": chi2_p_value(lm_stat, df),
        "aux_r2": aux["r2"],
    }


def influence_diagnostics(rows, result, x_cols, id_cols=None):
    id_cols = id_cols or []
    x = [[1.0] + [row[col] for col in x_cols] for row in rows]
    xtx_inv = inverse(matmul(transpose(x), x))
    n = result["n"]
    p = len(result["beta"])
    sse = sum(e**2 for e in result["residuals"])
    mse = sse / (n - p)
    diagnostics = []
    for row, x_i, residual in zip(rows, x, result["residuals"]):
        leverage = matmul([x_i], matmul(xtx_inv, [[v] for v in x_i]))[0][0]
        denom = math.sqrt(mse * max(1 - leverage, 1e-12))
        student_resid = residual / denom if denom else math.inf
        cooks_d = (residual**2 / (p * mse)) * leverage / max((1 - leverage) ** 2, 1e-12)
        item = {col: row.get(col) for col in id_cols}
        item.update(
            {
                "residual": residual,
                "student_resid": student_resid,
                "leverage": leverage,
                "cooks_d": cooks_d,
            }
        )
        diagnostics.append(item)
    return sorted(diagnostics, key=lambda item: item["cooks_d"], reverse=True)


def print_residual_diagnostics(rows, result, x_cols, id_cols=None, top_n=5):
    summary = residual_summary(result)
    print("Residual summary")
    for key in ["count", "mean", "std", "min", "q1", "median", "q3", "max", "skew", "kurtosis"]:
        value = summary[key]
        print(f"{key:<12}: {value:.4f}" if isinstance(value, float) else f"{key:<12}: {value}")

    jb_p = summary["jarque_bera_p"]
    print(
        f"Jarque-Bera: stat={summary['jarque_bera']:.4f}, "
        f"p-value={'NA' if jb_p is None else f'{jb_p:.4f}'}"
    )

    bp = breusch_pagan_test(rows, result, x_cols)
    print(
        f"Breusch-Pagan: LM={bp['lm_stat']:.4f}, df={bp['df']}, "
        f"p-value={'NA' if bp['p_value'] is None else f'{bp['p_value']:.4f}'}"
    )

    print(f"\nTop {top_n} Cook's distance")
    diagnostics = influence_diagnostics(rows, result, x_cols, id_cols=id_cols)[:top_n]
    columns = list(id_cols or []) + ["residual", "student_resid", "leverage", "cooks_d"]
    print(" | ".join(columns))
    print("-" * 96)
    for item in diagnostics:
        values = []
        for col in columns:
            value = item.get(col)
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        print(" | ".join(values))


def plot_residual_diagnostics(result):
    import matplotlib.pyplot as plt

    residuals = result["residuals"]
    fitted = result["fitted"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(fitted, residuals)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals vs Fitted")
    axes[0].set_xlabel("Fitted")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=10, edgecolor="black")
    axes[1].set_title("Residual Histogram")
    axes[1].set_xlabel("Residual")

    try:
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title("QQ Plot")
    except ImportError:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Install scipy for QQ plot", ha="center", va="center")

    plt.tight_layout()
    return fig


def print_summary(result):
    print(
        f"n = {result['n']}, R^2 = {result['r2']:.4f}, "
        f"Adjusted R^2 = {result['adj_r2']:.4f}, RMSE(log) = {result['rmse']:.4f}"
    )
    print("-" * 112)
    print(f"{'變數':<24}{'係數':>14}{'標準誤':>14}{'t值':>12}{'p值':>12}{'約略%影響':>14}")
    print("-" * 112)
    for name, beta, se, t, p in zip(
        result["names"],
        result["beta"],
        result["std_err"],
        result["t_stats"],
        result["p_values"],
    ):
        pct = "" if name == "截距" else f"{(math.exp(beta) - 1) * 100:.2f}%"
        print(f"{name:<24}{beta:>14.6f}{se:>14.6f}{t:>12.3f}{p:>12.4f}{pct:>14}")


def predict_price(result, features, target):
    x = [1.0] + [target[col] for col in features]
    ln_pred = sum(b * value for b, value in zip(result["beta"], x))
    return ln_pred, math.exp(ln_pred)
