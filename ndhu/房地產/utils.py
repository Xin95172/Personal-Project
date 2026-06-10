import math
import statistics
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


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

                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//a:t", ns))
                else:
                    value_node = cell.find("a:v", ns)
                    value = "" if value_node is None else value_node.text
                if cell_type == "s" and value != "":
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
            if col in clean:
                clean[col] = to_float(clean.get(col))
        clean["成交年分數"] = parse_date_fraction(clean.get("成交日期"))
        clean["ln_price"] = math.log(clean["成交總價_含車位_萬元"])
        model_rows.append(clean)
    return model_rows


def describe_columns(rows, columns):
    summary = []
    for col in columns:
        vals = [row[col] for row in rows if row.get(col) is not None]
        summary.append(
            {
                "欄位": col,
                "min": min(vals),
                "max": max(vals),
                "unique": len(set(vals)),
            }
        )
    return summary


def filter_rows(rows, exclude_case_ids=None, id_col="Case_ID"):
    exclude_case_ids = {str(case_id) for case_id in (exclude_case_ids or set())}
    if not exclude_case_ids:
        return list(rows), []
    included_rows = [row for row in rows if str(row.get(id_col)) not in exclude_case_ids]
    excluded_rows = [row for row in rows if str(row.get(id_col)) in exclude_case_ids]
    return included_rows, excluded_rows


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


def add_relative_time(rows, time_col="成交年分數", relative_col="成交年分數_相對起點", base_value=None):
    base_value = min(row[time_col] for row in rows) if base_value is None else base_value
    new_rows = []
    for row in rows:
        new_row = dict(row)
        new_row[relative_col] = new_row[time_col] - base_value
        new_rows.append(new_row)
    return new_rows, base_value


def _design_matrix(rows, x_cols):
    df = pd.DataFrame(rows)
    x = sm.add_constant(df[x_cols], has_constant="add")
    return x


def ols(rows, y_col, x_cols, exclude_case_ids=None, id_col="Case_ID"):
    included_rows, excluded_rows = filter_rows(rows, exclude_case_ids, id_col=id_col)
    df = pd.DataFrame(included_rows)
    y = df[y_col]
    x = _design_matrix(included_rows, x_cols)
    model = sm.OLS(y, x).fit()
    robust_model = model.get_robustcov_results(cov_type="HC3")

    return {
        "model": model,
        "robust_model": robust_model,
        "rows": included_rows,
        "excluded_rows": excluded_rows,
        "excluded_case_ids": sorted(str(row.get(id_col)) for row in excluded_rows),
        "features": list(x_cols),
        "y_col": y_col,
        "x": x,
        "y": y,
        "names": list(model.params.index),
        "beta": list(model.params),
        "std_err": list(model.bse),
        "t_stats": list(model.tvalues),
        "p_values": list(model.pvalues),
        "fitted": list(model.fittedvalues),
        "residuals": list(model.resid),
        "n": int(model.nobs),
        "all_n": int(model.nobs) + len(excluded_rows),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "rmse": math.sqrt((model.resid**2).mean()),
    }


def print_summary(result, robust=False):
    excluded_n = len(result.get("excluded_rows", []))
    if excluded_n:
        print(f"原始 n = {result.get('all_n', result['n'])}, 排除 n = {excluded_n}, 回歸 n = {result['n']}")
        print(f"排除 Case_ID: {', '.join(result.get('excluded_case_ids', []))}")
        print()
    if robust:
        print(result["robust_model"].summary())
    else:
        print(result["model"].summary())


def coefficient_report(result, robust=False):
    model = result["robust_model"] if robust else result["model"]
    names = result["model"].params.index
    table = pd.DataFrame(
        {
            "變數": names,
            "係數": model.params,
            "標準誤": model.bse,
            "t值": model.tvalues,
            "p值": model.pvalues,
        }
    )
    table["約略%影響"] = [
        "" if name == "const" else f"{(math.exp(beta) - 1) * 100:.2f}%"
        for name, beta in zip(table["變數"], table["係數"])
    ]
    return table


def variance_inflation_factors(rows, x_cols):
    x = _design_matrix(rows, x_cols)
    table = pd.DataFrame(
        {
            "變數": x.columns,
            "VIF": [variance_inflation_factor(x.values, i) for i in range(x.shape[1])],
        }
    )
    table["判讀"] = [
        "截距項不判讀" if name == "const" else ("嚴重" if vif >= 10 else ("偏高" if vif >= 5 else "可接受"))
        for name, vif in zip(table["變數"], table["VIF"])
    ]
    return table


def print_vif_table(vif_table):
    print(vif_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def residual_summary(result):
    resid = pd.Series(result["model"].resid)
    jb_stat, jb_p, skew, kurtosis = sm.stats.jarque_bera(resid)
    return pd.DataFrame(
        [
            {
                "count": resid.count(),
                "mean": resid.mean(),
                "std": resid.std(),
                "min": resid.min(),
                "q1": resid.quantile(0.25),
                "median": resid.median(),
                "q3": resid.quantile(0.75),
                "max": resid.max(),
                "skew": skew,
                "kurtosis": kurtosis,
                "jarque_bera": jb_stat,
                "jarque_bera_p": jb_p,
            }
        ]
    )


def breusch_pagan_test(result):
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(result["model"].resid, result["x"])
    return pd.DataFrame(
        [
            {
                "LM statistic": lm_stat,
                "LM p-value": lm_pvalue,
                "F statistic": f_stat,
                "F p-value": f_pvalue,
            }
        ]
    )


def influence_diagnostics(result, id_cols=None):
    id_cols = id_cols or []
    rows = result["rows"]
    influence = result["model"].get_influence()
    frame = pd.DataFrame({col: [row.get(col) for row in rows] for col in id_cols})
    frame["residual"] = result["model"].resid.values
    frame["student_resid"] = influence.resid_studentized_internal
    frame["leverage"] = influence.hat_matrix_diag
    frame["cooks_d"] = influence.cooks_distance[0]
    return frame.sort_values("cooks_d", ascending=False)


def print_residual_diagnostics(rows, result, x_cols, id_cols=None, top_n=5):
    # rows and x_cols are retained for notebook backward compatibility.
    print("Residual summary")
    print(residual_summary(result).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print("Breusch-Pagan test")
    print(breusch_pagan_test(result).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Top {top_n} Cook's distance")
    print(influence_diagnostics(result, id_cols=id_cols).head(top_n).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def robust_ols_result(rows, result, x_cols, cov_type="HC3"):
    robust = dict(result)
    robust["robust_model"] = result["model"].get_robustcov_results(cov_type=cov_type)
    return robust


def print_model_comparison(models):
    rows = []
    for label, model_rows, result, features in models:
        vif_table = variance_inflation_factors(model_rows, features)
        bp = breusch_pagan_test(result)
        rows.append(
            {
                "模型": label,
                "n": result["n"],
                "R^2": result["r2"],
                "Adj R^2": result["adj_r2"],
                "RMSE(log)": result["rmse"],
                "最高VIF": vif_table.loc[vif_table["變數"] != "const", "VIF"].max(),
                "BP p": bp.loc[0, "LM p-value"],
                "AIC": result["model"].aic,
                "BIC": result["model"].bic,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def plot_residual_diagnostics(result):
    import matplotlib.pyplot as plt
    from scipy import stats

    residuals = result["model"].resid
    fitted = result["model"].fittedvalues

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(fitted, residuals)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals vs Fitted")
    axes[0].set_xlabel("Fitted")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=10, density=True, edgecolor="black", alpha=0.45)
    kde = stats.gaussian_kde(residuals)
    x_min, x_max = min(residuals), max(residuals)
    padding = (x_max - x_min) * 0.15 if x_max > x_min else 1
    x_grid = [x_min - padding + i * (x_max - x_min + 2 * padding) / 199 for i in range(200)]
    axes[1].plot(x_grid, kde(x_grid), linewidth=2, color="tab:blue", label="KDE")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Residual Histogram + KDE")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("QQ Plot")

    plt.tight_layout()
    return fig


def predict_price(result, features, target):
    x = pd.DataFrame([{col: target[col] for col in features}])
    x = sm.add_constant(x, has_constant="add")
    x = x[result["model"].model.exog_names]
    ln_pred = float(result["model"].predict(x).iloc[0])
    return ln_pred, math.exp(ln_pred)
