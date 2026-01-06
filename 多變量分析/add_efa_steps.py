import json
import os

nb_path = r'd:/Github/Personal-Project/多變量分析/research.ipynb'

print(f"Reading notebook from {nb_path}")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Define new cells
new_cells_content = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. 因子命名與解釋 (Factor Interpretation)\n",
            "\n",
            "列出每個因子中，負荷量 (Loading) 絕對值大於 0.5 的變數，以輔助命名。"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def get_factor_loadings_report(loadings_df, threshold=0.5):\n",
            "    factors = loadings_df.columns\n",
            "    report = {}\n",
            "    for factor in factors:\n",
            "        # 篩選出負荷量絕對值 > threshold 的變數\n",
            "        high_loading_vars = loadings_df[factor][abs(loadings_df[factor]) > threshold]\n",
            "        # 依絕對值大小排序\n",
            "        high_loading_vars = high_loading_vars.reindex(\n",
            "            high_loading_vars.abs().sort_values(ascending=False).index\n",
            "        )\n",
            "        report[factor] = high_loading_vars\n",
            "        \n",
            "    return report\n",
            "\n",
            "loading_report = get_factor_loadings_report(loadings, threshold=0.5)\n",
            "\n",
            "for factor, vars_series in loading_report.items():\n",
            "    print(f\"=== {factor} ===\")\n",
            "    if vars_series.empty:\n",
            "        print(\"  (No variables > threshold)\")\n",
            "    else:\n",
            "        for var_name, loading_val in vars_series.items():\n",
            "            print(f\"  {var_name}: {loading_val:.3f}\")\n",
            "    print()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. 信度分析 (Reliability Analysis)\n",
            "\n",
            "使用 **Cronbach's Alpha** 檢驗各因子內部一致性 (Internal Consistency)。\n",
            "*   $\\alpha > 0.7$: 信度可接受\n",
            "*   $\\alpha > 0.8$: 信度良好"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def cronbach_alpha(df):\n",
            "    # 1. 計算變數數 k\n",
            "    k = df.shape[1]\n",
            "    if k < 2:\n",
            "        return np.nan\n",
            "    \n",
            "    # 2. 計算各變數變異數總和\n",
            "    sum_var_items = df.var(ddof=1).sum()\n",
            "    \n",
            "    # 3. 計算總分變異數\n",
            "    var_total = df.sum(axis=1).var(ddof=1)\n",
            "    \n",
            "    # 4. 公式\n",
            "    alpha = (k / (k - 1)) * (1 - (sum_var_items / var_total))\n",
            "    return alpha\n",
            "\n",
            "print(\"Cronbach's Alpha per Factor:\")\n",
            "for factor in loading_report.keys():\n",
            "    # 取出該因子下的變數名稱\n",
            "    items = loading_report[factor].index.tolist()\n",
            "    \n",
            "    if len(items) > 1:\n",
            "        # 從原始資料 df_X 中取出這些欄位\n",
            "        factor_df = df_X[items]\n",
            "        alpha = cronbach_alpha(factor_df)\n",
            "        print(f\"  {factor} (items={len(items)}): {alpha:.4f}\")\n",
            "    else:\n",
            "        print(f\"  {factor}: Item count < 2, cannot calculate Alpha\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. 因子分數萃取 (Factor Scores Extraction)\n",
            "\n",
            "計算每筆資料在各個因子上的得分，以便進行後續分析 (Regression, Clustering etc.)。"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 計算因子分數\n",
            "factor_scores = fa.transform(df_X)\n",
            "\n",
            "# 轉成 DataFrame\n",
            "df_scores = pd.DataFrame(\n",
            "    factor_scores, \n",
            "    columns=[f'Score_{i+1}' for i in range(factor_scores.shape[1])]\n",
            ")\n",
            "\n",
            "# 合併回原始資料 (Optional)\n",
            "df_full = pd.concat([df_X.reset_index(drop=True), df_scores], axis=1)\n",
            "\n",
            "print(\"因子分數前 5 筆:\")\n",
            "df_scores.head()"
        ]
    }
]

print(f"Appending {len(new_cells_content)} new cells to notebook...")
cells.extend(new_cells_content)

nb['cells'] = cells

print(f"Writing updated notebook to {nb_path}...")
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Successfully added EFA steps.")
