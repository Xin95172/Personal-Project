import json
import os

nb_path = r'd:/Github/Personal-Project/多變量分析/research.ipynb'

print(f"Reading notebook from {nb_path}")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the new structure content
cells = []

# --- 1. Setup & Data ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 探索性因子分析 (Exploratory Factor Analysis, EFA)",
        "\n",
        "本 Notebook 展示如何使用 Python 進行完整的 EFA 分析流程。主要包含以下步驟：\n",
        "\n",
        "1.  **環境設定與資料準備**：載入套件與模擬資料。\n",
        "2.  **資料適性檢定**：檢查資料是否適合進行因子分析 (KMO, Bartlett)。\n",
        "3.  **決定因子個數**：透過陡坡圖 (Scree Plot) 與平行分析 (Parallel Analysis) 決定 $K$ 值。\n",
        "4.  **因子萃取與旋轉**：使用 `Promax` 旋轉進行因子擬合。\n",
        "5.  **因子模型檢驗**：檢查共同性 (Communalities) 與因子相關性。\n",
        "6.  **因子解釋**：檢視因子負荷量 (Factor Loadings) 並替因子命名。\n",
        "7.  **信度分析**：計算 Cronbach's Alpha 以確認內部一致性。\n",
        "8.  **因子分數萃取**：產出因子分數供後續分析使用。"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. 環境設定與資料準備 (Setup & Data Preparation)"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import plotly.express as px\n",
        "import plotly.graph_objects as go\n",
        "from factor_analyzer import FactorAnalyzer\n",
        "from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity\n",
        "from factor_analyzer.factor_analyzer import calculate_kmo\n",
        "import warnings\n",
        "\n",
        "# 忽略部分套件警告版本問題\n",
        "warnings.filterwarnings('ignore')"
    ]
})

# Recopy the simulation function from the original notebook to ensure no code loss
# I'll hardcode it here based on what I read, to ensure it's clean and preserved.
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def simulate_efa_features(\n",
        "    N=3000,                 # 新聞篇數\n",
        "    K=5,                    # 潛在因子數 (縮減為 5)\n",
        "    feature_names=None,     # features 名稱\n",
        "    make_share=True,        # 比例型資料\n",
        "    noise_sd=0.6,           # 雜訊\n",
        "    seed=42\n",
        "):\n",
        "    rng = np.random.default_rng(seed)\n",
        "\n",
        "    # 定義因子結構字典 {Factor_Index: [Feature_List]}\n",
        "    factor_structure = {\n",
        "        0: [\"pos\", \"neg\", \"modal_strong\", \"modal_weak\"], \n",
        "        1: [\"uncertainty\", \"litigious\", \"constraining\", \"risk\", \"conflict\"],\n",
        "        2: [\"esg\", \"environment\", \"social\", \"governance\", \"carbon\", \"diversity\"], \n",
        "        3: [\"inflation\", \"interest_rate\", \"forex\", \"political\", \"supply_chain\", \"oil_price\"], \n",
        "        4: [\"profitability\", \"growth\", \"liquidity\", \"debt\", \"innovation\", \"tech\", \"merger_acquisition\"] \n",
        "    }\n",
        "\n",
        "    # 若未指定 feature_names，從結構中生成\n",
        "    if feature_names is None:\n",
        "        feature_names = []\n",
        "        for feats in factor_structure.values():\n",
        "            feature_names.extend(feats)\n",
        "        # 去除重複並保持順序\n",
        "        feature_names = list(dict.fromkeys(feature_names))\n",
        "    \n",
        "    P = len(feature_names)\n",
        "    feat_idx = {name: i for i, name in enumerate(feature_names)}\n",
        "    \n",
        "    # 1) 生成潛在因子 F (N x K)\n",
        "    F = rng.normal(0, 1, size=(N, K))\n",
        "    \n",
        "    # 2) 生成 Loading Matrix L (P x K)\n",
        "    L = np.zeros((P, K))\n",
        "    \n",
        "    for k, subjects in factor_structure.items():\n",
        "        if k >= K: continue\n",
        "        for subj in subjects:\n",
        "            if subj in feat_idx:\n",
        "                # 主負荷量 (Main Loading)\n",
        "                L[feat_idx[subj], k] = rng.uniform(0.65, 0.95)\n",
        "                \n",
        "                # 隨機添加一些 Cross-loading\n",
        "                if rng.random() < 0.15: \n",
        "                    other_k = rng.integers(0, K)\n",
        "                    if other_k != k:\n",
        "                        L[feat_idx[subj], other_k] = rng.uniform(0.2, 0.4)\n",
        "\n",
        "    # 手動添加一些邏輯上的 Cross-loadings\n",
        "    if \"neg\" in feat_idx and 1 < K: \n",
        "        L[feat_idx[\"neg\"], 1] += 0.4 \n",
        "    if \"debt\" in feat_idx and 1 < K:\n",
        "        L[feat_idx[\"debt\"], 1] += 0.5 \n",
        "    if \"inflation\" in feat_idx and 1 < K:\n",
        "        L[feat_idx[\"inflation\"], 1] += 0.3 \n",
        "    if \"supply_chain\" in feat_idx and 4 < K:\n",
        "        L[feat_idx[\"supply_chain\"], 4] += 0.3 \n",
        "\n",
        "    # 3) 生成 features: X = F @ L.T + noise\n",
        "    X = F @ L.T + rng.normal(0, noise_sd, size=(N, P))\n",
        "    X = X + np.abs(X.min()) + 0.1\n",
        "    \n",
        "    # 4) 轉為比例 (Share)\n",
        "    if make_share:\n",
        "        row_sum = X.sum(axis=1, keepdims=True)\n",
        "        X = X / np.clip(row_sum, 1e-8, None)\n",
        "        \n",
        "    df_X = pd.DataFrame(X, columns=feature_names)\n",
        "    return df_X, L"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 產生測試資料\n",
        "df_X, L_true = simulate_efa_features(\n",
        "    N=3000,\n",
        "    K=5,\n",
        "    make_share=False,\n",
        "    noise_sd=0.6\n",
        ")\n",
        "\n",
        "print(f\"資料筆數: {len(df_X)}\")\n",
        "print(f\"變數欄位: {df_X.columns.tolist()}\")\n",
        "df_X.head()"
    ]
})

# --- 2. Suitability ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. 資料適性檢定 (Data Suitability Tests)\n",
        "\n",
        "在進行因子分析前，需確認變數間是否存在足夠的相關性。\n",
        "\n",
        "*   **Bartlett’s Test of Sphericity**：\n",
        "    *   檢查變數之間是否互相獨立。H0: 變數間無相關（矩陣為單位矩陣）。\n",
        "    *   若 **p-value < 0.05**，則拒絕 H0，表示變數間有相關性，**適合**進行 EFA。\n",
        "*   **KMO (Kaiser-Meyer-Olkin) Test**：\n",
        "    *   測量變數間的偏相關性是否夠小，數值介於 0~1。\n",
        "    *   一般準則：**> 0.6** 為勉強接受，**> 0.7** 為中等，**> 0.8** 為良好。"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 1. Bartlett's Test\n",
        "chi_square_value, p_value = calculate_bartlett_sphericity(df_X)\n",
        "print(f\"Bartlett's Test p-value: {p_value:.4e}\")\n",
        "\n",
        "# 2. KMO Test\n",
        "kmo_all, kmo_model = calculate_kmo(df_X)\n",
        "print(f\"KMO Test Value: {kmo_model:.4f}\")\n",
        "\n",
        "if p_value < 0.05 and kmo_model > 0.6:\n",
        "    print(\"=> 結果：資料適合進行因子分析\")\n",
        "else:\n",
        "    print(\"=> 結果：資料可能不適合進行因子分析，請檢查變數相關性\")"
    ]
})

# --- 3. Selection ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. 決定因子個數 (Factor Selection)\n",
        "\n",
        "我們要決定保留多少個因子 ($K$)。常用的準則有：\n",
        "\n",
        "1.  **Scree Plot (陡坡圖)**：尋找特徵值 (Eigenvalue) 曲線的「轉折點」 (Elbow Point)。\n",
        "2.  **Kaiser Criterion**：保留特徵值 **> 1** 的因子。\n",
        "3.  **Parallel Analysis (平行分析)**：比較真實數據與隨機數據的特徵值，若真實數據特徵值 > 隨機數據 95% 分位數，則保留。此法通常比 Kaiser Criterion 更穩健。"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "fa = FactorAnalyzer(n_factors=len(df_X.columns), rotation=None)\n",
        "fa.fit(df_X)\n",
        "\n",
        "# 取得真實特徵值\n",
        "ev, v = fa.get_eigenvalues()\n",
        "\n",
        "# --- 平行分析 (Parallel Analysis) ---\n",
        "def run_parallel_analysis(df, n_iter=100, seed=42):\n",
        "    n, p = df.shape\n",
        "    rng = np.random.default_rng(seed)\n",
        "    random_eigenvalues = []\n",
        "    \n",
        "    print(f\"進行平行分析中 (模擬 {n_iter} 次)...\")\n",
        "    for _ in range(n_iter):\n",
        "        random_data = rng.normal(0, 1, size=(n, p))\n",
        "        corr_matrix = np.corrcoef(random_data, rowvar=False)\n",
        "        evs = np.linalg.eigvalsh(corr_matrix)\n",
        "        random_eigenvalues.append(evs[::-1])\n",
        "        \n",
        "    random_eigenvalues = np.array(random_eigenvalues)\n",
        "    percentile_95 = np.percentile(random_eigenvalues, 95, axis=0)\n",
        "    return percentile_95\n",
        "\n",
        "random_ev_95 = run_parallel_analysis(df_X)\n",
        "\n",
        "# 建議因子個數\n",
        "n_factors_ev1 = sum(ev > 1)\n",
        "n_factors_pa = sum(ev > random_ev_95)\n",
        "\n",
        "print(f\"\\n建議保留因子個數：\")\n",
        "print(f\" - Eigenvalue > 1 準則: {n_factors_ev1}\")\n",
        "print(f\" - Parallel Analysis 準則: {n_factors_pa}\")\n",
        "\n",
        "# --- 繪圖 ---\n",
        "fig = go.Figure()\n",
        "\n",
        "# 真實特徵值\n",
        "fig.add_trace(go.Scatter(\n",
        "    x=list(range(1, len(ev)+1)),\n",
        "    y=ev,\n",
        "    mode='lines+markers',\n",
        "    name='Actual Eigenvalues',\n",
        "    line=dict(color='blue')\n",
        "))\n",
        "\n",
        "# 隨機特徵值\n",
        "fig.add_trace(go.Scatter(\n",
        "    x=list(range(1, len(random_ev_95)+1)),\n",
        "    y=random_ev_95,\n",
        "    mode='lines',\n",
        "    name='Random Data (95th %ile)',\n",
        "    line=dict(color='red', dash='dash')\n",
        "))\n",
        "\n",
        "fig.add_hline(y=1, line_dash=\"dot\", line_color=\"gray\", annotation_text=\"Eigenvalue=1\")\n",
        "\n",
        "fig.update_layout(\n",
        "    title='Scree Plot & Parallel Analysis',\n",
        "    xaxis_title='Factors',\n",
        "    yaxis_title='Eigenvalue',\n",
        "    template='plotly_white'\n",
        ")\n",
        "fig.show()"
    ]
})

# --- 4. Extraction & Rotation ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. 因子萃取與旋轉 (Factor Extraction & Rotation)\n",
        "\n",
        "根據上一步驟，我們將 $K$ 設定為 **5**。\n",
        "\n",
        "*   **Extraction**: 使用預設的 `minres` (Minimum Residual) 或 `ml` (Maximum Likelihood)。這裡維持預設。\n",
        "*   **Rotation**: 使用 **Promax**。這是一種**斜交旋轉 (Oblique Rotation)**，允許因子之間存在相關性。相較於 Varimax (正交旋轉)，Promax 更符合真實社會科學數據特性（概念間通常有關聯）。"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 設定因子數為 5，並使用 promax 旋轉\n",
        "K_FINAL = 5\n",
        "fa = FactorAnalyzer(n_factors=K_FINAL, rotation='promax')\n",
        "fa.fit(df_X)\n",
        "\n",
        "print(f\"已完成因子分析擬合 (Factors={K_FINAL}, Rotation='promax')\")"
    ]
})

# --- 5. Validation (Communalities, Correlation, Variance) ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. 因子模型檢驗 (Model Validation)\n",
        "\n",
        "在解釋因子之前，先確認模型的適配狀況。\n",
        "\n",
        "### 5.1 共同性 (Communalities)\n",
        "共同性代表「每個變數的變異」中，能被「提取出的因子」解釋的比例。若某變數共同性過低 (例如 < 0.2 或 0.3)，表示它與這些因子格格不入，可考慮移除。"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "communalities = fa.get_communalities()\n",
        "comm_df = pd.DataFrame(communalities, index=df_X.columns, columns=['Communalities'])\n",
        "\n",
        "# 檢視低共同性變數\n",
        "low_comm_vars = comm_df[comm_df['Communalities'] < 0.3]\n",
        "if not low_comm_vars.empty:\n",
        "    print(\"\\n[Warning] 以下變數共同性較低 (< 0.3)，可能需考慮移除：\")\n",
        "    print(low_comm_vars)\n",
        "else:\n",
        "    print(\"\\n所有變數共同性皆 > 0.3，保留狀況良好。\")\n",
        "\n",
        "# 繪圖\n",
        "fig = px.bar(comm_df, x=comm_df.index, y='Communalities', title='Variable Communalities')\n",
        "fig.add_hline(y=0.3, line_dash=\"dash\", line_color=\"red\", annotation_text=\"Threshold 0.3\")\n",
        "fig.show()"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 5.2 因子相關矩陣 (Factor Correlation Matrix)\n",
        "因為使用 Promax 斜交旋轉，因子之間不再是獨立的。此矩陣可以告訴我們因子背後的構念是否相關。"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "try:\n",
        "    factor_corr = pd.DataFrame(\n",
        "        fa.phi_,\n",
        "        index=[f'Factor{i+1}' for i in range(fa.phi_.shape[1])],\n",
        "        columns=[f'Factor{i+1}' for i in range(fa.phi_.shape[1])]\n",
        "    )\n",
        "    print(\"因子相關矩陣 (Factor Correlation Matrix):\")\n",
        "    print(factor_corr.round(3))\n",
        "    \n",
        "    fig = px.imshow(\n",
        "        factor_corr, \n",
        "        text_auto='.2f',\n",
        "        color_continuous_scale='RdBu_r', \n",
        "        zmin=-1, zmax=1,\n",
        "        title='Factor Correlation Matrix'\n",
        "    )\n",
        "    fig.show()\n",
        "except:\n",
        "    print(\"無法取得因子相關矩陣 (可能是正交旋轉或未擬合成功)。\")"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 5.3 總解釋變異量 (Total Variance Explained)\n",
        "查看這些因子總共解釋了多少資料變異。"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "variance_tuple = fa.get_factor_variance()\n",
        "variance_arr = np.array(variance_tuple)\n",
        "variance_df = pd.DataFrame(\n",
        "    variance_arr, \n",
        "    index=['SS Loadings', 'Proportion Var', 'Cumulative Var'],\n",
        "    columns=[f'Factor{i+1}' for i in range(variance_arr.shape[1])]\n",
        ")\n",
        "\n",
        "print(f\"累計解釋變異量: {variance_df.iloc[2, -1]:.2%}\")\n",
        "variance_df"
    ]
})

# --- 6. Interpretation ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. 因子解釋與命名 (Factor Interpretation)\n",
        "\n",
        "觀察 **因子負荷量 (Factor Loadings)** 表，找出每個因子由哪些變數組成（通常取 Loadings 絕對值 > 0.4 或 0.5）。\n",
        "*   Loading 代表變數與該因子的相關程度。\n",
        "*   紅色代表正相關，藍色代表負相關。"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "loadings = pd.DataFrame(\n",
        "    fa.loadings_, \n",
        "    index=df_X.columns, \n",
        "    columns=[f'Factor{i+1}' for i in range(fa.loadings_.shape[1])]\n",
        ")\n",
        "\n",
        "fig = px.imshow(\n",
        "    loadings,\n",
        "    x=loadings.columns,\n",
        "    y=loadings.index,\n",
        "    text_auto='.2f',\n",
        "    aspect=\"auto\",\n",
        "    color_continuous_scale='RdBu_r',\n",
        "    origin='upper',\n",
        "    title='Factor Loadings Heatmap'\n",
        ")\n",
        "fig.update_layout(height=800, width=600)\n",
        "fig.show()"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def print_factor_loadings_report(loadings_df, threshold=0.5):\n",
        "    print(f\"--- Factor Loadings Report (Threshold > {threshold}) ---\")\n",
        "    for factor in loadings_df.columns:\n",
        "        # 篩選出負荷量絕對值 > threshold 的變數\n",
        "        high_loading_vars = loadings_df[factor][abs(loadings_df[factor]) > threshold]\n",
        "        # 排序\n",
        "        high_loading_vars = high_loading_vars.reindex(\n",
        "            high_loading_vars.abs().sort_values(ascending=False).index\n",
        "        )\n",
        "        \n",
        "        print(f\"\\n【{factor}】\")\n",
        "        if high_loading_vars.empty:\n",
        "            print(\"  (No variables > threshold)\")\n",
        "        else:\n",
        "            for var_name, val in high_loading_vars.items():\n",
        "                print(f"  {var_name:<20} : {val:.3f}")\n",
        "\n",
        "# 產生報表以供命名\n",
        "loading_report = print_factor_loadings_report(loadings, threshold=0.5)"
    ]
})

# --- 7. Reliability ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7. 信度分析 (Reliability Analysis)\n",
        "\n",
        "針對每個因子所包含的變數，計算其 **Cronbach's Alpha**，以檢驗內部一致性。\n",
        "*   $\\alpha > 0.7$: 可接受\n",
        "*   $\\alpha > 0.8$: 良好"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def cronbach_alpha(df):\n",
        "    k = df.shape[1]\n",
        "    if k < 2: return np.nan\n",
        "    sum_var_items = df.var(ddof=1).sum()\n",
        "    var_total = df.sum(axis=1).var(ddof=1)\n",
        "    alpha = (k / (k - 1)) * (1 - (sum_var_items / var_total))\n",
        "    return alpha\n",
        "\n",
        "print(\"Cronbach's Alpha per Factor (based on items > 0.5 loading):\")\n",
        "\n",
        "for factor in loadings.columns:\n",
        "    # 找出該因子 Loading > 0.5 的變數\n",
        "    items = loadings.index[abs(loadings[factor]) > 0.5].tolist()\n",
        "    \n",
        "    if len(items) >= 2:\n",
        "        factor_df = df_X[items]\n",
        "        alpha = cronbach_alpha(factor_df)\n",
        "        print(f\"  {factor} (Items: {len(items)}): {alpha:.4f}\")\n",
        "    else:\n",
        "        print(f\"  {factor}: 變數過少 (<2)，無法計算 Alpha\")"
    ]
})

# --- 8. Scores ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 8. 因子分數萃取 (Factor Scores)\n",
        "\n",
        "最後，我們計算每筆資料在各個因子上的得分。這些分數可視為降維後的新特徵，用於後續的迴歸、分群等分析。"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "factor_scores = fa.transform(df_X)\n",
        "\n",
        "df_scores = pd.DataFrame(\n",
        "    factor_scores, \n",
        "    columns=[f'Score_{i+1}' for i in range(factor_scores.shape[1])]\n",
        ")\n",
        "\n",
        "# 合併回原始資料\n",
        "df_final = pd.concat([df_X.reset_index(drop=True), df_scores], axis=1)\n",
        "\n",
        "print(\"因子分數範例:\")\n",
        "df_scores.head()"
    ]
})

# Update the notebook object
nb['cells'] = cells

print(f"Writing restructured notebook to {nb_path}...")
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Redesign complete.")
