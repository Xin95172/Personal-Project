# 第三章 研究方法 (Research Methodology)

本研究旨在透過自然語言處理（NLP）與機器學習技術，針對智慧財產權相關判決書進行自動化分類與勝敗訴預測。研究流程包含資料收集、前處理、特徵提取、模型訓練與效能評估。

## 3.1 資料收集 (Data Collection)

本研究之資料來源為司法院公開之判決書數據（JSON 格式）。首先透過關鍵字篩選機制，提取與智慧財產權高度相關之案件。

- **篩選關鍵字**：包含「智慧財產」、「專利」、「商標」、「著作權」及「營業秘密」。
- **資料清理**：針對符合篩選條件之判決書，提取其案號（JCASE）、主文（JTITLE）及全文（JFULL），並移除文本中的換行符號（`\r\n`）與多餘空白，以利後續自然語言處理程式讀取。

## 3.2 資料前處理 (Data Preprocessing)

由於法律判決書屬於長文本且包含許多法律專有名詞，本研究採用以下步驟進行前處理：

1.  **中文斷詞 (Word Segmentation)**：採用中央研究院開發的 **CKIP Transformers** 進行斷詞，該模型基於 Transformer 架構，能有效處理繁體中文之語意斷詞，提升法律術語的辨識準確度。
2.  **停用詞過濾 (Stopwords Removal)**：載入繁體中文停用詞表（`stopwords-ch-jiebar-zht.txt`）以及自定義的刪除詞彙表（`delete_vocab.txt`），過濾掉對分類無顯著幫助的虛詞、連接詞與常見冗詞。
3.  **標籤整合**：整合判決書之勝敗訴結果（如：勝訴、敗訴、部分勝訴/敗訴）作為模型訓練的目標標籤（Metadata）。

## 3.3 特徵工程 (Feature Engineering)

本研究將非結構化的文本數據轉換為機器可讀的數值特徵向量：

- **詞袋模型 (Bag-of-Words, BoW)**：建立文件-詞彙矩陣（Document-Term Matrix, DTM），計算每個詞彙在判決書中出現的頻率。程式碼中亦保留了 TF-IDF (Term Frequency-Inverse Document Frequency) 的實作，但在主要的模型訓練階段（如神經網路與樹模型）優先使用了 BoW 特徵矩陣（`dtm_csr_BoW`）。
- **資料集劃分**：將資料集依照 **80:10:10** 的比例劃分為訓練集（Training Set）、驗證集（Validation Set）與測試集（Test Set），以確保模型評估的客觀性。
    - 訓練集：80%
    - 驗證集：10%
    - 測試集：10%

## 3.4 機器學習模型 (Machine Learning Models)

本研究採用多種監督式學習演算法進行比較分析，探討不同模型在法律文本分類上的效能：

1.  **單純貝氏分類器 (Naive Bayes Classifier)**：使用 MultinomialNB，適用於離散計數特徵（如 BoW）的文本分類任務。
2.  **決策樹 (Decision Tree)**：設定最大深度（max_depth）為 10，採用資訊熵（Entropy）作為分裂準則，以建立可解釋性較高的樹狀模型。
3.  **隨機森林 (Random Forest)**：集成 100 棵決策樹（n_estimators=100），同樣設定最大深度為 10，利用多棵樹的投票機制降低過擬合風險。
4.  **K-近鄰演算法 (K-Nearest Neighbors, KNN)**：設定鄰居數（K）為 5，採用閔可夫斯基距離（Minkowski distance, p=2，即歐幾里得距離）計算文本相似度。
5.  **支持向量機 (Support Vector Machine, SVM)**：設定最大迭代次數為 5000 次，尋找能將不同類別樣本區分開的最佳超平面。
6.  **類神經網路 (Neural Network)**：建構全連接神經網路（Fully Connected Network），包含隱藏層（維度設定為 [512, 256]），並加入 Dropout (0.3) 機制以防止過擬合，使用 CrossEntropyLoss 作為損失函數進行訓練。

## 3.5 模型評估指標 (Evaluation Metrics)

為全面評估模型效能，本研究採用以下指標，並針對「勝訴」、「敗訴」及「部分勝訴/敗訴」各類別分別計算：

- **準確率 (Accuracy)**：模型正確分類案件的比例。
- **精確率 (Precision)**：在模型預測為某一類別的案件中，實際屬於該類別的比例。
- **召回率 (Recall)**：在實際為某一類別的案件中，模型成功預測出的比例。
- **F1-Score**：精確率與召回率的調和平均數，用於綜合評估模型在各類別上的表現。
- **混淆矩陣 (Confusion Matrix)**：視覺化呈現模型在各類別間的預測分佈與誤判情形。
