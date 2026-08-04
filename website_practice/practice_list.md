# 網站設計師完整知識地圖（Website Development Roadmap）

> 適用對象：想成為網站設計師、前端工程師、全端工程師、網站接案者、數位行銷人員

---

## 目錄

- [Part 1：知識地圖](#part-1知識地圖) — 網站開發所需的完整知識架構
- [Part 2：作品集規劃](#part-2作品集規劃) — 每個作品對應的練習目標與能力展現

---

# Part 1：知識地圖

## 一、網站完整架構

一個真正可以營運的網站，由下列系統共同組成：

```
網域(Domain)
        │
        ▼
DNS
        │
        ▼
主機(Server / Hosting)
        │
        ▼
網站程式
(Next.js / WordPress / Laravel ...)
        │
        ├──────────────┐
        ▼              ▼
   前端網站        後端API
        │              │
        ▼              ▼
      使用者       資料庫(Database)
                      │
                      ▼
                 第三方服務(API)
```

---

## 二、網站基礎

### 1. 網域（Domain）

例如：`example.com`、`google.com`、`apple.com`

- **用途**：網站網址、Email 網域、SSL 憑證、DNS 設定、子網域
- **常見服務**：GoDaddy、Namecheap、Cloudflare、Porkbun

### 2. DNS

負責把網址導向網站。

- **常見設定**：A Record、CNAME、MX、TXT、SPF、DKIM

### 3. Hosting（主機）

網站真正執行的地方。

- **常見平台**：Netlify、Vercel、Railway、Render、Hostinger、VPS、AWS、GCP、Azure
- **主要功能**：SSL、CDN、Cache、自動部署、Rollback、Log、Environment Variables

---

## 三、網站程式

### 前端

- **負責**：UI、UX、動畫、表單、互動
- **常見技術**：HTML、CSS、JavaScript、TypeScript、React、Vue、Next.js、Tailwind CSS

### 後端

- **負責**：API、驗證、金流、訂單、資料存取、Email、權限
- **常見框架**：Next.js API、Node.js、Express、NestJS、Laravel、Django

### 資料庫

- **存放內容**：會員、商品、訂單、文章、留言、問答、金流紀錄、Log
- **常見方案**：PostgreSQL、MySQL、MongoDB、Supabase、Firebase

---

## 四、前台功能

### 首頁

Hero Banner、品牌介紹、服務、商品、CTA、FAQ、Footer

### 導覽列

Logo、Menu、Search、Login、Cart、Language

### 商品

商品列表、商品詳細、規格、庫存、價格、圖片、評價、推薦商品

### 部落格

分類、標籤、作者、SEO、留言、分享

### FAQ

搜尋、分類、收合

### 聯絡我們

表單、Google Map、電話、Email、LINE、Messenger

---

## 五、會員系統

| 功能 | 細項 |
|---|---|
| 登入 | Email、Password、OAuth、Magic Link |
| 註冊 | Email 驗證、手機驗證、社群登入 |
| 忘記密碼 | Email Reset |
| 個人資料 | 姓名、Email、密碼、頭像 |
| 會員中心 | 訂單、收藏、預約、發票、點數 |

---

## 六、權限系統

角色層級：

```
Admin > Editor > Author > Member > Guest
```

| 角色 | 權限範圍 |
|---|---|
| Admin | 新增文章、刪除文章、編輯商品、看訂單、看會員 |
| Member | 看自己的訂單、修改自己的資料 |

---

## 七、後台 CMS

| 管理模組 | 功能 |
|---|---|
| 文章管理 | 新增、修改、刪除、草稿、發布 |
| 商品管理 | 上架、下架、價格、庫存 |
| 會員管理 | 權限、停權、查詢 |
| 其他管理 | 留言管理、問答管理、Banner 管理、首頁管理、SEO 管理 |

---

## 八、購物車

加入購物車、修改數量、刪除、優惠券、小計、運費

---

## 九、訂單系統

### 流程

```
加入購物車 → 填寫資料 → 付款 → 建立訂單 → 寄 Email → 物流 → 完成
```

### 訂單內容

訂單編號、狀態、付款、出貨

---

## 十、金流

- **常見平台**：綠界、藍新、Stripe、PayPal、TapPay、Line Pay、Apple Pay、Google Pay
- **付款方式**：信用卡、ATM、超商代碼、超商條碼、匯款
- **技術需求**：Callback、Webhook、簽章驗證

---

## 十一、物流

- **串接平台**：黑貓、新竹物流、宅配通、7-11、全家
- **功能**：出貨、配送、查詢

---

## 十二、電子發票

- **串接平台**：綠界、ezPay、財政部
- **功能**：開立、作廢、折讓、載具

---

## 十三、預約系統

- **適用場景**：美容、診所、健身房
- **功能**：行事曆、時段、預約、Email 通知

---

## 十四、客服系統

可串接：LINE、Messenger、WhatsApp、Crisp、Zendesk、Tawk.to

---

## 十五、通知系統

Email、SMS、LINE Notify、LINE OA、Firebase Push、Web Push

---

## 十六、搜尋

關鍵字、Auto Complete、Filter、Sort

---

## 十七、SEO

Title、Description、Canonical、robots、sitemap.xml、schema.org、Open Graph、Twitter Card

---

## 十八、分析

Google Analytics、Google Search Console、Hotjar、Microsoft Clarity、Meta Pixel、TikTok Pixel、LinkedIn Insight、Google Tag Manager

---

## 十九、AI

- **可串接平台**：OpenAI、Claude、Gemini、DeepSeek
- **用途**：客服、文案、SEO、商品描述、問答

---

## 二十、API 串接

常見第三方 API：Google Maps、Weather、Facebook、Instagram、YouTube、OpenAI、LINE、Discord、Slack、Notion、Google Calendar、Google Drive

---

## 二十一、檔案管理

- **支援格式**：圖片、PDF、Word、Excel、影片、音訊
- **儲存方案**：Supabase Storage、S3、Cloudinary

---

## 二十二、安全

HTTPS、JWT、Session、OAuth、CSRF、XSS、SQL Injection、Rate Limit、Captcha、2FA、Backup

---

## 二十三、網站速度

圖片壓縮、Lazy Loading、CDN、Cache、Minify、SSR、ISR、Edge Functions

---

## 二十四、多語系

- **語言**：繁中、英文、日文、韓文
- **功能**：URL 切換、自動偵測、翻譯

---

## 二十五、網站維運

每日事項：備份、更新、Log、Error、Security

---

## 二十六、部署流程

```
VS Code → Git → GitHub → Netlify / Vercel → 正式網站
```

---

## 二十七、網站專案交付清單

| 類別 | 交付項目 |
|---|---|
| 前端 | ✅ RWD（手機版）、✅ SEO、✅ 首頁、✅ 各功能頁、✅ 表單 |
| 後端 | ✅ API、✅ 資料庫、✅ 登入、✅ 權限 |
| 第三方 | ✅ 金流、✅ 物流、✅ 發票、✅ Email、✅ Google Analytics、✅ Search Console |
| 後台 | ✅ 商品管理、✅ 文章管理、✅ Banner 管理、✅ 問答管理、✅ 會員管理、✅ 訂單管理 |
| 安全 | ✅ SSL、✅ RLS、✅ JWT、✅ Backup |
| 維運 | ✅ GitHub Repository、✅ Netlify/Vercel、✅ Domain、✅ DNS、✅ Environment Variables、✅ README、✅ 操作手冊 |

---

## 二十八、建議學習順序（由淺入深）

1. HTML
2. CSS
3. JavaScript
4. TypeScript
5. Git / GitHub
6. React
7. Next.js
8. Tailwind CSS
9. API（REST）
10. 資料庫（PostgreSQL / Supabase）
11. Authentication（登入、權限）
12. 後台 CMS
13. 金流串接
14. 物流串接
15. 電子發票
16. SEO
17. Google Analytics / Search Console
18. Docker（可選）
19. AWS / GCP（進階）
20. CI/CD（GitHub Actions、Netlify、Vercel）
21. AI API（OpenAI、Gemini 等）
22. 系統架構與效能優化

---

## 最終目標能力

完成以上內容後，你將能獨立規劃、開發與部署一個具備商業營運能力的網站，包括：

- 企業形象官網
- 電商網站
- 新聞媒體平台
- 部落格／內容管理系統（CMS）
- 預約系統
- 會員網站
- SaaS 平台
- AI 應用網站
- 具備完整後台、金流、物流、SEO、分析與維運能力的全端網站。

---
---

# Part 2：作品集規劃

> 目標不是單純累積網站數量，而是讓每個作品都對應一組明確能力。
> 建議作品集配置為：1 個完整主專案＋2～3 個不同類型網站。

---

## 一、內容管理／媒體網站

### 專案定位

適合製作：新聞網站、專欄平台、品牌知識網站、問答平台、部落格 CMS、內容型企業網站

### 主要練習內容

#### 1. 前台內容呈現

首頁資訊架構、文章列表、文章詳細頁、分類與標籤、作者資訊、相關文章、搜尋功能、分頁功能

#### 2. 後台 CMS

新增文章、編輯文章、刪除文章、草稿與發布、文章預覽、封面圖片上傳、文章分類管理、作者管理、發布時間管理

#### 3. 會員與權限

- 登入、登出、忘記密碼、Session
- 管理員角色、一般會員角色
- 後台權限驗證、API 權限驗證、Supabase RLS

#### 4. 資料庫設計

設計表格：

```text
users、profiles、articles、categories、article_categories、questions、answers、media
```

理解關聯：一對一、一對多、多對多、外鍵、權限、資料狀態

#### 5. SEO

動態 title、Meta description、Open Graph、Sitemap、robots.txt、Canonical、文章 Schema、Breadcrumb、內部連結、可讀網址

### 展現能力

| 面向 | 內容 |
|---|---|
| 技術能力 | Next.js App Router、Supabase、Auth、API Route、CRUD、RLS、Server／Client Component、雲端部署 |
| 商業能力 | 內容平台規劃、SEO 架構、使用者閱讀動線、後台營運需求、編輯流程設計 |

> **作品集重點**：我不只會做漂亮頁面，也能建立一套可登入、可管理、可發布、可持續營運的內容系統。

---

## 二、小型電商網站

### 專案定位

適合製作：健康食品商城、咖啡豆商城、香氛品牌、文創商品、小型服飾店、地方特產商店

### 主要練習內容

#### 1. 商品系統

商品列表、商品詳細頁、商品分類、商品規格、商品圖片、售價、庫存、上架與下架、推薦商品

#### 2. 購物車

加入購物車、修改數量、刪除商品、規格判斷、小計、運費、優惠券、總金額

#### 3. 訂單系統

建立訂單、訂單編號、訂單項目、訂單狀態、付款狀態、出貨狀態、取消訂單、訂單查詢

#### 4. 金流串接

- **練習流程**：金流測試環境、建立付款交易、回傳付款頁、Callback、Webhook、簽章驗證、付款成功、付款失敗、重複通知處理、退款流程
- **可選擇平台**：綠界、藍新、Stripe、TapPay、LINE Pay

#### 5. 物流與發票

進階練習：超商取貨、宅配、物流單號、配送狀態、電子發票、載具、統一編號、發票作廢、折讓

#### 6. 會員中心

訂單紀錄、收件地址、收藏商品、會員資料、優惠券、點數

### 展現能力

| 面向 | 內容 |
|---|---|
| 技術能力 | 商品資料建模、購物車狀態管理、訂單流程、金流 API、Webhook、資料一致性、後端安全 |
| 商業能力 | 電商轉換流程、商品資訊層級、結帳流程、付款失敗處理、訂單營運流程、後台訂單管理 |

> **作品集重點**：我理解的不只是付款按鈕，而是商品、購物車、訂單、付款、庫存、物流與退款之間的完整商業流程。

---

## 三、預約／服務型網站

### 專案定位

適合製作：美甲預約、攝影工作室、顧問預約、健身教練、民宿、課程預約、餐廳訂位、場地租借

### 主要練習內容

#### 1. 服務項目

服務名稱、服務介紹、所需時間、價格、人員、適用對象、注意事項

#### 2. 時段管理

可預約日期、可預約時段、休假日、已滿時段、人員排班、特殊時段、預約名額

#### 3. 預約流程

```text
選擇服務 → 選擇人員 → 選擇日期 → 選擇時段 → 填寫資料 → 支付訂金 → 完成預約
```

#### 4. 防止重複預約

後端重新驗證、Database Constraint、Transaction、時段鎖定、競爭條件、重複送出防護

#### 5. 通知與行事曆

預約成功 Email、預約提醒、改期通知、取消通知、Google Calendar、LINE 通知、簡訊通知

#### 6. 後台管理

預約列表、日期篩選、人員篩選、修改狀態、改期、取消、完成服務、匯出資料

### 展現能力

| 面向 | 內容 |
|---|---|
| 技術能力 | 日期時間處理、狀態管理、預約 API、行事曆串接、防止重複預約、Email／通知系統、後台篩選 |
| 商業能力 | 服務流程設計、預約阻力降低、時段管理、人力排班、訂金邏輯、客戶通知流程 |

> **作品集重點**：我能處理具有時間、名額、人員與狀態限制的服務流程，而不只是建立一張聯絡表單。

---

## 四、企業形象／SEO 官網

### 專案定位

適合製作：顧問公司、徵信社、律師事務所、健康食品品牌、建設公司、在地服務業、B2B 公司、專業服務品牌

### 主要練習內容

#### 1. 品牌視覺

品牌定位、主色、輔助色、字體、圖片風格、按鈕系統、卡片系統、整體視覺一致性

#### 2. 商業文案

Hero 標題、品牌價值、服務介紹、信任元素、FAQ、CTA、聯絡導流、風險與限制說明

#### 3. 轉換設計

流程：

```text
搜尋進站 → 看懂服務 → 建立信任 → 消除疑慮 → 加入 LINE／填表／打電話
```

追蹤指標：LINE 點擊、電話點擊、表單送出、服務頁瀏覽、CTA 點擊

#### 4. SEO

關鍵字規劃、搜尋意圖、服務頁 SEO、在地 SEO、FAQ Schema、Organization Schema、Breadcrumb Schema、Google Search Console、Sitemap、內部連結

#### 5. RWD 與效能

手機版設計、LINE 內建瀏覽器、iPhone Safari、圖片壓縮、WebP、Lazy Loading、Core Web Vitals、CTA 點擊區域

### 展現能力

| 面向 | 內容 |
|---|---|
| 設計能力 | 品牌視覺、UI 系統、RWD、圖文編排、商業質感 |
| 行銷能力 | SEO、CTA、使用者動線、信任建立、轉換追蹤、內容策略 |

> **作品集重點**：我不只會製作畫面，也能根據品牌定位、搜尋需求與轉換目標，設計真正有商業用途的網站。

---

## 五、個人作品集網站

### 專案定位

用來整合以上所有作品與能力。

### 主要練習內容

#### 1. 個人定位

我是誰、我擅長什麼、我服務哪些客戶、我使用哪些技術、我如何解決問題

#### 2. 作品案例

每個案例不要只放截圖，應包含：

```text
專案背景、問題、目標、角色、資訊架構、設計決策、技術架構、開發流程、遇到的問題、解決方式、成果、下一步
```

#### 3. 能力分類

```text
網站規劃、UI／UX、品牌視覺、Next.js、Supabase、後台 CMS、SEO、GA4、金流、自動化、部署
```

#### 4. 聯絡與轉換

Email、LINE、履歷、GitHub、LinkedIn、合作表單、服務項目

### 展現能力

個人品牌、專案整理能力、問題解決能力、設計思考、技術溝通能力、商業理解、提案能力

> **作品集重點**：我能把設計、開發、SEO、資料分析與商業流程整合成完整解決方案。

---

## 六、四個作品的能力分工

| 專案 | 主要練習 | 主要展現 |
|---|---|---|
| 內容管理／媒體網站 | CMS、Auth、CRUD、RLS、SEO | 全端開發與內容營運能力 |
| 小型電商 | 商品、購物車、訂單、金流、Webhook | 商業流程與第三方 API 串接 |
| 預約／服務網站 | 時段、人員、通知、行事曆 | 複雜流程與狀態管理 |
| 企業形象／SEO 官網 | 品牌、RWD、SEO、CTA、分析 | 視覺設計與數位行銷能力 |
| 個人作品集 | 案例整理、個人定位、成果呈現 | 綜合能力與專業形象 |

---

## 七、建議完成順序

### 第一階段：內容管理網站

- 完成：登入、管理員權限、文章 CRUD、問答管理、圖片上傳、SEO、部署
- > 核心目標：建立第一個真正完整、可以持續營運的網站。

### 第二階段：小型電商

- 完成：商品、購物車、訂單、測試金流、Webhook、後台訂單管理
- > 核心目標：理解網站如何與真實商業交易流程連接。

### 第三階段：預約系統

- 完成：服務、時段、預約、通知、行事曆、後台
- > 核心目標：練習複雜條件、時間與狀態管理。

### 第四階段：企業形象官網

- 完成：品牌視覺、商業文案、RWD、SEO、GA4、LINE 導流
- > 核心目標：展現設計、行銷與轉換能力。

### 第五階段：作品集網站

- 完成：個人定位、案例研究、技術能力、聯絡方式、GitHub、Demo
- > 核心目標：把做過的網站轉化成可用於求職與接案的證明。

---

## 八、每個專案都應該回答的問題

| 面向 | 問題 |
|---|---|
| 規劃面 | 這個網站解決什麼問題？使用者是誰？商業目標是什麼？核心行動是什麼？ |
| 設計面 | 為什麼使用這套色彩？為什麼這樣安排頁面？手機版如何處理？如何降低使用者疑慮？ |
| 技術面 | 使用什麼技術？資料庫如何設計？權限如何管理？API 如何運作？如何部署？ |
| 營運面 | 後台如何操作？SEO 如何做？流量如何追蹤？錯誤如何處理？如何維護與備份？ |

---

## 九、作品完成標準

每個作品至少要有：

- [ ] 清楚的專案目標
- [ ] Sitemap
- [ ] User Flow
- [ ] 桌機版
- [ ] 手機版
- [ ] 真實內容
- [ ] 正常功能
- [ ] 錯誤處理
- [ ] Loading 狀態
- [ ] 權限驗證
- [ ] SEO
- [ ] Analytics
- [ ] GitHub
- [ ] 雲端 Demo
- [ ] README
- [ ] 專案案例說明

---

## 十、最終作品集定位

完成這些作品後，你可以對外呈現：

> 我能規劃並製作企業官網、內容管理平台、小型電商與預約系統，具備品牌視覺、RWD、SEO、後台管理、會員權限、資料庫、金流串接、第三方 API 與雲端部署能力。