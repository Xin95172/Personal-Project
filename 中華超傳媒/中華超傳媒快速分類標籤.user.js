// ==UserScript==
// @name         中華超傳媒｜快速加入分類與標籤
// @namespace    https://chinanewscloud.com/
// @version      2.1
// @description  分類只比對；缺少的標籤自動建立並選取
// @match        https://chinanewscloud.com/admin/add-blog*
// @match        https://chinanewscloud.com/admin/edit-blog*
// @grant        none
// ==/UserScript==

(function () {
    "use strict";

    const $ = window.jQuery;
    const TAG_API = "/admin/blog-tag-load";

    const normalize = value =>
        String(value ?? "")
            .replace(/\s+/g, " ")
            .trim()
            .toLowerCase();

    const escapeHtml = value =>
        String(value ?? "").replace(/[&<>"']/g, char => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;",
        })[char]);
    function parseInput(value) {
        const seen = new Set();

        return String(value ?? "")
        // 移除整份陣列最外層的中括號。
            .trim()
            .replace(/^\s*\[/, "")
            .replace(/\]\s*$/, "")

        // 支援換行、逗號、頓號、分號、直線與空格分隔。
            .split(/[\s,，、;；|]+/)

            .map(item =>
                 item
                 .trim()

                 // 移除編號與項目符號。
                 .replace(/^\d+[.、)]\s*/, "")
                 .replace(/^[-*#•]\s*/, "")

                 // 移除前後引號與括號。
                 .replace(/^[\s"'“”‘’\[\]{}()（）]+/, "")
                 .replace(/[\s"'“”‘’\[\]{}()（）]+$/, "")

                 // 合併多餘空白。
                 .replace(/\s+/g, " ")
                 .trim()
                )

            .filter(item => {
            const key = normalize(item);

            if (!key || seen.has(key)) {
                return false;
            }

            seen.add(key);
            return true;
        });
    }
    function findOption(select, name) {
        return [...select.options].find(
            option => normalize(option.textContent) === normalize(name)
        );
    }

    function refreshSelect(select) {
        if ($) {
            $(select).trigger("change");
        } else {
            select.dispatchEvent(
                new Event("change", { bubbles: true })
            );
        }
    }

    function getCsrfToken() {
        const meta =
            document.querySelector('meta[name="_token"]') ||
            document.querySelector('meta[name="csrf-token"]');

        if (!meta?.content) {
            throw new Error("找不到 CSRF Token，請重新整理頁面");
        }

        return meta.content;
    }

    function getTagId(html, tagName) {
        const doc = new DOMParser().parseFromString(
            html,
            "text/html"
        );

        const ids = [...doc.querySelectorAll("option")]
            .filter(
                option =>
                    normalize(option.textContent) ===
                    normalize(tagName)
            )
            .map(option => Number(option.value))
            .filter(Number.isInteger);

        return ids.length ? Math.max(...ids) : null;
    }

    async function createTag(tagName) {
        const body = new URLSearchParams({
            tag: tagName,
            permalink: tagName,
        });

        const response = await fetch(TAG_API, {
            method: "POST",
            credentials: "same-origin",
            headers: {
                "X-CSRF-TOKEN": getCsrfToken(),
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type":
                    "application/x-www-form-urlencoded; charset=UTF-8",
                Accept: "application/json",
            },
            body,
        });

        if (response.status === 419) {
            throw new Error("CSRF Token 已失效，請重新整理頁面");
        }

        if (!response.ok) {
            throw new Error(`新增標籤失敗，HTTP ${response.status}`);
        }

        const result = await response.json();
        const tagId = getTagId(result.view || "", tagName);

        if (!tagId) {
            throw new Error(`回傳內容找不到標籤：${tagName}`);
        }

        return tagId;
    }

    function processCategories(names) {
        const select = document.querySelector("#category-select");
        const found = [];
        const missing = [];

        for (const name of names) {
            const option = findOption(select, name);

            if (option) {
                option.selected = true;
                found.push(name);
            } else {
                missing.push(name);
            }
        }

        refreshSelect(select);

        return { found, missing };
    }

    async function processTags(names, setStatus) {
        const select = document.querySelector("#tag-select");
        const existing = [];
        const created = [];
        const failed = [];

        for (let index = 0; index < names.length; index++) {
            const name = names[index];

            setStatus(
                `正在處理標籤 ${index + 1}/${names.length}：${name}`
            );

            const option = findOption(select, name);

            if (option) {
                option.selected = true;
                existing.push(name);
                continue;
            }

            try {
                const id = await createTag(name);

                select.add(
                    new Option(name, String(id), true, true)
                );

                created.push(name);
            } catch (error) {
                failed.push(`${name}：${error.message}`);
            }
        }

        refreshSelect(select);

        return { existing, created, failed };
    }

    function renderResult(result) {
        const lines = [];

        if (result.categories.found.length) {
            lines.push(
                `<div class="ok">✓ 已加入分類：${result.categories.found
                    .map(escapeHtml)
                    .join("、")}</div>`
            );
        }

        if (result.categories.missing.length) {
            lines.push(
                `<div class="bad">✕ 找不到分類：${result.categories.missing
                    .map(escapeHtml)
                    .join("、")}</div>`
            );
        }

        if (result.tags.existing.length) {
            lines.push(
                `<div class="ok">✓ 已選取標籤：${result.tags.existing
                    .map(escapeHtml)
                    .join("、")}</div>`
            );
        }

        if (result.tags.created.length) {
            lines.push(
                `<div class="new">＋ 已建立標籤：${result.tags.created
                    .map(escapeHtml)
                    .join("、")}</div>`
            );
        }

        if (result.tags.failed.length) {
            lines.push(
                `<div class="bad">✕ 新增失敗：${result.tags.failed
                    .map(escapeHtml)
                    .join("<br>")}</div>`
            );
        }

        document.querySelector("#qt-status").innerHTML =
            lines.join("") || "沒有輸入內容";
    }

    async function apply() {
        const button = document.querySelector("#qt-apply");
        const status = document.querySelector("#qt-status");

        const categories = parseInput(
            document.querySelector("#qt-categories").value
        );

        const tags = parseInput(
            document.querySelector("#qt-tags").value
        );

        button.disabled = true;
        button.textContent = "處理中…";

        try {
            const categoryResult =
                processCategories(categories);

            const tagResult = await processTags(
                tags,
                message => {
                    status.textContent = message;
                }
            );

            renderResult({
                categories: categoryResult,
                tags: tagResult,
            });
        } catch (error) {
            status.innerHTML =
                `<div class="bad">${escapeHtml(error.message)}</div>`;
        } finally {
            button.disabled = false;
            button.textContent = "比對並加入";
        }
    }

    function createPanel() {
        const panel = document.createElement("div");

        panel.id = "qt-panel";
        panel.innerHTML = `
            <div class="title">
                <span>快速加入分類與標籤</span>
                <button type="button" id="qt-toggle">−</button>
            </div>

            <div id="qt-body">
                <label>分類</label>
                <textarea
                    id="qt-categories"
                    placeholder="產經新聞&#10;最新消息"
                ></textarea>

                <small>分類不存在時不會新增。</small>

                <label>標籤</label>
                <textarea
                    id="qt-tags"
                    placeholder="台股&#10;半導體&#10;AI概念股"
                ></textarea>

                <small>不存在的標籤會自動建立。</small>

                <button type="button" id="qt-apply">
                    比對並加入
                </button>

                <div id="qt-status">等待輸入。</div>
            </div>
        `;

        document.body.appendChild(panel);

        document
            .querySelector("#qt-apply")
            .addEventListener("click", apply);

        document
            .querySelector("#qt-toggle")
            .addEventListener("click", event => {
                const body = document.querySelector("#qt-body");
                const hidden = body.hidden;

                body.hidden = !hidden;
                event.target.textContent = hidden ? "−" : "＋";
            });
    }

    function addStyles() {
        const style = document.createElement("style");

        style.textContent = `
            #qt-panel {
                position: fixed;
                top: 90px;
                right: 24px;
                z-index: 999999;
                width: 330px;
                padding: 15px;
                box-sizing: border-box;
                background: white;
                border: 1px solid #ddd;
                border-radius: 10px;
                box-shadow: 0 8px 25px rgba(0,0,0,.18);
                font: 14px/1.5 "Microsoft JhengHei", sans-serif;
            }

            #qt-panel .title {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 17px;
                font-weight: bold;
            }

            #qt-panel label {
                display: block;
                margin-top: 12px;
                font-weight: bold;
            }

            #qt-panel textarea {
                width: 100%;
                min-height: 75px;
                padding: 8px;
                box-sizing: border-box;
                border: 1px solid #ccc;
                border-radius: 6px;
                resize: vertical;
            }

            #qt-tags {
                min-height: 120px !important;
            }

            #qt-panel small {
                color: #777;
            }

            #qt-apply {
                width: 100%;
                margin-top: 14px;
                padding: 9px;
                border: 0;
                border-radius: 6px;
                background: #2f6fed;
                color: white;
                font-weight: bold;
                cursor: pointer;
            }

            #qt-apply:disabled {
                opacity: .6;
                cursor: wait;
            }

            #qt-status {
                margin-top: 12px;
                padding: 9px;
                background: #f5f6f8;
                border-radius: 6px;
                word-break: break-word;
            }

            #qt-panel .ok {
                color: #157347;
            }

            #qt-panel .new {
                color: #087990;
            }

            #qt-panel .bad {
                color: #b42318;
            }

            #qt-toggle {
                border: 0;
                background: transparent;
                font-size: 20px;
                cursor: pointer;
            }

            @media (max-width: 768px) {
                #qt-panel {
                    top: auto;
                    right: 10px;
                    bottom: 10px;
                    left: 10px;
                    width: auto;
                    max-height: 70vh;
                    overflow-y: auto;
                }
            }
        `;

        document.head.appendChild(style);
    }

    function init(attempt = 0) {
        const categorySelect =
            document.querySelector("#category-select");

        const tagSelect =
            document.querySelector("#tag-select");

        if (!categorySelect || !tagSelect) {
            if (attempt < 30) {
                setTimeout(() => init(attempt + 1), 1000);
            }

            return;
        }

        if (document.querySelector("#qt-panel")) {
            return;
        }

        addStyles();
        createPanel();

        console.log("快速分類與標籤插件已載入");
    }

    document.readyState === "loading"
        ? document.addEventListener("DOMContentLoaded", () => init())
        : init();
})();