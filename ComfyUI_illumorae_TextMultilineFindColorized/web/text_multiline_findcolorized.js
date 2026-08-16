import { app } from "../../scripts/app.js";

const NODE_NAME = "illumoraeTextMultilineFindColorizedNode";

// ------------------------------------------------------------------
// Colors
// ------------------------------------------------------------------
const COLORS = {
    bg: "#1a1a1a",                 // dark dark grey for multiline editor
    text: "#cccccc",               // light grey for normal text
    bracket_tag: "#00a2ff",        // bright saturated blue for <lora:...> tags
    search_highlight: "#ff9800",   // orange for search matches
    search_current: "#ff6f00",     // darker orange for current match
    search_bar_bg: "#252525",
    search_bar_border: "#3a3a3a",
    search_bar_text: "#cccccc",
    search_bar_placeholder: "#6c7086",
    match_count_text: "#a6adc8",
    btn_bg: "#313244",
    btn_hover: "#45475a",
    btn_text: "#cccccc",
};

// ------------------------------------------------------------------
// HTML escape
// ------------------------------------------------------------------
function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

// ------------------------------------------------------------------
// Caret helpers (character-offset based for contentEditable)
// ------------------------------------------------------------------
function isSelectionInside(container) {
    const sel = window.getSelection();
    if (!sel.rangeCount) return false;
    const node = sel.anchorNode;
    if (!node) return false;
    return container === node || container.contains(node);
}

function getCaretOffset(container) {
    const sel = window.getSelection();
    if (!sel.rangeCount) return -1;
    const range = sel.getRangeAt(0).cloneRange();
    const preRange = document.createRange();
    preRange.selectNodeContents(container);
    preRange.setEnd(range.startContainer, range.startOffset);
    return preRange.toString().length;
}

function setCaretOffset(container, offset) {
    const sel = window.getSelection();
    const range = document.createRange();
    let charCount = 0;
    let found = false;

    function traverse(node) {
        if (found) return;
        if (node.nodeType === Node.TEXT_NODE) {
            const nextCount = charCount + node.textContent.length;
            if (offset <= nextCount) {
                range.setStart(node, Math.max(0, offset - charCount));
                range.collapse(true);
                found = true;
                return;
            }
            charCount = nextCount;
        } else {
            for (const child of node.childNodes) {
                traverse(child);
                if (found) return;
            }
        }
    }

    traverse(container);
    if (found) {
        sel.removeAllRanges();
        sel.addRange(range);
    }
}

// ------------------------------------------------------------------
// Highlighter: syntax coloring for <...> tags + search match highlighting
// ------------------------------------------------------------------
function highlightText(text, searchQuery, currentMatchIdx) {
    const n = text.length;

    // --- Phase 1: compute per-character style flags ---
    const isBracket = new Array(n).fill(false);
    const isSearch = new Array(n).fill(false);
    const isCurrent = new Array(n).fill(false);

    // Mark <...> bracket tag ranges
    let i = 0;
    while (i < n) {
        if (text[i] === "<") {
            const end = text.indexOf(">", i);
            if (end !== -1) {
                for (let j = i; j <= end; j++) isBracket[j] = true;
                i = end + 1;
                continue;
            }
        }
        i++;
    }

    // Mark search match ranges
    let totalMatches = 0;
    if (searchQuery) {
        const lowerText = text.toLowerCase();
        const lowerQuery = searchQuery.toLowerCase();
        let idx = 0;
        while (idx < n) {
            const found = lowerText.indexOf(lowerQuery, idx);
            if (found === -1) break;
            for (let j = found; j < found + searchQuery.length && j < n; j++) {
                isSearch[j] = true;
                if (totalMatches === currentMatchIdx) isCurrent[j] = true;
            }
            totalMatches++;
            idx = found + searchQuery.length;
        }
    }

    // --- Phase 2: build HTML by grouping same-style runs ---
    const parts = [];
    let j = 0;
    while (j < n) {
        const bracket = isBracket[j];
        const search = isSearch[j];
        const current = isCurrent[j];

        let k = j;
        while (
            k < n &&
            isBracket[k] === bracket &&
            isSearch[k] === search &&
            isCurrent[k] === current
        )
            k++;

        const segment = text.slice(j, k);
        let html = escapeHtml(segment);

        // Apply bracket color first (innermost)
        if (bracket) {
            html = `<span style="color:${COLORS.bracket_tag};">${html}</span>`;
        }

        // Apply search highlight (wraps around bracket color)
        if (current) {
            html = `<mark style="background:${COLORS.search_current};color:#fff;border-radius:2px;padding:0;">${html}</mark>`;
        } else if (search) {
            html = `<mark style="background:${COLORS.search_highlight};color:#000;border-radius:2px;padding:0;">${html}</mark>`;
        }

        parts.push(html);
        j = k;
    }

    return { html: parts.join(""), totalMatches };
}

// ------------------------------------------------------------------
// Build the editor with search bar
// ------------------------------------------------------------------
function buildEditor(initialValue, onChange) {
    // Root container
    const container = document.createElement("div");
    container.className = "illumorae-tmfc-container";
    container.style.cssText = "width:100%;position:relative;display:flex;flex-direction:column;flex:1 1 auto;min-height:160px;";

    // --- Search bar (hidden by default) ---
    const searchBar = document.createElement("div");
    searchBar.className = "illumorae-tmfc-searchbar";
    searchBar.style.cssText = [
        "display:none",
        "align-items:center",
        "gap:4px",
        "padding:4px 8px",
        "flex:0 0 auto",
        `background:${COLORS.search_bar_bg}`,
        `border:1px solid ${COLORS.search_bar_border}`,
        "border-radius:4px 4px 0 0",
        "box-sizing:border-box",
    ].join(";");

    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.placeholder = "Find...";
    searchInput.style.cssText = [
        "flex:1",
        "min-width:60px",
        "background:transparent",
        "border:none",
        "outline:none",
        `color:${COLORS.search_bar_text}`,
        `font-size:12px`,
        "font-family:inherit",
    ].join(";");

    const matchCount = document.createElement("span");
    matchCount.style.cssText = [
        "font-size:11px",
        `color:${COLORS.match_count_text}`,
        "white-space:nowrap",
        "min-width:40px",
        "text-align:center",
    ].join(";");

    const btnPrev = document.createElement("button");
    btnPrev.textContent = "^";
    btnPrev.title = "Previous match (Shift+Enter)";
    btnPrev.style.cssText = buttonStyle();

    const btnNext = document.createElement("button");
    btnNext.textContent = "v";
    btnNext.title = "Next match (Enter)";
    btnNext.style.cssText = buttonStyle();

    const btnClose = document.createElement("button");
    btnClose.textContent = "x";
    btnClose.title = "Close search (Esc)";
    btnClose.style.cssText = buttonStyle();

    searchBar.appendChild(searchInput);
    searchBar.appendChild(matchCount);
    searchBar.appendChild(btnPrev);
    searchBar.appendChild(btnNext);
    searchBar.appendChild(btnClose);
    container.appendChild(searchBar);

    // --- Editor (contentEditable) ---
    const editor = document.createElement("div");
    editor.className = "illumorae-tmfc-editor";
    editor.contentEditable = "true";
    editor.style.cssText = [
        "width:100%",
        "flex:1 1 auto",
        "min-height:120px",
        "overflow:auto",
        "font-family:'Consolas','Monaco','Courier New',monospace",
        "font-size:13px",
        "line-height:1.5",
        "padding:8px",
        "border-radius:4px",
        "white-space:pre-wrap",
        "word-break:break-word",
        "outline:none",
        "cursor:text",
        "user-select:text",
        "-webkit-user-select:text",
        `background:${COLORS.bg}`,
        `color:${COLORS.text}`,
    ].join(";");

    container.appendChild(editor);

    // --- State ---
    let searchQuery = "";
    let currentMatchIdx = 0;
    let totalMatches = 0;
    let isUpdating = false;

    function buttonStyle() {
        return [
            "background:transparent",
            "border:none",
            "cursor:pointer",
            `color:${COLORS.btn_text}`,
            "font-size:13px",
            "padding:2px 6px",
            "border-radius:3px",
            "line-height:1",
        ].join(";");
    }

    // --- Rendering ---
    function render() {
        const text = editor.innerText;
        // Only save/restore the caret when the editor itself is focused,
        // otherwise typing in the search input would steal focus into the editor.
        const editorFocused = isSelectionInside(editor);
        const offset = editorFocused ? getCaretOffset(editor) : -1;

        // Preserve scroll position across the innerHTML replacement.
        // Setting innerHTML resets scrollTop/scrollLeft to 0, and restoring
        // the caret via sel.addRange() triggers an auto-scroll-to-caret.
        // That auto-scroll can be asynchronous in Chrome, so a plain
        // scrollTop restore after addRange() gets overridden - the editor
        // would creep downward on every keystroke.
        //
        // Fix: temporarily switch the editor to overflow:hidden so the
        // browser physically cannot scroll it while addRange() runs, then
        // restore overflow and the saved scroll position. A rAF backup
        // catches any deferred scroll-to-caret queued by the browser.
        const savedScrollTop = editor.scrollTop;
        const savedScrollLeft = editor.scrollLeft;

        const result = highlightText(text, searchQuery, currentMatchIdx);
        totalMatches = result.totalMatches;

        isUpdating = true;
        const prevOverflow = editor.style.overflow;
        editor.style.overflow = "hidden";
        editor.innerHTML = result.html || "<br>";
        if (offset >= 0) setCaretOffset(editor, Math.min(offset, text.length));
        editor.style.overflow = prevOverflow;
        editor.scrollTop = savedScrollTop;
        editor.scrollLeft = savedScrollLeft;
        isUpdating = false;

        // Async safety net: Chrome sometimes queues the scroll-to-caret
        // from addRange() as a microtask/layout step that runs after the
        // synchronous restore above. Re-apply in the next frame.
        requestAnimationFrame(() => {
            editor.scrollTop = savedScrollTop;
            editor.scrollLeft = savedScrollLeft;
        });

        updateMatchCount();
    }

    function updateMatchCount() {
        if (!searchQuery) {
            matchCount.textContent = "";
            return;
        }
        if (totalMatches === 0) {
            matchCount.textContent = "0/0";
        } else {
            matchCount.textContent = `${currentMatchIdx + 1}/${totalMatches}`;
        }
    }

    function navigateMatch(direction) {
        if (totalMatches === 0) return;
        currentMatchIdx = (currentMatchIdx + direction + totalMatches) % totalMatches;
        render();
    }

    // --- Search bar controls ---
    function openSearch() {
        searchBar.style.display = "flex";
        searchInput.focus();
        searchInput.select();
    }

    function closeSearch() {
        searchBar.style.display = "none";
        searchQuery = "";
        currentMatchIdx = 0;
        searchInput.value = "";
        render();
        editor.focus();
    }

    searchInput.addEventListener("input", () => {
        searchQuery = searchInput.value;
        currentMatchIdx = 0;
        render();
    });

    searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            navigateMatch(e.shiftKey ? -1 : 1);
        } else if (e.key === "Escape") {
            e.preventDefault();
            closeSearch();
        }
    });

    btnPrev.addEventListener("click", () => navigateMatch(-1));
    btnNext.addEventListener("click", () => navigateMatch(1));
    btnClose.addEventListener("click", closeSearch);

    // --- Editor events ---
    editor.addEventListener("input", () => {
        if (isUpdating) return;
        render();
        onChange(editor.innerText);
    });

    // Ctrl+F to open search, Esc to close (when search bar is visible)
    editor.addEventListener("keydown", (e) => {
        // Stop Ctrl+V from bubbling to LiteGraph/ComfyUI canvas keydown
        // handlers that would paste workflow nodes. preventDefault is NOT
        // called here so the browser still fires the native paste event,
        // which the editor's paste listener below intercepts to insert
        // plain text. Also stop Ctrl+C/Ctrl+X for the same reason.
        if ((e.ctrlKey || e.metaKey) && (e.key === "v" || e.key === "c" || e.key === "x")) {
            e.stopPropagation();
            if (e.key !== "v") return; // let copy/cut fall through to default
        }
        if ((e.ctrlKey || e.metaKey) && e.key === "f") {
            e.preventDefault();
            openSearch();
            return;
        }
        if (e.key === "Escape" && searchBar.style.display !== "none") {
            e.preventDefault();
            closeSearch();
            return;
        }
        // Enter inserts a newline (not a <div> block)
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            document.execCommand("insertLineBreak");
        }
    });

    // Paste plain text only.
    // stopPropagation prevents the paste event from bubbling up to
    // ComfyUI's document-level paste handler, which checks whether the
    // event target is a native <textarea>/<input>. A contentEditable div
    // fails that check, so without stopPropagation ComfyUI falls through
    // to pasting workflow nodes instead of letting the text paste land.
    editor.addEventListener("paste", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const text = (e.clipboardData || window.clipboardData).getData("text/plain");
        document.execCommand("insertText", false, text);
    });

    // --- Initialize ---
    editor.textContent = initialValue || "";
    if (initialValue) render();

    // Expose methods for external use
    container._editor = editor;
    container._openSearch = openSearch;
    container._closeSearch = closeSearch;
    container._forceUpdate = render;
    container._setText = (text) => {
        editor.textContent = text || "";
        render();
    };

    // Dynamically fit the editor to a target pixel height (node height minus
    // the chrome above it). The container is a flex column, so the editor
    // (flex:1) fills whatever height is left after the search bar.
    container._fitHeight = (targetNodeHeight) => {
        // Offset accounts for: node title bar + "text" widget label + padding.
        const HEADER_OFFSET = 50;
        const h = Math.max(160, (targetNodeHeight || 300) - HEADER_OFFSET);
        container.style.height = h + "px";
    };

    return container;
}

// ------------------------------------------------------------------
// ComfyUI extension
// ------------------------------------------------------------------
app.registerExtension({
    name: "illumorae.TextMultilineFindColorized",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            requestAnimationFrame(() => {
                const textWidget = this.widgets?.find((w) => w.name === "text");
                if (!textWidget?.inputEl) return;

                const textarea = textWidget.inputEl;
                const parent = textarea.parentElement;
                if (!parent) return;

                // Hide the native textarea. ComfyUI re-renders widgets after
                // workflow execution and can reset these inline styles, so we
                // also install a MutationObserver below to re-apply them.
                const hideTextarea = () => {
                    textarea.style.display = "none";
                    textarea.style.height = "0px";
                    textarea.style.padding = "0px";
                    textarea.style.margin = "0px";
                    textarea.style.border = "none";
                    textarea.style.overflow = "hidden";
                    textarea.style.position = "absolute";
                    textarea.style.width = "1px";
                    textarea.style.opacity = "0";
                };
                hideTextarea();

                // Create our advanced editor
                const editor = buildEditor(
                    textarea.value,
                    (newText) => {
                        textarea.value = newText;
                        textWidget.value = newText;
                    }
                );

                // Insert before the hidden textarea
                parent.insertBefore(editor, textarea);

                // Keep the native textarea hidden even after ComfyUI re-renders
                // widgets (e.g. on workflow execution / completion).
                const styleObserver = new MutationObserver(() => {
                    if (textarea.style.display !== "none" ||
                        textarea.style.height !== "0px") {
                        hideTextarea();
                    }
                });
                styleObserver.observe(textarea, {
                    attributes: true,
                    attributeFilter: ["style", "class"],
                });

                // Re-sync editor content from the widget value and ensure our
                // custom editor is still visible/attached.
                const resyncEditor = () => {
                    hideTextarea();
                    if (editor.parentElement !== parent) {
                        parent.insertBefore(editor, textarea);
                    }
                    editor.style.display = "flex";
                    const txt = textWidget.value ?? textarea.value ?? "";
                    editor._setText(txt);
                    editor._fitHeight(this.size?.[1]);
                };

                // Store refs on the node
                this._tmfc_editor = editor;
                this._tmfc_textarea = textarea;
                this._tmfc_widget = textWidget;
                this._tmfc_styleObserver = styleObserver;
                this._tmfc_resyncEditor = resyncEditor;

                // Fit the editor to the node's current height, then keep it
                // in sync whenever the node is resized by the user.
                const fitToNode = () => editor._fitHeight(this.size?.[1]);
                fitToNode();

                const origOnResize = this.onResize;
                this.onResize = function (size) {
                    const rr = origOnResize ? origOnResize.apply(this, arguments) : undefined;
                    editor._fitHeight(size?.[1] ?? this.size?.[1]);
                    return rr;
                };
            });

            return r;
        };

        // Re-sync editor content when a workflow is loaded
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            requestAnimationFrame(() => {
                const textWidget = this.widgets?.find((w) => w.name === "text");
                if (textWidget && this._tmfc_editor) {
                    const txt = textWidget.value || "";
                    this._tmfc_editor._setText(txt);
                    this._tmfc_editor._fitHeight(this.size?.[1]);
                }
            });
            return r;
        };

        // After a workflow executes, ComfyUI re-renders widgets which can
        // re-show the native textarea and wipe our custom editor's state.
        // Re-apply hiding + re-sync content so colorization/search persist.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            requestAnimationFrame(() => {
                if (typeof this._tmfc_resyncEditor === "function") {
                    this._tmfc_resyncEditor();
                }
            });
            return r;
        };

        // Clean up the observer when the node is removed.
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this._tmfc_styleObserver) {
                this._tmfc_styleObserver.disconnect();
                this._tmfc_styleObserver = null;
            }
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});

console.log("[illumorae] TextMultilineFindColorized extension loaded");
