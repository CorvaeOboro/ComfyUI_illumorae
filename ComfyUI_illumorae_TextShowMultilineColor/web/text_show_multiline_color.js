import { app } from "../../scripts/app.js";

const NODE_NAME = "illumoraeTextShowMultilineColorNode";

app.registerExtension({
    name: "illumorae.TextShowMultilineColor",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Container for the colourised text
            const wrap = document.createElement("div");
            wrap.className = "illumorae-tshowmc-wrap";
            wrap.style.width = "100%";
            wrap.style.minHeight = "120px";
            wrap.style.maxHeight = "600px";
            wrap.style.overflow = "auto";
            wrap.style.borderRadius = "6px";
            wrap.style.boxSizing = "border-box";

            // The <pre> element holds the HTML from the backend.
            // user-select:text + cursor:text make it selectable & copyable.
            const pre = document.createElement("pre");
            pre.className = "illumorae-tshowmc-pre";
            pre.style.margin = "0";
            pre.style.padding = "12px";
            pre.style.fontFamily = "'Consolas','Monaco','Courier New',monospace";
            pre.style.fontSize = "13px";
            pre.style.lineHeight = "1.5";
            pre.style.whiteSpace = "pre-wrap";
            pre.style.wordBreak = "break-word";
            pre.style.userSelect = "text";
            pre.style.cursor = "text";
            pre.style.borderRadius = "6px";
            pre.textContent = "Waiting for execution...";

            wrap.appendChild(pre);

            const widget = this.addDOMWidget("multiline_color_text", "CUSTOMTEXT", wrap, {
                getValue() { return ""; },
                setValue(v) { },
            });

            // Tell ComfyUI how much vertical space to reserve.
            widget.computeSize = function (width) {
                const w = width || 300;
                return [w, 220];
            };

            this._tshowmc_pre = pre;
            this._tshowmc_widget = widget;

            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const pre = this._tshowmc_pre;
            if (!pre) return;

            if (message?.text?.[0]) {
                pre.innerHTML = message.text[0];
                // Dark vs light theme: extract background from the inline style
                const htmlStr = message.text[0];
                const bgMatch = htmlStr.match(/background:([^;]+)/);
                if (bgMatch) {
                    pre.style.background = bgMatch[1].trim();
                }
                const colorMatch = htmlStr.match(/color:([^;]+)/);
                if (colorMatch) {
                    // The first colour in the <pre> is the base text colour
                    pre.style.color = colorMatch[1].trim();
                }
            } else {
                // No output from the backend (should rarely happen because the
                // Python node always emits something, but we keep this safe).
                pre.textContent = "[No output]";
                pre.style.background = "#1e1e2e";
                pre.style.color = "#6c7086";
            }
        };
    },
});

console.log("[illumorae] TextShowMultilineColor extension loaded");
