import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.illumoraeTextTokenCount",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "illumoraeTextTokenCountNode") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            this.size = [260, 200];
            this._lastTokenCount = undefined;

            setTimeout(() => {
                if (!this.widgets) return;
                const tw = this.widgets.find(w => w.name === "threshold");
                if (tw) {
                    const orig = tw.callback;
                    tw.callback = (v) => {
                        if (orig) orig.apply(this, [v]);
                        this.setDirtyCanvas(true, true);
                    };
                }
            }, 100);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);
            let val = message?.token_count;
            if (Array.isArray(val)) val = val[0];
            if (val !== undefined && val !== null) {
                this._lastTokenCount = String(val);
            }
            this.setDirtyCanvas(true, true);
        };

        nodeType.prototype.onDrawBackground = function(ctx) {
            if (this.flags.collapsed) return;

            const w = this.size[0];
            const h = this.size[1];

            ctx.save();
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(0, 0, w, h, 6);
            } else {
                ctx.rect(0, 0, w, h);
            }
            ctx.clip();

            // Dark grey node background
            ctx.fillStyle = "#333333";
            ctx.fillRect(0, 0, w, h);

            // Thick black border around the node
            ctx.strokeStyle = "#000000";
            ctx.lineWidth = 4;
            if (ctx.roundRect) {
                ctx.beginPath();
                ctx.roundRect(0, 0, w, h, 6);
                ctx.stroke();
            } else {
                ctx.strokeRect(0, 0, w, h);
            }

            ctx.restore();
        };

        nodeType.prototype.onDrawForeground = function(ctx) {
            if (this.flags.collapsed) return;

            const w = this.size[0];
            const h = this.size[1];

            // Get threshold value
            const thresholdWidget = this.widgets?.find(wgt => wgt.name === "threshold");
            const threshold = thresholdWidget ? thresholdWidget.value : 700;

            // Get token count string
            let displayValue = this._lastTokenCount;
            if (displayValue === undefined || displayValue === "") {
                displayValue = "--";
            }

            const tokenCount = parseInt(displayValue, 10);
            const exceeded = !isNaN(tokenCount) && tokenCount > threshold;

            const normalColor = "#dddddd";
            const thresholdColor = "#ff4444";
            const textColor = exceeded ? thresholdColor : normalColor;

            // Number display panel
            const margin = 14;
            const panelTop = 85;
            const panelBottomMargin = 30;
            const panelH = Math.max(20, h - panelTop - panelBottomMargin);
            const panelW = w - margin * 2;

            // Panel background
            ctx.fillStyle = "#2a2a2a";
            if (ctx.roundRect) {
                ctx.beginPath();
                ctx.roundRect(margin, panelTop, panelW, panelH, 4);
                ctx.fill();
            } else {
                ctx.fillRect(margin, panelTop, panelW, panelH);
            }

            // Panel border
            ctx.strokeStyle = "#000000";
            ctx.lineWidth = 3;
            if (ctx.roundRect) {
                ctx.beginPath();
                ctx.roundRect(margin, panelTop, panelW, panelH, 4);
                ctx.stroke();
            } else {
                ctx.strokeRect(margin, panelTop, panelW, panelH);
            }

            // Dynamic font size
            const fontSize = Math.min(panelW * 0.38, panelH * 0.6);
            const cx = w / 2;
            const cy = panelTop + panelH / 2 + fontSize * 0.35;

            // Draw numbers with outer border (stroke) for high visibility
            ctx.save();
            ctx.font = `bold ${fontSize}px monospace`;
            ctx.textAlign = "center";
            ctx.textBaseline = "alphabetic";
            ctx.lineJoin = "round";

            // Thick black outline around numbers
            ctx.strokeStyle = "#000000";
            ctx.lineWidth = Math.max(3, fontSize * 0.1);
            ctx.strokeText(displayValue, cx, cy);

            // Bold big numbers of light grey (or threshold color)
            ctx.fillStyle = textColor;
            ctx.fillText(displayValue, cx, cy);

            ctx.restore();
        };
    }
});
