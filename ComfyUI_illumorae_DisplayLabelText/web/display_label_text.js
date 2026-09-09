// illumoraeDisplayLabelText - a standalone frontend-only ComfyUI virtual node.
//
// Drops a floating text label anywhere on the graph. The node has no Python
// backend; it is registered entirely in the browser via LiteGraph. The text is
// the node Title; font size, family, color, alignment, background color,
// padding, border radius, and rotation angle are adjustable from the
// properties panel (double-click the node).
//
// Inspired by the rgthree-comfy "Label" node, reimplemented standalone with no
// dependency on the rgthree nodepack.
import { app } from "../../scripts/app.js";

const NODE_TYPE = "illumoraeDisplayLabelText";
const NODE_TITLE = "Display Label Text (illumorae)";
const CATEGORY = "illumorae";

// Module-level mouse state used to let clicks pass through pinned labels.
// We wrap LGraphCanvas.processMouseDown to capture the current event without
// depending on any external state manager.
let processingMouseDown = false;
let lastMouseEvent = null;

class DisplayLabelText extends LGraphNode {
    constructor(title = NODE_TITLE) {
        super(title);
        this.comfyClass = NODE_TYPE;
        this.isVirtualNode = true;
        this.resizable = false;
        this.serialize_widgets = true;

        // Default label properties. These show up in the properties panel.
        this.properties = this.properties || {};
        this.properties["fontSize"] = 12;
        this.properties["fontFamily"] = "Arial";
        this.properties["fontColor"] = "#ffffff";
        this.properties["textAlign"] = "left";
        this.properties["backgroundColor"] = "transparent";
        this.properties["padding"] = 0;
        this.properties["borderRadius"] = 0;
        this.properties["angle"] = 0;

        // Keep the node chrome fully transparent; the label draws itself.
        this.color = "#fff0";
        this.bgcolor = "#fff0";

        this.onConstructed();
    }

    draw(ctx) {
        this.flags = this.flags || {};
        // When pinned, let pointer interactions fall through to nodes below.
        this.flags.allow_interaction = !this.flags.pinned;

        ctx.save();
        // Some extensions reset these aggressively; clear them each draw.
        this.color = "#fff0";
        this.bgcolor = "#fff0";

        const fontColor = this.properties["fontColor"] || "#ffffff";
        const backgroundColor = this.properties["backgroundColor"] || "";
        const fontSize = Math.max(Number(this.properties["fontSize"]) || 0, 1);
        const fontFamily = this.properties["fontFamily"] ?? "Arial";
        const padding = Number(this.properties["padding"]) ?? 0;

        ctx.font = `${fontSize}px ${fontFamily}`;

        // Support literal "\n" sequences as newlines and trim trailing newlines.
        const processedTitle = (this.title ?? "").replace(/\\n/g, "\n").replace(/\n*$/, "");
        const lines = processedTitle.split("\n");

        const maxWidth = Math.max(...lines.map((s) => ctx.measureText(s).width));
        this.size[0] = maxWidth + padding * 2;
        this.size[1] = fontSize * lines.length + padding * 2;

        // Rotate around the center when an angle is set.
        const angleDeg = parseInt(String(this.properties["angle"] ?? 0), 10) || 0;
        if (angleDeg) {
            const cx = this.size[0] / 2;
            const cy = this.size[1] / 2;
            ctx.translate(cx, cy);
            ctx.rotate((angleDeg * Math.PI) / 180);
            ctx.translate(-cx, -cy);
        }

        // Optional rounded background.
        if (backgroundColor && backgroundColor !== "transparent") {
            const borderRadius = Number(this.properties["borderRadius"]) || 0;
            ctx.beginPath();
            if (typeof ctx.roundRect === "function") {
                ctx.roundRect(0, 0, this.size[0], this.size[1], [borderRadius]);
            } else {
                // Fallback for canvas runtimes without roundRect.
                ctx.rect(0, 0, this.size[0], this.size[1]);
            }
            ctx.fillStyle = backgroundColor;
            ctx.fill();
        }

        // Horizontal alignment maps to an x offset.
        ctx.textAlign = "left";
        let textX = padding;
        if (this.properties["textAlign"] === "center") {
            ctx.textAlign = "center";
            textX = this.size[0] / 2;
        } else if (this.properties["textAlign"] === "right") {
            ctx.textAlign = "right";
            textX = this.size[0] - padding;
        }
        ctx.textBaseline = "top";
        ctx.fillStyle = fontColor;

        let currentY = padding;
        for (let i = 0; i < lines.length; i++) {
            ctx.fillText(lines[i] || " ", textX, currentY);
            currentY += fontSize;
        }
        ctx.restore();
    }

    onDblClick(event, pos, canvas) {
        // Everything is editable from the properties panel, so open it.
        if (LGraphCanvas.active_canvas && LGraphCanvas.active_canvas.showShowNodePanel) {
            LGraphCanvas.active_canvas.showShowNodePanel(this);
        }
    }

    onShowCustomPanelInfo(panel) {
        // The Mode and Color rows are not meaningful for a floating label.
        panel.querySelector('div.property[data-property="Mode"]')?.remove();
        panel.querySelector('div.property[data-property="Color"]')?.remove();
    }

    inResizeCorner(x, y) {
        // There is both a resizable flag and this method that gates the icon.
        return this.resizable;
    }

    getHelp() {
        return `
      <p>
        The ${NODE_TITLE} node lets you add a floating text label to your workflow.
      </p>
      <p>
        The text shown is the "Title" of the node. Adjust the font size, font family,
        font color, text alignment, background color, padding, border radius, and
        rotation angle from the node's properties. Double-click the node to open the
        properties panel.
      </p>
      <ul>
        <li><p><strong>Tip #1:</strong> Add multiline text from the properties panel
          (ComfyUI allows shift+enter there only).</p></li>
        <li><p><strong>Tip #2:</strong> Use ComfyUI's native "pin" option in the
          right-click menu to make the label stick to the workflow and let clicks
          pass through. Right-click again to unpin.</p></li>
        <li><p><strong>Tip #3:</strong> Color values are hex strings, like "#FFFFFF"
          for white or "#660000" for dark red. A 7th/8th digit (or 5th in shorthand)
          adds translucency, e.g. "#FFFFFF88" is semi-transparent white.</p></li>
      </ul>`;
    }
}

// Static metadata used by LiteGraph and the properties panel widgets.
DisplayLabelText.type = NODE_TYPE;
DisplayLabelText.title = NODE_TITLE;
DisplayLabelText.title_mode = LiteGraph.NO_TITLE;
DisplayLabelText.collapsable = false;
DisplayLabelText.category = CATEGORY;
DisplayLabelText._category = CATEGORY;

// Property panel widget types so the properties panel renders the right editors.
DisplayLabelText["@fontSize"] = { type: "number" };
DisplayLabelText["@fontFamily"] = { type: "string" };
DisplayLabelText["@fontColor"] = { type: "string" };
DisplayLabelText["@textAlign"] = { type: "combo", values: ["left", "center", "right"] };
DisplayLabelText["@backgroundColor"] = { type: "string" };
DisplayLabelText["@padding"] = { type: "number" };
DisplayLabelText["@borderRadius"] = { type: "number" };
DisplayLabelText["@angle"] = { type: "number" };

// Override drawNode so that when a DisplayLabelText is drawn, the default node
// chrome is suppressed (kept transparent) and our custom draw runs instead.
const oldDrawNode = LGraphCanvas.prototype.drawNode;
LGraphCanvas.prototype.drawNode = function (node, ctx) {
    if (node instanceof DisplayLabelText) {
        node.bgcolor = "transparent";
        node.color = "transparent";
        const v = oldDrawNode.apply(this, arguments);
        node.draw(ctx);
        return v;
    }
    return oldDrawNode.apply(this, arguments);
};

// Track mousedown state so pinned labels can be click-through. We wrap
// processMouseDown to set a flag and capture the event, mirroring the rgthree
// approach but self-contained.
const oldProcessMouseDown = LGraphCanvas.prototype.processMouseDown;
LGraphCanvas.prototype.processMouseDown = function (e) {
    processingMouseDown = true;
    lastMouseEvent = e;
    try {
        return oldProcessMouseDown.apply(this, arguments);
    } finally {
        processingMouseDown = false;
    }
};

// Override getNodeOnPos so that during a left-button mousedown (that is not a
// double click) pinned labels are filtered out of hit testing, letting the
// click reach nodes underneath. Right-click and double-click still hit the
// label so it can be unpinned and edited.
const oldGetNodeOnPos = LGraph.prototype.getNodeOnPos;
LGraph.prototype.getNodeOnPos = function (x, y, nodes_list) {
    if (
        nodes_list &&
        processingMouseDown &&
        lastMouseEvent &&
        String(lastMouseEvent.type || "").includes("down") &&
        lastMouseEvent.which === 1
    ) {
        const isDoubleClick =
            LiteGraph.getTime() - LGraphCanvas.active_canvas.last_mouseclick < 300;
        if (!isDoubleClick) {
            nodes_list = [...nodes_list].filter(
                (n) => !(n instanceof DisplayLabelText) || !n.flags?.pinned,
            );
        }
    }
    return oldGetNodeOnPos.apply(this, [x, y, nodes_list]);
};

app.registerExtension({
    name: "illumorae.DisplayLabelText",
    registerCustomNodes() {
        LiteGraph.registerNodeType(NODE_TYPE, DisplayLabelText);
        DisplayLabelText.category = CATEGORY;
    },
});
