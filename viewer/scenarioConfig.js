/**
 * Scenario Configuration UI for CosmoSim Web Viewer
 * 
 * Provides dynamic parameter editing and CLI command generation
 * based on PSS (Parameterized Scenario System) schemas.
 */

// Import generated scenario metadata
import { ScenarioSchemas } from "./generated_scenario_metadata.js";

/**
 * ScenarioConfigUI Class
 * 
 * Dynamically generates parameter input forms and CLI commands.
 */
export class ScenarioConfigUI {
    constructor(rootElement) {
        this.root = rootElement;
        this.schemas = ScenarioSchemas;

        // DOM References
        this.selector = document.getElementById("scenarioSelector");
        this.presetSelector = document.getElementById("presetSelector");
        this.paramsContainer = document.getElementById("paramsContainer");
        this.stepsInput = document.getElementById("stepsInput");
        this.outputDirInput = document.getElementById("outputDirInput");
        this.commandOutput = document.getElementById("scenarioCommandOutput");
        this.commandContainer = document.getElementById("scenarioCommandContainer");
        this.copyBtn = document.getElementById("copyScenarioCommandBtn");

        // State
        this.currentParams = {};
        this.currentScenario = null;

        // Initialize
        this.populateScenarios();

        // Event Listeners
        this.selector.onchange = () => this.onScenarioChange();
        if (this.presetSelector) {
            this.presetSelector.onchange = () => this.onPresetChange();
        }

        // Live command update on any input change
        this.root.addEventListener("input", () => this.updateCommandPreview());
        this.root.addEventListener("change", () => this.updateCommandPreview());

        if (this.copyBtn) {
            this.copyBtn.onclick = () => this.copyCommand();
        }
    }

    /**
     * Populate scenario dropdown with available scenarios
     */
    populateScenarios() {
        const names = Object.keys(this.schemas).sort();

        // Clear existing options
        this.selector.innerHTML = "";

        for (const name of names) {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            this.selector.appendChild(opt);
        }

        if (names.length > 0) {
            this.selector.value = names[0];
            this.onScenarioChange();
        }
    }

    /**
     * Handle scenario selection change
     */
    onScenarioChange() {
        this.currentScenario = this.selector.value;
        this.populatePresets();
        this.renderParams();
        this.updateCommandPreview();
    }

    /**
     * Populate preset dropdown for selected scenario
     */
    populatePresets() {
        if (!this.presetSelector) return;

        const scenario = this.schemas[this.currentScenario];
        const presets = scenario?.presets || {};
        const presetNames = Object.keys(presets);

        // Clear and add "None" option
        this.presetSelector.innerHTML = "";

        const noneOpt = document.createElement("option");
        noneOpt.value = "";
        noneOpt.textContent = "(None)";
        this.presetSelector.appendChild(noneOpt);

        for (const name of presetNames) {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            this.presetSelector.appendChild(opt);
        }
    }

    /**
     * Handle preset selection - autofill parameter fields
     */
    onPresetChange() {
        const presetName = this.presetSelector.value;
        if (!presetName) return;

        const scenario = this.schemas[this.currentScenario];
        const presetValues = scenario?.presets?.[presetName] || {};

        // Apply preset values to form fields
        for (const [key, value] of Object.entries(presetValues)) {
            const input = this.paramsContainer.querySelector(`[data-param="${key}"]`);
            if (input) {
                if (input.type === "checkbox") {
                    input.checked = !!value;
                } else {
                    input.value = value;
                }
                this.currentParams[key] = value;
            }
        }

        this.updateCommandPreview();
    }

    /**
     * Render parameter input fields for selected scenario
     */
    renderParams() {
        const scenario = this.schemas[this.currentScenario];
        const params = scenario?.params || {};

        // Clear previous
        this.paramsContainer.innerHTML = "";
        this.currentParams = {};

        if (Object.keys(params).length === 0) {
            const msg = document.createElement("div");
            msg.textContent = "No editable parameters for this scenario.";
            msg.style.color = "#aaa";
            msg.style.fontSize = "12px";
            msg.style.marginTop = "10px";
            this.paramsContainer.appendChild(msg);
            return;
        }

        // Generate input fields
        for (const [key, spec] of Object.entries(params)) {
            const wrapper = document.createElement("div");
            wrapper.style.marginBottom = "12px";

            const label = document.createElement("label");
            label.textContent = key;
            label.style.display = "block";
            label.style.marginBottom = "4px";
            label.style.fontSize = "13px";
            label.style.color = "#fff";
            wrapper.appendChild(label);

            // Description tooltip
            if (spec.description) {
                const desc = document.createElement("div");
                desc.textContent = spec.description;
                desc.style.fontSize = "10px";
                desc.style.color = "#aaa";
                desc.style.marginBottom = "4px";
                wrapper.appendChild(desc);
            }

            // Input field
            let input;

            if (spec.type === "bool") {
                input = document.createElement("input");
                input.type = "checkbox";
                input.checked = !!spec.default;
                input.style.transform = "scale(1.2)";
            } else if (spec.allowed && Array.isArray(spec.allowed)) {
                // Dropdown for allowed values
                input = document.createElement("select");
                input.style.width = "100%";
                input.style.padding = "6px";
                input.style.borderRadius = "4px";
                input.style.border = "1px solid #555";
                input.style.background = "#222";
                input.style.color = "#fff";

                for (const val of spec.allowed) {
                    const opt = document.createElement("option");
                    opt.value = val;
                    opt.textContent = val;
                    if (val === spec.default) opt.selected = true;
                    input.appendChild(opt);
                }
            } else if (spec.type === "str") {
                input = document.createElement("input");
                input.type = "text";
                input.value = spec.default || "";
                input.style.width = "100%";
                input.style.padding = "6px";
                input.style.borderRadius = "4px";
                input.style.border = "1px solid #555";
                input.style.background = "#222";
                input.style.color = "#fff";
            } else {
                // Numeric input (int or float)
                input = document.createElement("input");
                input.type = "number";
                input.value = spec.default;
                input.style.width = "100%";
                input.style.padding = "6px";
                input.style.borderRadius = "4px";
                input.style.border = "1px solid #555";
                input.style.background = "#222";
                input.style.color = "#fff";

                if (spec.type === "int") {
                    input.step = "1";
                } else {
                    input.step = "any";
                }
                if (spec.min !== undefined) {
                    input.min = spec.min;
                }
                if (spec.max !== undefined) {
                    input.max = spec.max;
                }
            }

            // Data attribute for lookup
            input.dataset.param = key;

            // Update handler
            input.onchange = () => {
                this.currentParams[key] = this.readParamValue(spec, input);
                this.updateCommandPreview();
            };
            input.oninput = () => {
                this.currentParams[key] = this.readParamValue(spec, input);
                this.updateCommandPreview();
            };

            // Initialize current params
            this.currentParams[key] = spec.default;

            wrapper.appendChild(input);
            this.paramsContainer.appendChild(wrapper);
        }
    }

    /**
     * Read typed value from input element
     */
    readParamValue(spec, input) {
        if (spec.type === "bool") {
            return input.checked;
        } else if (spec.type === "int") {
            return parseInt(input.value, 10);
        } else if (spec.type === "float") {
            return parseFloat(input.value);
        }
        return input.value;
    }

    /**
     * Update CLI command preview (live)
     */
    updateCommandPreview() {
        const scenario = this.currentScenario;
        const preset = this.presetSelector?.value || "";
        const steps = this.stepsInput?.value || 300;
        const outputDir = this.outputDirInput?.value || "outputs/demo";

        // Build params string
        const pieces = [];
        for (const [key, value] of Object.entries(this.currentParams)) {
            pieces.push(`${key}=${value}`);
        }
        const paramStr = pieces.join(",");

        // Build command
        let cmd = `python cosmosim.py --scenario ${scenario}`;

        if (preset) {
            cmd += ` --preset ${preset}`;
        }

        if (paramStr) {
            cmd += ` --params "${paramStr}"`;
        }

        cmd += ` --steps ${steps}`;
        cmd += ` --view web`;
        cmd += ` --output-dir ${outputDir}`;

        if (this.commandOutput) {
            this.commandOutput.value = cmd;
        }
        if (this.commandContainer) {
            this.commandContainer.style.display = "block";
        }
    }

    /**
     * Copy command to clipboard
     */
    copyCommand() {
        if (!this.commandOutput) return;

        this.commandOutput.select();
        this.commandOutput.setSelectionRange(0, 99999);

        if (navigator.clipboard) {
            navigator.clipboard.writeText(this.commandOutput.value)
                .then(() => {
                    this.copyBtn.textContent = "✓ Copied!";
                    setTimeout(() => {
                        this.copyBtn.textContent = "Copy to Clipboard";
                    }, 2000);
                })
                .catch(() => {
                    document.execCommand("copy");
                });
        } else {
            document.execCommand("copy");
        }
    }
}

// Re-export for backward compatibility
export { ScenarioSchemas };
