<div align="center">

# ⚡ NNIA – Neural Network Inference Accelerator  

<h3 align="center"><i>A Custom AI Inference Accelerator with a Hybrid Tiled Systolic Array Architecture</i></h3>

<p align="center"><i>Hardware–Software Co-Designed | RTL-Verified Against Python Golden Model | Multi-Layer MLP Inference for OTT Recommendation</i></p>

<br>

<img src="https://github.com/user-attachments/assets/919fe548-f83f-4118-948a-eca1c2c87eb0" width="80%">

</div>

<br>

## ✨ <i>Overview</i>

NNIA is a custom-designed AI inference accelerator that executes neural network inference directly in hardware through a **fully RTL-based pipeline** using fixed-point arithmetic.

It combines structured dataflow, parallel compute, and controlled execution to deliver **deterministic and scalable neural inference**.

This project implements a **complete hardware–software co-designed inference system**, spanning:

- 🧠 model preparation and training (Python)  
- 🔢 quantization and memory generation  
- ⚙️ RTL-based execution  
- 🧪 verification against a Python golden reference  

It supports **multi-layer neural network inference** and is demonstrated using an **OTT-style recommendation pipeline**.

This project demonstrates how neural network inference can be efficiently mapped from software models to deterministic hardware execution.

---

## 🧠 <i>Architecture Overview</i>

<p>
NNIA implements a <b>systolic-style dataflow architecture</b> — 
<i>a well-established compute paradigm widely used in accelerators such as Google's TPU</i> — 
adapted for efficient FPGA-based execution.
</p>

The design combines a systolic compute fabric with explicit buffering and tile-based control to form a **coordinated inference pipeline**.

- activations propagate left-to-right across the PE array  
- weights stream top-to-bottom across columns  
- computation occurs when activations and weights align within the array  
- partial sums accumulate across tile steps  

The architecture integrates:

- dedicated input, weight, PSUM, and output buffers  
- tile controller and address generator  
- structured systolic PE array  

This enables **efficient data reuse, synchronized execution, and scalable inference scheduling**.

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/1a97ac26-f121-4100-8d21-2f3a24ae9e41" width="80%">
</p>

<p align="center">
  🔗 <a href="./rtl/README.md"><b>Detailed RTL architecture and execution flow</b></a>
</p>

---

## ⚙️ <i>Key Design Features</i>

- custom RTL-based AI inference accelerator  
- hybrid tiled systolic array for parallel computation  
- 4×4 PE array with structured dataflow  
- explicit input, weight, PSUM, and output buffering  
- tile-based execution for controlled scheduling  
- multi-layer neural inference support  
- end-to-end hardware–software co-design  
- demonstrated using OTT-style recommendation pipeline  

---

## 🔄 <i>Execution Flow & Documentation</i>

NNIA is organized into two tightly connected flows:

<div align="center">

| Flow | Description |
|------|------------|
| 🧠 Python Flow | data, training, quantization, memory generation |
| ⚙️ RTL Flow | hardware execution, dataflow, buffering, control |

</div>

### 🔗 Documentation

- 📽️ <a href="./python/README.md"><b>OTT Recommendation Flow</b></a> 🍿  
- ⚙️ <a href="./rtl/README.md"><b>RTL Execution, Dataflow & Hardware Design</b></a> 🔧  

---

<br>

<h2 align="center">📊 Results & Verification</h2>

<p align="center">
Validated across software and hardware to ensure <b>correctness</b>, <b>alignment</b>, and <b>reliable inference execution</b>.
</p>

<br>

<div align="center">

<table>
<tr>

<td align="center" width="33%">

<b>📈 Model Performance</b><br><br>
<a href="results/python_results/nnia_model_accuracy_summary.png">Accuracy Metrics</a>

</td>

<td align="center" width="33%">

<b>🧪 RTL Verification</b><br><br>
<a href="results/python_results/nnia_rtl_vs_golden_comparison_result_pass.png">PASS Result</a>

</td>

<td align="center" width="33%">

<b>⚙️ Hardware Analysis</b><br><br>
<a href="results/vivado_results/nnia_resource_utilization.png">Utilization</a> · 
<a href="results/vivado_results/nnia_timing_summary.png">Timing</a>

</td>

</tr>
</table>

</div>

<br>

<p align="center">
Demonstrates <b>cycle-accurate hardware inference</b> aligned with the software reference.
</p>

---

## 📁 <i>Repository Structure</i>

```text
rtl/        → compute fabric, buffers, control, integration
python/     → model pipeline, quantization, OTT flow
tb/         → verification testbenches
scripts/    → automation and Vivado execution
results/    → outputs, reports, and validation artifacts
