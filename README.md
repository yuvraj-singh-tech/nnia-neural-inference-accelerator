<h1 align="center"><b>NNIA – Neural Network Inference Accelerator</b></h1>

<h3 align="center"><i>A Custom AI Inference Accelerator with a Hybrid Tiled Systolic Array Architecture</i></h3>

<p align="center"><i>Hardware–Software Co-Designed | RTL-Verified Against Python Golden Model | Multi-Layer MLP Inference for OTT Recommendation</i></p>

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/919fe548-f83f-4118-948a-eca1c2c87eb0" alt="NNIA Hero Image" width="100%">
</p>


---

## ✨ <i>Overview</i>

NNIA is a custom-designed AI inference accelerator that executes neural network models using fixed-point arithmetic in a fully RTL-based hardware pipeline. The design adopts a hybrid tiled systolic array architecture to enable efficient parallel computation and structured data reuse.

The project spans the complete inference stack, including Python-based model preparation, quantization, memory generation, and RTL simulation, with outputs verified against a Python golden reference model. It supports structured multi-layer neural network inference and is demonstrated through an OTT-style movie recommendation pipeline.

---

## 🧠 <i>Architecture Overview</i>

NNIA combines a systolic array compute fabric with explicit memory buffering and tile-based control. Activations propagate left-to-right across the processing elements, while weights stream top-to-bottom, forming a structured systolic dataflow for dense inference workloads.

The architecture integrates dedicated input, weight, partial-sum (PSUM), and output buffers to improve data reuse and execution efficiency. A tile controller and address generator coordinate tiled execution across the compute array, enabling scalable and organized inference scheduling.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1a97ac26-f121-4100-8d21-2f3a24ae9e41" alt="NNIA Architecture" width="100%">
</p>

<p align="center">
  🔗 <a href="rtl/README.md"><b>Detailed RTL architecture and execution flow</b></a>
</p>

---

## ⚙️ <i>Key Design Features</i>

- Custom RTL-based AI inference accelerator design  
- Hybrid tiled systolic array architecture for parallel computation  
- 4×4 Processing Element (PE) array with structured dataflow  
- Dedicated input, weight, PSUM, and output buffering for efficient data reuse  
- Supports multi-layer neural network inference with structured layer-wise execution  
- End-to-end hardware–software co-design across Python and RTL domains  
- Demonstrated using an OTT-style movie recommendation pipeline  

---

## 🔄 <i>Execution Flow & Documentation</i>

The detailed execution flow, model preparation pipeline, quantization path, memory generation, and OTT inference setup are documented separately for clarity.

- 🐍 <a href="python/README.md"><b>Python Pipeline & OTT Recommendation Flow</b></a>  
- 🔧 <a href="rtl/README.md"><b>RTL Execution, Dataflow & Hardware Design</b></a>  

---

## 📊 <i>Results & Verification</i>

NNIA has been validated through both software-side and hardware-side verification. The repository includes model evaluation artifacts, RTL-vs-golden comparison outputs, and Vivado synthesis/timing summaries.

### 🔹 <i>Python Results</i>

- <a href="results/python_results/nnia_model_accuracy_summary.png"><b>Model Accuracy Summary</b></a>  
- <a href="results/python_results/nnia_inference_performance_summary.png"><b>Inference Performance Summary</b></a>  

### 🔹 <i>RTL Verification</i>

- <a href="results/python_results/nnia_rtl_vs_golden_comparison_result_pass.png"><b>RTL vs Golden Model Comparison</b></a>  

### 🔹 <i>Vivado Results</i>

- <a href="results/vivado_results/nnia_resource_utilization.png"><b>Resource Utilization Report</b></a>  
- <a href="results/vivado_results/nnia_timing_summary.png"><b>Timing Summary Report</b></a>  

<p>
  📁 <b>Full results folder:</b> <a href="results/"><code>results/</code></a>
</p>

---

## 📁 <i>Repository Structure</i>

```text
rtl/        → Hardware modules (PE array, buffers, controller, etc.)
python/     → Model pipeline, quantization, data generation, OTT flow
tb/         → Testbenches for RTL verification
scripts/    → Automation scripts (Vivado TCL, flow control)
results/    → Python and Vivado outputs, reports, and screenshots
