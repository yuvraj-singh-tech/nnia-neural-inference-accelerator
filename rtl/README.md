<h1 align="center"><b>⚙️ NNIA RTL Design</b></h1>

<h3 align="center"><i>Hybrid Tiled Systolic Architecture for Structured Neural Network Inference</i></h3>

<p align="center"><i>Parallel Compute • Controlled Dataflow • Tile-Based Execution • Deterministic Verification</i></p>

---

## 🧠 <i>RTL Overview</i>

The RTL implementation of NNIA realizes a structured hardware pipeline for neural network inference using fixed-point arithmetic.

The design integrates compute, buffering, and control into a coordinated execution system, where data movement and computation are tightly aligned. Rather than relying on a single architectural style, NNIA combines multiple execution mechanisms to achieve both efficiency and controllability.

---

## 🏗️ <i>Architecture Overview</i>

<p align="center">
  <img src="https://github.com/user-attachments/assets/1a97ac26-f121-4100-8d21-2f3a24ae9e41" width="77%">
</p>

The architecture is organized around three tightly coupled components:

- **Systolic compute fabric (PE array)**  
- **Explicit buffering stages**  
- **Tile-based execution control**

These together form a structured and repeatable inference pipeline.

---

## ⚡<i>Why Hybrid Architecture?</i>

NNIA is termed **hybrid** because it combines:

- **Systolic array-based computation**
- **Explicit buffer-controlled data movement**

---

### 🧠 Systolic Compute Layer

Computation is performed inside  
👉 <a href="pe_array_4x4.v"><code>pe_array_4x4.v</code></a>  

- Data propagates across the PE array in a structured manner  
- Each PE performs localized MAC operations  
- Partial sums accumulate across propagation steps  
- Execution is driven by data alignment rather than centralized control  

This enables:
- efficient parallel computation  
- structured propagation of results  
- reuse of intermediate data  

---

### 🗂️ Buffer-Controlled Data Movement

Data is explicitly staged before entering compute:

- <a href="input_buffer.v"><code>input_buffer.v</code></a> → activation scheduling  
- <a href="weight_buffer.v"><code>weight_buffer.v</code></a> → weight scheduling  
- <a href="psum_buffer.v"><code>psum_buffer.v</code></a> → accumulation persistence  
- <a href="output_buffer.v"><code>output_buffer.v</code></a> → output staging  

Buffers ensure:
- controlled data injection  
- synchronization of operands  
- separation of memory and compute  

---

### 🔗 Why This Combination Matters

The key idea is not just using a systolic array —  
it is **controlling when and how data enters the array**.

- The systolic array defines **how computation propagates**  
- The buffers and control logic define **when computation happens**  

👉 This results in:

- predictable execution timing  
- structured and synchronized computation  
- scalable tile-based execution  

---

### ⚙️ Final Insight

> **Computation is data-driven (systolic), while execution is control-driven (buffers + tiles).**

This separation enables both high efficiency and controlled execution without tightly coupling memory and compute.

---

## 🌊 <i>Dataflow and Execution</i>

- Activations move across rows  
- Weights move across columns  
- MAC operations occur when aligned  
- Partial sums accumulate across tile steps  

This creates a coordinated execution flow rather than independent operations.

---

## ⚙️ <i>Core RTL Modules</i>

### 🧩 Compute Fabric

- <a href="pe_unit.v"><code>pe_unit.v</code></a>  
- <a href="pe_array_4x4.v"><code>pe_array_4x4.v</code></a>  
- <a href="mac_unit.v"><code>mac_unit.v</code></a>  

---

### 🧠 Post-Processing

- <a href="quant_bias_relu.v"><code>quant_bias_relu.v</code></a>  
- <a href="relu_unit.v"><code>relu_unit.v</code></a>  
- <a href="postprocess_array.v"><code>postprocess_array.v</code></a>  

---

### 🗂️ Buffers

- <a href="input_buffer.v"><code>input_buffer.v</code></a>  
- <a href="weight_buffer.v"><code>weight_buffer.v</code></a>  
- <a href="psum_buffer.v"><code>psum_buffer.v</code></a>  
- <a href="output_buffer.v"><code>output_buffer.v</code></a>  

---

### 🎛️ Control

- <a href="tile_controller.v"><code>tile_controller.v</code></a>  
- <a href="tile_addr_gen.v"><code>tile_addr_gen.v</code></a>  
- <a href="nnia_perf_counters.v"><code>nnia_perf_counters.v</code></a>  

---

### 🔗 Integration

- <a href="top_nnia.v"><code>top_nnia.v</code></a> — full pipeline  

---

## 🔄 <i>Execution Pipeline</i>

Input (.mem)  
↓  
input_buffer  
↓  
weight_buffer  
↓  
pe_array_4x4  
↓  
psum_buffer  
↓  
postprocess_array  
↓  
output_buffer  
↓  
final output assembly  

---



## 🧪 <i>Verification & Golden Model Alignment</i>

NNIA RTL is verified against a Python-based golden reference model to ensure functional correctness across the complete inference pipeline.

The verification flow uses deterministic input generation, tile-aware reference computation, and strict output comparison between software and hardware results.

---

### 🧠 <i>Python Verification Modules</i>

- <a href="../python/shared/fixed_point_utils.py"><code>shared/fixed_point_utils.py</code></a>  
- <a href="../python/cores/generate_data.py"><code>cores/generate_data.py</code></a>  
- <a href="../python/cores/tile_golden_model.py"><code>cores/tile_golden_model.py</code></a>  
- <a href="../python/shared/compare_output.py"><code>shared/compare_output.py</code></a>  

---

### 🔄 <i>Verification Execution Flow</i>

Python → Golden Model → .mem → RTL Simulation → Compare → PASS  

👉 <a href="../python/shared/compare_output.py"><b>View Comparison Logic</b></a>  

---

## 📊 <i>Validation & Results</i>

The following artifacts validate both the functional correctness and hardware behavior of the NNIA RTL pipeline after verification against the Python golden model.

---

### 🧠 <i>Hardware & RTL Analysis</i>

- 🏗️ <a href="../results/vivado_results/nnia_rtl_architecture.png"><b>RTL Architecture View</b></a>  
- 🌊 <a href="../results/vivado_results/nnia_full_rtl_waveform.png"><b>Simulation Waveform</b></a>  
- 📦 <a href="../results/vivado_results/nnia_resource_utilization.png"><b>Resource Utilization</b></a>  
- ⏱️ <a href="../results/vivado_results/nnia_timing_summary.png"><b>Timing Summary</b></a>  
- ⚡ <a href="mac_unit.v"><b>MAC Compute Engine (Fixed-Point Multiply–Accumulate)</b></a>

---

### 🧪 <i>Functional Verification</i>

- ✅ <a href="../results/python_results/nnia_rtl_vs_golden_comparison_result_pass.png"><b>RTL vs Golden Model (PASS)</b></a>  

---

### 🔗 <i>Complete Results</i>

📁 <a href="../results/"><b>View Full Results Directory</b></a>  

---

## ✨ <i>Design Summary</i>

NNIA RTL implements a structured inference accelerator where:

- computation is localized within a systolic array  
- data movement is explicitly controlled via buffers  
- execution is coordinated through tile-based control  
- verification is aligned with a software reference  

This results in a **balanced, scalable, and deterministic hardware inference design**.
