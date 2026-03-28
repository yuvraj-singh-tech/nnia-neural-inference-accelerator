<div align="center">

# ⚙️ NNIA RTL Architecture  
### <i>Hybrid Tiled Systolic Design for Structured Neural Network Inference</i>

<br>

<p>
  A hardware implementation of NNIA that combines <b>systolic computation</b>,
  <b>explicit buffering</b>, and <b>tile-based control</b> to execute
  fixed-point neural inference in a structured and deterministic manner.
</p>

<br>

<img src="https://github.com/user-attachments/assets/1a97ac26-f121-4100-8d21-2f3a24ae9e41" width="80%">

</div>

---

## ⚡ What This Design Achieves

This RTL design forms the **core compute engine of NNIA**.

It enables:
- ⚙️ structured parallel computation  
- 🔢 fixed-point neural inference  
- 🧩 controlled data movement  
- 🔄 repeatable tile-based execution  

---

## 🧠 Architecture Overview

The design is built around three tightly integrated components:

- **Systolic compute fabric (PE array)**  
- **Explicit buffering stages**  
- **Tile-based execution control**

Together, they create a **coordinated inference pipeline** where data and compute are aligned.

---

## ⚡ Why Hybrid Architecture

NNIA combines two key ideas:

- **Data-driven computation (systolic array)**  
- **Control-driven execution (buffers + tiles)**  

---

### 🧠 Systolic Compute Fabric

👉 <a href="pe_array_4x4.v"><code>pe_array_4x4.v</code></a>

- activations flow across rows  
- weights flow across columns  
- MAC operations occur on alignment  
- partial sums accumulate locally  

This provides:
- parallel execution  
- structured propagation  
- efficient data reuse  

---

### 🗂️ Buffer-Controlled Dataflow

- <a href="input_buffer.v"><code>input_buffer.v</code></a> → activation staging  
- <a href="weight_buffer.v"><code>weight_buffer.v</code></a> → weight staging  
- <a href="psum_buffer.v"><code>psum_buffer.v</code></a> → accumulation storage  
- <a href="output_buffer.v"><code>output_buffer.v</code></a> → output staging  

Buffers ensure:
- synchronized operand delivery  
- separation of memory and compute  
- deterministic execution timing  

---

### 🔗 Key Insight

> Computation flows through the array, while execution is precisely controlled around it.

This separation enables both:
- efficiency (systolic compute)  
- control (buffer + tile scheduling)  

---

## 🌊 Dataflow

- activations → horizontal propagation  
- weights → vertical propagation  
- aligned data → MAC execution  
- partial sums → accumulated across tiles  

Result: a **fully coordinated execution stream**.

---

## ⚙️ Core RTL Modules

### 🧩 Compute

- <a href="pe_unit.v"><code>pe_unit.v</code></a> → single processing element performing MAC and data forwarding  
- <a href="pe_array_4x4.v"><code>pe_array_4x4.v</code></a> → 4×4 systolic array enabling parallel MAC execution  
- <a href="mac_unit.v"><code>mac_unit.v</code></a> → fixed-point multiply–accumulate engine  

---

### 🧠 Post-Processing

- <a href="quant_bias_relu.v"><code>quant_bias_relu.v</code></a> → requantization, bias addition, and activation  
- <a href="relu_unit.v"><code>relu_unit.v</code></a> → standalone ReLU activation  
- <a href="postprocess_array.v"><code>postprocess_array.v</code></a> → parallel post-processing across output tile  

---

### 🗂️ Buffers

- <a href="input_buffer.v"><code>input_buffer.v</code></a> → schedules activations into the PE array  
- <a href="weight_buffer.v"><code>weight_buffer.v</code></a> → streams weights column-wise into compute  
- <a href="psum_buffer.v"><code>psum_buffer.v</code></a> → stores and restores partial sums across tiles  
- <a href="output_buffer.v"><code>output_buffer.v</code></a> → captures final outputs per tile  

---

### 🎛️ Control

- <a href="tile_controller.v"><code>tile_controller.v</code></a> → orchestrates tile execution and sequencing  
- <a href="tile_addr_gen.v"><code>tile_addr_gen.v</code></a> → generates tile traversal addresses  
- <a href="nnia_perf_counters.v"><code>nnia_perf_counters.v</code></a> → tracks execution events for performance observation  

---

### 🔗 Integration

- <a href="top_nnia.v"><code>top_nnia.v</code></a> → integrates all modules into the full inference pipeline  

---

## 🔄 Execution Pipeline

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
final output  

---

## 🧪 Verification Flow

RTL correctness is validated against a Python-based golden reference.

### Flow

Python → Golden Model → `.mem` → RTL → Compare → PASS  

👉 <a href="../python/shared/compare_output.py"><b>View comparison logic</b></a>

---

### 🧠 Verification Modules

- <a href="../python/shared/fixed_point_utils.py"><code>shared/fixed_point_utils.py</code></a> → fixed-point conversion utilities  
- <a href="../python/cores/generate_data.py"><code>cores/generate_data.py</code></a> → input and reference data generation  
- <a href="../python/cores/tile_golden_model.py"><code>cores/tile_golden_model.py</code></a> → tile-aware golden inference model  
- <a href="../python/shared/compare_output.py"><code>shared/compare_output.py</code></a> → RTL vs reference output comparison  
---

## 📊 Validation

### 🏗️ Hardware Analysis

- 🏗️ <a href="../results/vivado_results/nnia_rtl_architecture.png"><b>RTL Architecture</b></a>  
- 🌊 <a href="../results/vivado_results/nnia_full_rtl_waveform.png"><b>Waveform</b></a>  
- 📦 <a href="../results/vivado_results/nnia_resource_utilization.png"><b>Utilization</b></a>  
- ⏱️ <a href="../results/vivado_results/nnia_timing_summary.png"><b>Timing</b></a>  
- ⚡ <a href="mac_unit.v"><b>MAC Engine</b></a>  

---

### 🧪 Functional Verification

- ✅ <a href="../results/python_results/nnia_rtl_vs_golden_comparison_result_pass.png"><b>RTL vs Golden (PASS)</b></a>  

---

### 🔗 Full Results

📁 <a href="../results/"><b>View all results</b></a>  

---

## ⚙️ Implementation Context

- Synthesized on Artix-7 FPGA (Vivado 2022.1) with a 100 MHz timing constraint  
- Achieved positive slack of +3.7 ns, indicating timing closure with headroom (~150+ MHz estimated)
  
---

### ✨ Hybrid systolic compute with controlled dataflow — enabling structured, scalable neural inference in hardware

</div>
