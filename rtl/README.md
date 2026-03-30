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

<br>

## ⚡ What This Design Achieves

This RTL forms the **compute backbone of NNIA**, executing neural inference through a hybrid systolic–buffered architecture.

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

💡 Think of NNIA as a **systolic compute core wrapped with explicit data orchestration**.

---

## ⚡ Why Hybrid Architecture

NNIA combines two key ideas:

- **Data-driven computation (systolic array)**  
- **Control-driven execution (buffers + tiles)**  

---

### 🧠 Systolic Compute Fabric

[`pe_array_4x4.v`](./pe_array_4x4.v)

- activations flow across rows  
- weights flow across columns  
- MAC operations occur when activations and weights align in the array  
- partial sums accumulate locally  

This provides:
- parallel execution  
- structured propagation  
- efficient data reuse  

This design maximizes data reuse by streaming activations and weights once while reusing them across multiple MAC operations within the array.

---

### 🗂️ Buffer-Controlled Dataflow

- [`input_buffer.v`](./input_buffer.v) → activation staging  
- [`weight_buffer.v`](./weight_buffer.v) → weight staging  
- [`psum_buffer.v`](./psum_buffer.v) → accumulation storage  
- [`output_buffer.v`](./output_buffer.v) → output staging  

Buffers ensure:
- synchronized operand delivery  
- separation of memory and compute  
- deterministic execution timing  

The buffered, tile-driven execution ensures deterministic timing independent of data values.

---

### 🔗 Key Insight

> Computation flows through the array, while execution is precisely controlled around it.

This separation enables both:
- efficiency (systolic compute)  
- control (buffer + tile scheduling)  

---

## 🌊 Systolic Dataflow

- activations → horizontal propagation  
- weights → vertical propagation  
- aligned data → MAC execution  
- partial sums → accumulated across tiles  

Result: a **fully coordinated execution stream**.

---

## ⚙️ Core RTL Modules

### 🧩 Compute

- [`pe_unit.v`](./pe_unit.v) → single processing element performing MAC and data forwarding  
- [`pe_array_4x4.v`](./pe_array_4x4.v) → 4×4 systolic array enabling parallel MAC execution  
- [`mac_unit.v`](./mac_unit.v) → fixed-point multiply–accumulate engine  

---

### 🧠 Post-Processing

- [`quant_bias_relu.v`](./quant_bias_relu.v) → requantization, bias addition, and activation  
- [`relu_unit.v`](./relu_unit.v) → standalone ReLU activation  
- [`postprocess_array.v`](./postprocess_array.v) → parallel post-processing across output tile  

---

### 🗂️ Buffers

- [`input_buffer.v`](./input_buffer.v) → schedules activations into the PE array  
- [`weight_buffer.v`](./weight_buffer.v) → streams weights column-wise into compute  
- [`psum_buffer.v`](./psum_buffer.v) → stores and restores partial sums across tiles  
- [`output_buffer.v`](./output_buffer.v) → captures final outputs per tile  

---

### 🎛️ Control

- [`tile_controller.v`](./tile_controller.v) → orchestrates tile execution and sequencing  
- [`tile_addr_gen.v`](./tile_addr_gen.v) → generates tile traversal addresses  
- [`nnia_perf_counters.v`](./nnia_perf_counters.v) → tracks execution events for performance observation  

---

### 🔗 Integration

- [`top_nnia.v`](./top_nnia.v) → integrates all modules into the full inference pipeline  

---

## 🔄 Execution Pipeline

Input (.mem)  
↓  
input_buffer → weight_buffer  
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

This guarantees **bit-accurate alignment between software and hardware execution**.

👉 [`View comparison logic`](../python/shared/compare_output.py)

---

### 🧠 Verification Modules

- [`shared/fixed_point_utils.py`](../python/shared/fixed_point_utils.py) → fixed-point conversion utilities  
- [`cores/generate_data.py`](../python/cores/generate_data.py) → input and reference data generation  
- [`cores/tile_golden_model.py`](../python/cores/tile_golden_model.py) → tile-aware golden inference model  
- [`shared/compare_output.py`](../python/shared/compare_output.py) → RTL vs reference output comparison  

---

## 📊 Validation

### 🏗️ Hardware Analysis

- 🏗️ [`RTL Architecture`](../results/vivado_results/nnia_rtl_architecture.png)  
- 🌊 [`Waveform`](../results/vivado_results/nnia_full_rtl_waveform.png)  
- 📦 [`Utilization`](../results/vivado_results/nnia_resource_utilization.png)  
- ⏱️ [`Timing`](../results/vivado_results/nnia_timing_summary.png)  
- ⚡ [`MAC Engine`](./mac_unit.v)  

---

### 🧪 Functional Verification

- ✅ [`RTL vs Golden (PASS)`](../results/python_results/nnia_rtl_vs_golden_comparison_result_pass.png)  

---

### 🔗 Full Results

📁 [`View all results`](../results/)  

---

## ⚙️ Implementation Context

- Synthesized on Artix-7 FPGA (Vivado 2022.1) with a 100 MHz timing constraint  
- Achieved positive slack of +3.7 ns, indicating timing closure with headroom (~150+ MHz estimated)

---

This architecture demonstrates how structured hardware design can efficiently map neural workloads with predictable performance and scalable parallelism.

---

<div align="center">

### ✨ Hybrid systolic compute with controlled dataflow — enabling structured, scalable neural inference in hardware

</div>
