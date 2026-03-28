<div align="center">

# 🎬 OTT AI Inference Pipeline on NNIA  
### <i>From data to hardware to recommendation — one unified execution flow</i>

<br>

<p>
  This pipeline delivers a complete path from <b>real user behavior to hardware-executed intelligence</b>.
  It transforms learned models into <b>fixed-point NNIA execution</b> and produces
  <b>OTT-style recommendations</b> through a tightly integrated system.
</p>

<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/95c04aac-b7a6-4b32-8033-5611988a27b9" width="90%">
</div>

</div>

---

## ⚡ What This Represents

This is the **execution backbone of NNIA**.

It connects:
- 🧠 model learning  
- 🔢 hardware mapping  
- ⚙️ NNIA execution  
- 🎬 final recommendation output  

---

## 🎬 Real Dataset Integration

This pipeline is driven by **real-world recommendation data**, not synthetic inputs.

It uses the **MovieLens dataset** to model realistic OTT user behavior.

### What is used
- 🎞️ movie metadata (genres, titles)  
- ⭐ user ratings  
- 🕒 interaction patterns  

### How it flows through the system

- `create_dataset.py`  
  → converts MovieLens data into structured samples  

- `feature_encoder.py`  
  → maps behavior into a **fixed 16-feature vector**  

- `train_mlp.py`  
  → learns recommendation patterns from real interactions  

### Why this matters

- 🧠 captures real user preferences  
- 🎯 produces meaningful recommendations  
- 🔗 connects NNIA to a real application  
- 💡 demonstrates hardware inference on real data  

---

## 🧠 Multi-Layer Inference on NNIA

This pipeline runs **sequential multi-layer inference** on the same hardware.

### Why multi-layer
- real models require staged decision-making  
- deeper layers refine predictions  
- improves accuracy and confidence  

### How it works
- Layer 1 → extracts hidden features  
- Layer 2 → generates final output  
- same NNIA hardware executes both  

### What NNIA demonstrates
- ♻️ hardware reuse across layers  
- 🔢 stable fixed-point execution  
- ⚙️ controlled layer scheduling  
- 🚀 scalable inference structure  

---

## 🚀 Execution Flow

<div align="center">

| Step | Stage |
|------|------|
| 1 | 📦 Dataset build |
| 2 | 🧾 Feature validation |
| 3 | 🧠 Model training |
| 4 | 🔢 Quantization |
| 5 | 🧩 Layer 1 setup |
| 6 | ⚙️ NNIA run (L1) |
| 7 | ✅ Validation (L1) |
| 8 | 🧪 Reference run |
| 9 | 🧩 Layer 2 setup |
| 10 | ⚙️ NNIA run (L2) |
| 11 | ✅ Validation (L2) |
| 12 | 🎬 Final output |

</div>

---

## 🧠 Core Blocks

### 🎛️ Controller

#### [`ott_runner.py`](./ott_runner.py)

Drives the entire pipeline.

- runs all stages  
- integrates Python + Vivado  
- manages logs and checks  
- reports final status  

---

### 📦 Data Layer

- [`create_dataset.py`](./create_dataset.py) → dataset generation  
- [`feature_encoder.py`](./feature_encoder.py) → feature definition  

---

### 🧠 Model Layer

- [`train_mlp.py`](./train_mlp.py) → model training  
- [`export_quantized_model.py`](./export_quantized_model.py) → fixed-point conversion  

---

### 🧩 Hardware Prep

- [`prepare_layer1_mem.py`](./prepare_layer1_mem.py) → Layer 1 memory  
- [`prepare_layer2_mem.py`](./prepare_layer2_mem.py) → Layer 2 memory  
- [`mlp_inference_reference.py`](./mlp_inference_reference.py) → reference path  

---

### ✅ Validation

- [`shared/compare_output.py`](./shared/compare_output.py) → RTL validation  
- [`shared/fixed_point_utils.py`](./shared/fixed_point_utils.py) → numeric alignment  

---

### 🎬 Output Layer

- [`mlp_output_analyzer.py`](./mlp_output_analyzer.py) → recommendation output  

---

## 🔗 Code Access

<div align="center">

| Module | Link |
|--------|------|
| Runner | [`ott_runner.py`](./ott_runner.py) |
| Dataset | [`create_dataset.py`](./create_dataset.py) |
| Encoder | [`feature_encoder.py`](./feature_encoder.py) |
| Training | [`train_mlp.py`](./train_mlp.py) |
| Quantization | [`export_quantized_model.py`](./export_quantized_model.py) |
| Layer 1 | [`prepare_layer1_mem.py`](./prepare_layer1_mem.py) |
| Reference | [`mlp_inference_reference.py`](./mlp_inference_reference.py) |
| Layer 2 | [`prepare_layer2_mem.py`](./prepare_layer2_mem.py) |
| Analyzer | [`mlp_output_analyzer.py`](./mlp_output_analyzer.py) |
| Compare | [`shared/compare_output.py`](./shared/compare_output.py) |
| Fixed-Point | [`shared/fixed_point_utils.py`](./shared/fixed_point_utils.py) |

</div>

---

## 📸 Results

<div align="center">

| Output | Link |
|--------|------|
| Performance | [`nnia_inference_performance_summary.png`](../results/python_results/nnia_inference_performance_summary.png) |
| Accuracy | [`nnia_model_accuracy_summary.png`](../results/python_results/nnia_model_accuracy_summary.png) |
| Quantization | [`nnia_model_quantization_report.png`](../results/python_results/nnia_model_quantization_report.png) |
| Recommendation | [`nnia_watchly_strong_match_recommendation.png`](../results/python_results/nnia_watchly_strong_match_recommendation.png) |
| Rejection Case | [`nnia_watchly_low_match_rejection.png`](../results/python_results/nnia_watchly_low_match_rejection.png) |

</div>

---

## ⚙️ System Split

**Software**
- prepares data  
- builds model  
- generates memory  
- validates outputs  

**NNIA Hardware**
- executes inference  
- processes layers  
- produces outputs  

**Simulation**
- runs via Vivado  
- captures performance  
- enables verification  

---

## 📈 Run Insights

- ⏱️ latency  
- 🚀 throughput  
- ✅ layer validation  
- 🏁 final status  

---

## 🎯 Final Take

A complete **multi-layer neural inference system executed on hardware**,  
driven by real data and delivering real output.

<div align="center">

### 🔥 NNIA executes structured, multi-stage AI inference with real application output

</div>
