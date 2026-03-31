<div align="center">

# 🎬 OTT AI Inference Pipeline on NNIA  
### <i>From data to hardware to recommendation — one unified execution flow</i>

<br>

<p>
  This pipeline transforms <b>real user behavior into hardware-executed intelligence</b>.
  It integrates learning, fixed-point mapping, and NNIA execution to deliver
  <b>OTT-style recommendation output</b> through a single, structured system.
</p>

<br>

<img src="https://github.com/user-attachments/assets/95c04aac-b7a6-4b32-8033-5611988a27b9" width="90%">

</div>

<br>

## ⚡ What This Represents

This is the **execution backbone of NNIA**.

It connects:
- 🧠 model learning  
- 🔢 hardware mapping  
- ⚙️ NNIA execution  
- 🎬 final recommendation output  

Designed as a scalable architecture where performance improves with array size and clock frequency.

---

## 🎬 Real Dataset Integration

This pipeline is driven by **real-world recommendation data**, not synthetic inputs.

It uses the **MovieLens dataset** to model realistic OTT user behavior.

### What is used
- 🎞️ movie metadata (genres, titles)  
- ⭐ user ratings  
- 🕒 interaction patterns  

### How it flows through the system

📦 [`create_dataset.py`](./ott_recommender/create_dataset.py)  
&nbsp;&nbsp;&nbsp;&nbsp;
⬇️  
🧾 [`feature_encoder.py`](./ott_recommender/feature_encoder.py)  
&nbsp;&nbsp;&nbsp;&nbsp;
⬇️  
🧠 [`train_mlp.py`](./ott_recommender/train_mlp.py)

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

This system predicts whether a user will like a movie  
(<b>Recommended</b> / <b>Not Recommended</b>) based on past behavior.

### What NNIA demonstrates
- ♻️ hardware reuse across layers  
- 🔢 stable fixed-point execution  
- ⚙️ controlled layer scheduling  
- 🚀 scalable inference structure  

NNIA executes the same model that was trained in Python, demonstrating true <b>hardware–software alignment</b>.

💡 The same NNIA hardware is reused across multiple layers, proving scalability without architectural changes.

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

[`ott_runner.py`](./ott_recommender/ott_runner.py)

- drives the full pipeline  
- integrates Python and Vivado  
- manages execution and validation  

---

### 📦 Data Layer

- [`create_dataset.py`](./ott_recommender/create_dataset.py) → dataset generation  
- [`feature_encoder.py`](./ott_recommender/feature_encoder.py) → feature definition  

---

### 🧠 Model Layer

- [`train_mlp.py`](./ott_recommender/train_mlp.py) → model training  
- [`export_quantized_model.py`](./ott_recommender/export_quantized_model.py) → fixed-point conversion  

---

### 🧩 Hardware Prep

- [`prepare_layer1_mem.py`](./ott_recommender/prepare_layer1_mem.py) → Layer 1 memory  
- [`prepare_layer2_mem.py`](./ott_recommender/prepare_layer2_mem.py) → Layer 2 memory  
- [`mlp_inference_reference.py`](./ott_recommender/mlp_inference_reference.py) → reference path  

---

### ✅ Validation

- [`shared/compare_output.py`](./shared/compare_output.py) → RTL validation  
- [`shared/fixed_point_utils.py`](./shared/fixed_point_utils.py) → numeric alignment  

---

### 🎬 Output Layer

- [`mlp_output_analyzer.py`](./ott_recommender/mlp_output_analyzer.py) → recommendation output  

---

## ⚙️ Hardware–Software Co-Design

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
- enables verification  

---


<br>

<h2 align="center">📊 Results & Verification</h2>

<p align="center">
NNIA is validated through a hardware–software co-designed flow, ensuring 
<b>functional correctness</b>, <b>numerical alignment</b>, and <b>reliable inference execution</b>.
</p>

<br>

<div align="center">

<table>
<tr>

<td align="center" width="33%">

<b>🔬 Data Generation</b><br><br>
Python-based input, weight, and bias generation with fixed-point reference outputs<br><br>
<a href="results/python_results/nnia_data_generation_output.png">View Output</a>

</td>

<td align="center" width="33%">

<b>📈 Model Performance</b><br><br>
Training accuracy and evaluation metrics from the MLP model<br><br>
<a href="results/python_results/nnia_model_accuracy_summary.png">View Metrics</a>

</td>

<td align="center" width="33%">

<b>⚙️ Hardware Verification</b><br><br>
RTL outputs validated against Python golden reference<br><br>
<a href="results/python_results/nnia_rtl_vs_golden_comparison_result_pass.png">View Validation</a>

</td>

</tr>
</table>

</div>

<br>

<p align="center">
This confirms that NNIA achieves <b>cycle-accurate hardware inference</b> 
aligned with its software reference, demonstrating a robust and verifiable AI acceleration pipeline.
</p>
---

### 🔥 From model to hardware to recommendation — NNIA closes the loop with real multi-layer AI inference

</div>
