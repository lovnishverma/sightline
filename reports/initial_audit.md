# SightLine: Initial Repository & Manuscript Audit Report
**Date:** October 15, 2026  
**Auditor:** Senior Research Engineer & Systems Specialist  
**Evaluated Artifacts:**
- Primary Repository 1: `C:\Users\princ\Desktop\SightLine` (Remote: `https://github.com/lovnishverma/sightline`, Commit: `f8136ae`)
- Primary Repository 2: `C:\Users\princ\Desktop\SIGHTLINE — BUILD WITH BHARAT 3.0` (Remote: `https://github.com/lovnishverma/SightLine2.O.git`, Commit: `0a19d1e`)
- Primary Manuscript: `c:\Users\princ\Desktop\Paper\sightline_paper.docx` / `sightline_paper_revised.docx`

---

## 1. Executive Summary

This audit assesses the scientific rigor, architectural implementation, experimental reproducibility, and manuscript claims of the **SightLine** project: *Priority-Weighted Scene-Change Gating for Redundancy-Aware Assistive Vision-Language Models*.

The core thesis of SightLine is that wearable assistive vision systems cannot sustainably stream high-framerate video directly into heavy vision-language models (e.g., Florence-2, GPT-4V) due to extreme edge power constraints, thermal limits, and sensory cognitive overload for blind and low-vision (BLV) navigators. A CPU-side front-end gate evaluating incoming frames against an anchored reference frame can suppress redundant forward passes while preserving prompt detection of safety-critical environmental events.

Our audit identifies significant algorithmic strengths, verifies live physical GPU benchmarks on real-world egocentric video, uncovers divergences between the two repositories, and establishes the required corrective actions to ensure total publication readiness.

---

## 2. Repository Architecture & Authority Analysis

### 2.1 Repository Roles and Commit States
1. **`SightLine` (`C:\Users\princ\Desktop\SightLine`):**
   - **Role:** Authoritative research and algorithmic development testbench.
   - **Commit:** `f8136ae` (*"Add 2D omni-dHash, dead-zone, 4x4 ROI max-pooling, priority speech preemption, and real-video GPU benchmark"*).
   - **Key Modules:**
     - `src/vision/priority_gate.py`: Implements complete multi-cue `PrioritySceneGate` (2D omni-directional dHash, noise dead-zone $\varepsilon=7.0$, $4 \times 4$ ROI block max-pooling $S_{\text{roi\_max}}$, spatial CV, motion proxy, causal hazard persistence, latch cooldown $T_{\text{cool}}=6$, dynamic critical motion override $\tau_{\text{crit}}=0.35$, and two-tier adaptive watchdog).
     - `src/speech/audio_manager.py`: Implements priority-weighted audio queue (`AudioQueue`) with urgent hazard preemption ($\le 150\text{ ms}$).
     - `benchmark_real_video_gpu.py`: Standalone physical GPU benchmark evaluating Florence-2 FP16 on real 1080p egocentric workshop video.
     - `experiments/evaluate_gate.py`: Synthetic benchmark suite across 5 operational scenarios and 10 random seeds.
2. **`SightLine2.O` (`C:\Users\princ\Desktop\SIGHTLINE — BUILD WITH BHARAT 3.0`):**
   - **Role:** Interactive production application and Gradio deployment fork.
   - **Commit:** `0a19d1e` (*"Update README.md"*).
   - **Key Modules:**
     - `src/insights/`: DeepSearch query routing, multi-provider search, and document intelligence pipeline.
     - `src/ui/`: Interactive multi-modal Gradio web interface.
     - `src/vision/scene_change.py`: Implements only naive 1D dHash (`pixels[x] > pixels[x+1]`). **Lacks dead-zone, 2D gradients, ROI max-pooling, and causal hazard persistence.**
     - `tests/`: 33 unit and integration tests covering caching, document parsing, query routing, and model schemas (all 33 tests currently pass).

### 2.2 Authority Determination
- **Algorithmic Authority:** `SightLine` is the authoritative source for gating algorithms, benchmark protocols, and empirical measurements.
- **Application Authority:** `SightLine2.O` is the deployment UI.
- **Action Required:** Harmonize `SightLine2.O` by porting the validated `PrioritySceneGate` from `SightLine` into `SightLine2.O/src/vision/` to replace the naive hash function.

---

## 3. Implementation vs. Manuscript Discrepancy Audit

| Dimension | Manuscript Specification | `SightLine` Repo Implementation | `SightLine2.O` Repo Implementation | Status / Action |
| :--- | :--- | :--- | :--- | :--- |
| **Visual Gating ($S_{\text{vis}}$)** | 2D omni-directional dHash (horizontal + vertical), dead-zone $\varepsilon=7.0$ DN (Eq. 4–5) | Fully implemented in `priority_gate.py` (`use_2d_hash=True`, `eps=7.0`) | Naive 1D horizontal dHash, $\varepsilon=0.0$ | **Harmonized in Repo 1; must port to Repo 2.** |
| **Appearance ($S_{\text{app}}$)** | 48-bin RGB histogram, $L_2$ normalized, Cosine distance (Eq. 6) | Implemented in `priority_gate.py` (`compute_appearance_proxy`) | Absent | **Validated.** |
| **Motion & ROI ($S_{\text{mot}}$)** | $32 \times 32$ thumbnail, $4 \times 4$ ROI block max-pooling ($S_{\text{roi\_max}}$) (Eq. 7–8) | Implemented in `priority_gate.py` (`compute_motion_proxy`, 8x8 pixel blocks) | Absent | **Validated.** |
| **Hazard State ($S_{\text{haz}}$)** | Causal hazard persistence $z_t$, latch cooldown $T_{\text{cool}}=6$, override $\tau_{\text{crit}}=0.35$ (Eq. 9–10) | Implemented in `priority_gate.py` (`evaluate_change`, `update_hazard_state`) | Absent | **Validated.** |
| **Adaptive Watchdog** | Two-tier: $T_{\text{ambient}}=60$ ($2.0\text{ s}$), $T_{\text{active}}=6$ ($200\text{ ms}$) | Implemented in `priority_gate.py` (`adaptive_watchdog=True`) | Absent | **Validated.** |
| **Speech Preemption** | Priority-weighted audio queue (Eq. 11), urgent preemption $\le 150\text{ ms}$ | Implemented in `src/speech/audio_manager.py` (`AudioQueue`) | Basic FIFO `AudioQueue` | **Harmonized in Repo 1; port to Repo 2.** |
| **Gating Latency Accounting** | 3.45 ms on Jetson Orin Nano; 137.8–155.7 ms on full 1080p Pillow CPU downsampling | Verified experimentally: Pillow Lanczos downsampling takes ~140 ms; ISP thumbnail preview is 3.45 ms | N/A | **Distinction clarified in manuscript.** |

---

## 4. Empirical Evidence & Hardware Verification

### 4.1 Testbed Hardware
- **Host CPU:** AMD Ryzen 7 5800H / Intel Core i9 test apparatus.
- **Physical GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB GDDR6 VRAM, Ampere, CUDA 13.4, Driver 616.92).
- **Embedded Target Profile:** NVIDIA Jetson Orin Nano Developer Kit (ARM Cortex-A78AE, Ampere GPU, 8 GB LPDDR5).
- **Vision-Language Model:** Florence-2-base in native FP16 precision (VRAM: $452.6\text{ MiB}$ startup, $618.9\text{ MiB}$ peak during generation).

### 4.2 Real-World Egocentric Video Verification
- **Video:** `video.mov` (1080p, 600 frames evaluated at 30 fps = 20.0s real workshop video of manual tool manipulation).
- **Verified Benchmark Results:**
  1. *Continuous Streaming (30 fps):* 600/600 invocations ($0.0\%$ suppression), $423.45\text{ ms}$ GPU latency, 100% duty cycle, $11.50\text{ W}$ power, $2.83\text{ h}$ usable runtime ($32.56\text{ Wh}$).
  2. *Periodic ($k=6$, 5 Hz):* 100/600 invocations ($83.3\%$ suppression), $395.05\text{ ms}$ GPU latency, 100% duty cycle (queue saturated, $39.5\text{ s} > 20.0\text{ s}$), $11.50\text{ W}$ power, $2.83\text{ h}$ runtime.
  3. *Periodic ($k=12$, 2.5 Hz):* 50/600 invocations ($91.7\%$ suppression), $413.41\text{ ms}$ GPU latency, 100% duty cycle ($20.67\text{ s} > 20.0\text{ s}$), $11.50\text{ W}$ power, $2.83\text{ h}$ runtime.
  4. *Exact Perceptual Hash (1D dHash):* 600/600 invocations (**$0.0\%$ suppression**), $417.96\text{ ms}$ GPU latency, 100% duty cycle, $11.50\text{ W}$ power, $2.83\text{ h}$ runtime. **Proves Failure Mode FM-02: analog sensor noise flips bits on 100% of frames.**
  5. *SightLine Default ($\tau=0.18$):* **32/600 invocations ($94.7\%$ suppression)**, $367.01\text{ ms}$ GPU latency, **$58.7\%$ GPU duty cycle**, **$7.52\text{ W}$ avg power**, **$4.33\text{ hours}$ usable runtime (+53.0% battery endurance extension over continuous streaming)**.
  6. *SightLine++ (2D dHash + ROI Max-Pool):* 117/600 invocations ($80.5\%$ suppression), $362.43\text{ ms}$ GPU latency, 100% duty cycle, captures fine localized hand tool movements.

---

## 5. Unsupported Claims & Required Corrective Actions

1. **WorldScribe Comparison:**
   - *Issue:* Comparing an edge micro-model (Florence-2) directly against cloud foundation models (GPT-4V / Gemini) as a head-to-head empirical victory is methodologically flawed.
   - *Correction:* Reframe as an **architectural trade-off analysis** and position SightLine as a **complementary upstream front-end filter** that suppresses 80.5%–94.7% of redundant video frames before cloud transmission.
2. **Power & Energy Accounting:**
   - *Issue:* Ambiguity between nominal battery capacity ($37.0\text{ Wh}$) and usable regulated energy ($32.56\text{ Wh}$ under $88\%$ DC-DC efficiency); mixing measured idle power ($1.85\text{ W}$) with modeled active power ($11.50\text{ W}$).
   - *Correction:* Add **Table V-B** explicitly delineating *Directly Measured*, *Calibrated Modeled*, and *Assumed/Specified* quantities, and report both nominal and usable runtimes consistently.
3. **FMEA Safety Claims:**
   - *Issue:* Claiming FMEA "certifies" safety or guarantees collision avoidance.
   - *Correction:* Reframe as a **structured risk analysis** demonstrating an 88.5% RPN reduction, and explicitly document three physical residual risks (severe camera occlusion, low light <5 lux, and sub-threshold transient motion during heartbeat gaps).
4. **Human-Centered User Study:**
   - *Issue:* Text ambiguously suggesting participant evaluation has been completed.
   - *Correction:* Retitle Section IX-D to *"Proposed Human-Centered Accessibility Protocol (Future Participant Evaluation)"* and clearly state that no human participant trials have been conducted to date.
5. **Statistical Precision:**
   - *Issue:* Blanket claims of "100% hazard recall".
   - *Correction:* Clarify experimental units (PRNG seeds for noise vs. temporal video units) and qualify statements as *"100% observed dynamic recall on the tested benchmark under the stated protocol"*.

---

## 6. Execution Roadmap for Implementation Passes

- **Pass B:** Harmonize `SightLine2.O` with `PrioritySceneGate` and `PriorityAudioQueue`. Build the comprehensive 20-case automated test suite in `SightLine/tests/`.
- **Pass C:** Run fair baselines and systematic ablations (12 configurations) across both synthetic and real video. Generate machine-readable CSV/JSON logs.
- **Pass D:** Recompile the manuscript (`build_flawless_paper_v6.py`), generate clean DOCX and PDF, and finalize all reports (`reports/change_log.md`, `reports/experiment_protocol.md`, `reports/results_summary.md`, `reports/limitations.md`).
- **Pass E:** Perform skeptical peer review and deliver the final reviewer assessment.
