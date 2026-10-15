# SightLine Change Log & Engineering Audit Trail

**Project:** SightLine: Priority-Weighted Scene-Change Gating for Redundancy-Aware Assistive Vision-Language Models  
**Date:** October 15, 2026  
**Repositories:**  
- Primary Testbench: `c:\Users\princ\Desktop\SightLine` (`https://github.com/lovnishverma/sightline`)
- User-Facing Application: `c:\Users\princ\Desktop\SIGHTLINE — BUILD WITH BHARAT 3.0` (`https://github.com/lovnishverma/SightLine2.O.git`)
- Primary Manuscript: `c:\Users\princ\Desktop\Paper\sightline_paper.docx` & `sightline_paper_revised.docx`

---

## 1. Architectural & Source Code Harmonization

### 1.1 PrioritySceneGate Synchronisation
- **Location:** `src/vision/priority_gate.py` (SightLine), `src/vision/priority_gate.py` (SightLine2.O)
- **Status Before:** `SightLine2.O` was running a legacy, uncalibrated gating implementation without dead-zone thresholding, 2D dual-axis gradient hashing, or priority hazard latching.
- **Modifications:**
  - Exported the complete 8-component `PrioritySceneGate` pipeline into `SightLine2.O`.
  - Added safe `None` checking in `update_hazard_state()` to prevent crashes when hazards transition back to nominal scene states (`is_hazard=False`, `hazard_description=None`).
  - Added precision clamping to Cosine distance calculation (`s_app = 0.0` when `cos_sim >= 0.99999`) to eliminate floating-point drift artifacts on identical or static frames.
  - Aligned hardware downsampling paths: supported both direct 320x320 downsampled frame evaluation and 1080p downsampling fallback.
- **Verification:** Unit tested in both repositories; verified 33 passing tests in `SightLine2.O` (commit `c7f7458`) and 29 passing tests in `SightLine`.

### 1.2 Preemptive Priority Audio Queue Synchronization
- **Location:** `src/speech/audio_manager.py` (SightLine), `src/speech/audio_queue.py` (SightLine2.O)
- **Status Before:** FIFO queueing allowed high-volume ambient captions to block critical collision warnings indefinitely.
- **Modifications:**
  - Implemented 4-tier `Priority` enumeration: `HAZARD = 1`, `WARNING = 2`, `TEXT = 3`, `AMBIENT = 4`.
  - Added priority preemption: when an item of priority $\le 2$ (`HAZARD` or `WARNING`) arrives while a lower-priority message is playing, the playback is interrupted immediately.
  - Implemented strict priority heap eviction: when the queue capacity (default: 5) is reached, the lowest-priority, oldest ambient item is dropped.
  - Fixed synchronization bug between internal `_queue` list and `_priority_heap` heap during eviction.
- **Verification:** Tested in `tests/test_audio_queue.py` and `tests/test_20_scenarios.py` across full queue saturation, priority preemption, and edge cases.

---

## 2. Mathematical & Algorithmic Rigor

### 2.1 Formalization of Equation Set
All manuscript equations were rewritten and verified against the implementation:
1. **Perceptual 2D Dual-Axis Difference Hashing:**
   $$\Delta_{x}(i, j) = I(i, j+1) - I(i, j), \quad \Delta_{y}(i, j) = I(i+1, j) - I(i, j)$$
   $$H_{x}(i, j) = \mathbb{I}(\Delta_{x}(i, j) > \varepsilon), \quad H_{y}(i, j) = \mathbb{I}(\Delta_{y}(i, j) > \varepsilon)$$
   Incorporated dead-zone threshold $\varepsilon = 7.0$ (calibrated against CMOS sensor dark-current thermal noise).
2. **Normalized Hamming Distance:**
   $$d_H(H_A, H_B) = \frac{1}{B} \sum_{k=1}^B (H_A[k] \oplus H_B[k]), \quad B = 128\text{ bits}$$
3. **Appearance Dissimilarity:**
   $$s_{\text{app}}(F_t, F_{\text{ref}}) = 1 - \frac{\mathbf{h}_t \cdot \mathbf{h}_{\text{ref}}}{\|\mathbf{h}_t\|_2 \|\mathbf{h}_{\text{ref}}\|_2}$$
   Evaluated over 8x8x8 RGB color histogram (512 bins).
4. **Composite Scene-Change Dissimilarity:**
   $$S(F_t, F_{\text{ref}}) = w_{\text{hash}} d_H(H_t, H_{\text{ref}}) + w_{\text{app}} s_{\text{app}}(F_t, F_{\text{ref}})$$
   Calibrated weights: $w_{\text{hash}} = 0.65$, $w_{\text{app}} = 0.35$, global threshold $\tau_{\text{scene}} = 0.18$.
5. **Spatial ROI Max-Pooling for Localized Hazards:**
   $$M_{\text{max}} = \max_{r \in \mathcal{R}} \Delta_{\text{motion}}(r)$$
   Ensures localized motion in any of 4 quadrant cells triggers immediate override if $M_{\text{max}} \ge \tau_{\text{motion}} = 0.14$.
6. **Two-Rate Watchdog Scheduling:**
   $$T_{\text{watchdog}} = \begin{cases} T_{\text{hazard}} = 1.0\text{ s} (30\text{ frames}) & \text{if hazard latch active} \\ T_{\text{nominal}} = 4.0\text{ s} (120\text{ frames}) & \text{otherwise} \end{cases}$$

---

## 3. Experimental Methodology & Rigor

### 3.1 Dataset Separation & Honest Attribution
- **Synthetic Testbench (10 Random Seeds):**
  - Characterized as a synthetic controlled benchmark simulating 6 distinct assistive events: Static/Idle, Gradual Lighting Shift, Fast Camera Pan, Obstacle Encroachment, Sudden Hazard Onset, and Text Reading Mode.
  - Reported with explicit 95% confidence intervals across seeds:
    - SightLine Default: $84.2\% \pm 1.4\%$ suppression, $100\%$ hazard recall (10/10 synthetic occurrences), mean gating latency $2.41 \pm 0.12\text{ ms}$.
- **Real-World Egocentric Video Evaluation (1080p, 30 fps, 600 frames):**
  - Evaluated on uncurated, physical first-person indoor assistive navigation video (`video.mov`).
  - Addressed sensor noise failure mode FM-02: demonstrated that exact 1D hashing fails completely (0.0% suppression) under physical camera micro-jitter, whereas SightLine with dead-zone thresholding preserves $67.5\%$ suppression without missing sudden scene changes.
- **Physical GPU Telemetry (NVIDIA GeForce RTX 2050 Laptop GPU):**
  - Measured live inference of Florence-2-base in FP16 precision:
    - VRAM Allocation: 452.6 MiB idle baseline, 618.9 MiB active peak.
    - Forward Pass Latency: $362.4\text{--}423.5\text{ ms}$ (mean $380.0\text{ ms}$).
    - Verified queue saturation dynamics: under periodic sampling at 5 Hz ($k=6$), the GPU saturates at 100% duty cycle, preventing energy savings.

### 3.2 Calibrated Power & Battery-Endurance Model
- **Jetson Orin Nano Calibration:**
  - Base idle system power (Carrier + SoC idle + MIPI CSI sensor): $1.85\text{ W}$ (directly measured).
  - Active VLM inference power: $11.50\text{ W}$ (10W power profile cap with transient spikes).
  - Battery capacity: $37.0\text{ Wh}$ nominal, $32.56\text{ Wh}$ usable ($88\%$ DC-DC step-down conversion efficiency).
  - Effective power formula:
    $$P_{\text{avg}} = P_{\text{idle}} + D_{\text{GPU}} \cdot (P_{\text{active}} - P_{\text{idle}})$$
  - SightLine Default achieves $58.7\%$ GPU duty cycle, $7.52\text{ W}$ average power, and **$4.33\text{ hours}$ operational runtime (+53.0% improvement over continuous streaming)**.

---

## 4. Manuscript Restructuring & Claim Calibration

### 4.1 De-escalation of Unsubstantiated Claims
1. **WorldScribe Comparison:**
   - *Previous claim:* SightLine outperforms WorldScribe in efficiency and latency.
   - *Revised claim:* SightLine and WorldScribe represent orthogonal, complementary architectural levels. WorldScribe is an end-to-end multi-modal streaming model requiring heavy computation, whereas SightLine is an ultra-lightweight, hardware-agnostic upstream edge filter ($<3.5\text{ ms}$ on CPU) that can sit ahead of WorldScribe or Florence-2 to prune redundant frames before tokenization.
2. **Hazard Recall Claim:**
   - *Previous claim:* "Guaranteed 100% hazard safety."
   - *Revised claim:* "100% recall observed across 10 controlled synthetic hazard events ($N=10$); real-world hazard recall is subject to environmental lighting, occlusion, and motion velocity."
3. **Safety Certification:**
   - *Previous claim:* "Formal safety certification and ISO compliance."
   - *Revised claim:* "Structured Failure Mode and Effects Analysis (FMEA) identifying 7 key risk modes with 3 documented residual risks."
4. **BLV Human Participant Study:**
   - *Previous claim:* Ambiguous phrasing implying user tests were conducted.
   - *Revised claim:* Explicitly designated as a "Proposed Blind and Low-Vision Evaluation Protocol" with complete 15-participant design, IRB submission status, NASA-TLX cognitive load metrics, and safety boundaries.

---

## 5. Verification & Test Suite Summary

- **Total Test Cases:** 29 automated tests in `SightLine`, 33 automated tests in `SightLine2.O`.
- **Pass Rate:** 100% (0 failures, 0 errors, 0 warnings).
- **Core Test Files:**
  - `tests/test_algorithm1.py`: Tests Algorithm 1 mathematical specification, edge thresholding, and hash generation.
  - `tests/test_audio_queue.py`: Tests 4-tier priority preemption, queue overflow eviction, and playback thread safety.
  - `tests/test_20_scenarios.py`: 20 end-to-end edge-case scenarios covering static scenes, gradual drift, rapid panning, localized hazards, document reading lock, watchdog timeouts, and extreme lighting shifts.
