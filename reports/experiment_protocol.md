# SightLine: Formal Experimental Protocol & Scientific Framework
**Document Version:** 1.0  
**Date:** October 15, 2026  
**Status:** Canonical Experimental Specification  

---

## 1. Central Research Question & Formal Hypotheses

### 1.1 Central Research Question
> *"Can a lightweight, priority-weighted scene-change gate reduce redundant vision-language model (VLM) invocations in egocentric assistive video while maintaining acceptable event-detection recall, bounded event-to-alert latency, and measurable edge energy savings?"*

### 1.2 Formal Scientific Hypotheses
- **H1 (Redundancy Reduction):** Priority-weighted scene-change gating suppresses $\ge 80\%$ of redundant video frames relative to continuous 30 fps inference in static, ambient, and slow-moving scenes without increasing front-end gating delay beyond the pedestrian stopping budget ($W=6\text{ frames}$, $200\text{ ms}$).
- **H2 (Baselines Superiority):** SightLine provides a strictly superior efficiency–detection Pareto trade-off compared to naive frame-differencing and exact perceptual hashing under realistic CMOS sensor noise.
- **H3 (Safety-Aware Latency Bounding):** Persisted hazard tracking ($z_t$) combined with a two-tier adaptive watchdog ($T_{\text{ambient}}=60, T_{\text{active}}=6$) bounds worst-case detection delay to $\le 166.7\text{ ms}$ during active locomotion, while $4 \times 4$ ROI block max-pooling eliminates localized obstacle cue dilution ($p < 0.0001$).
- **H4 (Speech Contention Resolution):** Priority-weighted speech queue scheduling (Eq. 11) with urgent preemption reduces hazard warning delivery latency by $\ge 90\%$ under auditory queue backlog compared to standard FIFO speech queues.
- **H5 (Edge Energy Autonomy):** Frame suppression translates directly into reduced GPU active duty cycle and lower time-averaged power, extending physical battery operating hours by $\ge 50\%$ on wearable edge hardware.
- **H6 (Task Generalization & Limitations):** Standalone perceptual difference hashing fails on document reading due to high-frequency sensor noise on flat paper backgrounds and sign-invariance along text lines, requiring explicit dead-zone thresholding ($\varepsilon = 7.0$) or downstream OCR bounding boxes.

---

## 2. Experimental Units & Outcome Metrics

### 2.1 Units of Analysis
To avoid pseudo-replication from adjacent, highly correlated video frames:
1. **Synthetic Evaluation:** Evaluated across 10 deterministic pseudo-random number generator (PRNG) seeds: `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`. Each seed instantiates an independent sensor noise realization and trajectory.
2. **Real-World Video Evaluation:** Evaluated on continuous real-world egocentric workshop footage (`video.mov`, 1080p, 600 consecutive frames = 20.0s at 30 fps) capturing naturalistic human tool interactions, head rotation, hand occlusion, and industrial backgrounds.

### 2.2 Mathematical Metric Formulations
1. **Redundancy Suppression ($S$):**
   $$\text{Suppression} = \left(1 - \frac{N_{\text{inv}}}{T}\right) \times 100\%$$
   where $N_{\text{inv}} = \sum_{t=0}^{T-1} D_t$ is total downstream VLM forward passes and $T$ is total video frames.
2. **Observed Dynamic Recall ($\text{Recall}_{\text{dyn}}$):**
   $$\text{Recall}_{\text{dyn}} = \frac{1}{|G_{\text{dyn}}|} \sum_{g \in G_{\text{dyn}}} \mathbb{I}\left( \exists t \in [g, \min(T-1, g + W - 1)] : D_t = 1 \right)$$
   where $G_{\text{dyn}} = G \setminus \{0\}$ excludes the mandatory startup frame $t=0$, and $W = 6\text{ frames}$ ($200\text{ ms}$) is the kinematic evaluation window.
3. **True Detection Delay ($\Delta t_{\text{delay}}$):**
   $$\Delta t_{\text{delay}} = (t_{\text{first}} - g) \cdot \Delta t$$
   measured strictly for successfully detected events ($t_{\text{first}} \in [g, g + W - 1]$).
4. **End-to-End Latency ($\Delta t_{\text{E2E}}$):**
   $$\Delta t_{\text{E2E}} = t_{\text{gate}} + t_{\text{vlm}} + t_{\text{tts}}$$
   where $t_{\text{gate}}$ is CPU decision time, $t_{\text{vlm}}$ is GPU forward pass latency, and $t_{\text{tts}}$ is neural TTS start latency.
5. **GPU Active Duty Cycle ($D_{\text{gpu}}$):**
   $$D_{\text{gpu}} = \min\left(1.0, \frac{N_{\text{inv}} \cdot t_{\text{vlm}}}{T_{\text{seq}}}\right) \times 100\%$$
6. **Time-Averaged System Power ($P_{\text{avg}}$):**
   $$P_{\text{avg}} = D_{\text{gpu}} \cdot P_{\text{active}} + (1 - D_{\text{gpu}}) \cdot P_{\text{idle}}$$
7. **Usable Operating Runtime ($T_{\text{run}}$):**
   $$T_{\text{run}} = \frac{E_{\text{eff}}}{P_{\text{avg}}} = \frac{E_{\text{nominal}} \cdot \eta_{\text{reg}}}{P_{\text{avg}}}$$
   evaluated with $E_{\text{nominal}} = 37.0\text{ Wh}$, $\eta_{\text{reg}} = 0.88 \implies E_{\text{eff}} = 32.56\text{ Wh}$.

---

## 3. Evaluation Architectures & Configurations

The benchmark suite systematically evaluates 12 explicit configurations under identical input data:
1. **Continuous Streaming (30 fps):** $D_t = 1 \ \forall t$.
2. **Periodic Baseline ($k=6$, 5.0 Hz):** $D_t = 1 \iff (t \bmod 6 = 0)$.
3. **Periodic Baseline ($k=12$, 2.5 Hz):** $D_t = 1 \iff (t \bmod 12 = 0)$.
4. **Naive Frame-Diff ($\tau=0.08$):** Consecutive thumbnail diff $|I_t - I_{t-1}| \ge 0.08$.
5. **Exact Perceptual Hash (1D dHash):** Exact 1D Hamming distance $H(h_t, h_{t-1}) > 0$.
6. **dHash Threshold Gate ($\tau=0.18$):** 1D difference hash compared to reference frame $I_{t_p}$ with $\tau=0.18$.
7. **Appearance-Only Gate ($\beta=1.0$):** 48-bin RGB histogram Cosine distance $1 - \cos(\mathbf{c}_t, \mathbf{c}_{t_p}) \ge 0.18$.
8. **Motion-Only Gate ($\gamma=1.0$):** $32 \times 32$ grayscale MAD against $I_{t_p} \ge 0.18$.
9. **SightLine Default ($\tau=0.18$):** Multi-factor fusion $(\alpha=0.25, \beta=0.20, \gamma=0.30, \delta=0.25)$, 1D dHash, $T_{\text{ambient}}=60, T_{\text{active}}=6, T_{\text{cool}}=6$.
10. **SightLine w/o Hazard Persistence ($\delta=0.0$):** Multi-cue gate with hazard state disabled $(\alpha=0.35, \beta=0.25, \gamma=0.40, \delta=0.0)$.
11. **SightLine w/o Cooldown ($T_{\text{cool}}=0$):** Full multi-cue gate with cooldown latch disabled.
12. **SightLine++ (Proposed Full Pipeline):** 2D omni-directional dHash with dead-zone $\varepsilon=7.0$, $4 \times 4$ ROI block max-pooling ($S_{\text{roi\_max}}$), adaptive two-tier watchdog ($T=60/6$), and latch cooldown with dynamic critical override ($\tau_{\text{crit}}=0.35$).

---

## 4. Hardware Test Apparatus & Telemetry

1. **Host Workstation:** AMD Ryzen 7 5800H (8 cores, 16 threads, 3.2–4.4 GHz) / Intel Core i9-13900K, 32 GB RAM.
2. **Physical GPU Platform:** NVIDIA GeForce RTX 2050 Laptop GPU (2048 CUDA cores, 4 GB GDDR6, Ampere architecture, Driver 616.92, CUDA 13.4). Ingesting Florence-2-base in FP16 precision.
3. **Target Wearable Profile:** NVIDIA Jetson Orin Nano Developer Kit (6-core ARM Cortex-A78AE, 1024 Ampere GPU cores, 8 GB LPDDR5).
   - $P_{\text{idle}} = 1.85\text{ W}$ (Directly measured on carrier board with active MIPI CSI camera via USB-C inline power meter at 100 Hz).
   - $P_{\text{active}} = 11.50\text{ W}$ (Calibrated active compute under sustained Florence-2 inference).
   - Battery: Standard 10,000 mAh Li-ion power bank ($37.0\text{ Wh}$ nominal, $32.56\text{ Wh}$ usable under $88\%$ DC-DC regulator efficiency).

---

## 5. Statistical Analysis Protocol

1. **Uncertainty Reporting:** All synthetic results reported as Mean $\pm$ Standard Deviation across 10 independent seeds. 95% Confidence Intervals calculated via Student's $t$-distribution.
2. **Hypothesis Testing:**
   - Paired Student's $t$-test for normally distributed metric differences.
   - Wilcoxon signed-rank test for non-parametric paired latency distributions.
   - Two One-Sided Tests (TOST) equivalence testing for equivalence bounds ($|\Delta| \ge 10\%$).
   - Effect sizes quantified via Cohen's $d$.
