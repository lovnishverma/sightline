# SightLine: Priority-Weighted Scene-Change Gating for Redundancy-Aware Assistive Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests: 100% Passing](https://img.shields.io/badge/Tests-29%2F29%20Passing-brightgreen.svg)](tests/)
[![Paper: IEEE Format](https://img.shields.io/badge/Paper-Revised%20Manuscript-orange.svg)](paper/)

Official research repository and testbench for **"SightLine: Priority-Weighted Scene-Change Gating for Redundancy-Aware Assistive Vision-Language Models"**.

---

## 📖 Abstract

Wearable assistive devices powered by vision-language models (VLMs) offer rich semantic scene understanding for blind and low-vision (BLV) individuals. However, continuous frame-by-frame VLM execution imposes prohibitive computational loads on edge hardware ($>11\text{ W}$ active draw, $\sim 380\text{ ms}$ forward pass latency on compact edge SoCs), quickly draining battery reserves and causing thermal throttling. Naive periodic sampling fails by either saturating processing queues during bursts or introducing safety-critical detection delays during sudden hazard encroachment.

**SightLine** introduces an ultra-lightweight, multi-cue, CPU-side scene-change gate that executes in $<3.5\text{ ms}$ per frame. By synthesizing:
1. **2D dual-axis gradient perceptual hashing** with dead-zone noise thresholding ($\varepsilon = 7.0$),
2. **Normalized color-histogram appearance dissimilarity**,
3. **Spatial quadrant ROI max-pooling** for localized motion anomalies,
4. **Persistent hazard state latching** with cooldown protection, and
5. **Two-rate adaptive watchdog scheduling** ($1.0\text{ s}$ hazard / $4.0\text{ s}$ nominal),

SightLine reduces redundant VLM forward passes by **$84.2\% \pm 1.4\%$** in controlled multi-event benchmarks while preserving **100% hazard recall** across tested synthetic events. On uncurated physical 1080p egocentric video, SightLine maintains **$67.5\%$ suppression** against CMOS sensor noise, extending usable battery life from **$2.83\text{ hours}$ to $4.33\text{ hours}$ (+53.0% operational runtime)** on NVIDIA Jetson Orin Nano hardware without missing critical visual updates.

---

## 🚀 Key Contributions

- **Ultra-Lightweight Edge Gating:** Sub-3.5 ms CPU evaluation using hardware-downsampled thumbnails ($32\times32$) before invoking expensive GPU models.
- **Micro-Jitter & Sensor Noise Robustness:** Dead-zone difference hashing ($\varepsilon = 7.0$) eliminates false triggers caused by camera sensor thermal noise and head micro-tremors (resolving Failure Mode FM-02).
- **Localized Hazard Sensitivity:** Spatial ROI max-pooling over a $4\times4$ grid ($8\times8$ pixel sub-regions) captures localized obstacle encroachment even when global frame dissimilarity is negligible.
- **Preemptive 4-Tier Audio Scheduling:** Audio playback engine with immediate preemption for critical alerts (`HAZARD` priority $\le 2$) over ongoing ambient descriptions.
- **Empirically Calibrated Power Model:** Grounded in direct hardware telemetry on NVIDIA Jetson Orin Nano ($1.85\text{ W}$ idle, $11.50\text{ W}$ active) and live NVIDIA RTX 2050 GPU profiling of Florence-2 FP16.

---

## 🏛️ System Architecture

```text
               [ Incoming Video Stream (1080p @ 30 fps) ]
                                   │
                                   ▼
             [ Hardware ISP Downsampler (32x32 Thumbnail) ]
                                   │
          ┌────────────────────────┴────────────────────────┐
          ▼                                                 ▼
[ 2D Dual-Axis dHash ]                             [ 48-Bin Color Hist ]
(Dead-Zone eps = 7.0)                              (Appearance Proxy)
          │                                                 │
          └────────────────────────┬────────────────────────┘
                                   ▼
                   [ Composite Dissimilarity S(t) ]
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        ▼                                                     ▼
[ Global S(t) >= tau ]                               [ ROI Max-Pooling ]
        │                                            (M_max >= tau_mot)
        └──────────────────────────┬──────────────────────────┘
                                   ▼
                [ Two-Rate Watchdog / Cooldown Check ]
                                   │
                                   ├─── Suppressed (<3.5 ms CPU) ──> Skip Frame
                                   │
                                   ▼ Triggered
                       [ Edge VLM Forward Pass ]
                     (Florence-2 / 380 ms on GPU)
                                   │
                                   ▼
                   [ 4-Tier Preemptive Audio Queue ]
                     (Hazard Alerts Preempt Ambient)
                                   │
                                   ▼
                   [ Speech Synthesis Output (Piper) ]
```

---

## 📊 Experimental Results

### Controlled Synthetic Benchmark (10 Seeds, 95% Confidence Intervals)

| Configuration | Suppression Rate (%) | Hazard Recall (%) | Mean Gate Latency (ms) | Usable Runtime (32.56 Wh) |
|:---|:---:|:---:|:---:|:---:|
| Continuous Streaming (30 fps) | $0.0\% \pm 0.0\%$ | $100.0\% \pm 0.0\%$ | $0.00\text{ ms}$ | $2.83\text{ h}$ |
| Periodic Baseline ($k=6$, 5 Hz) | $83.3\% \pm 0.0\%$ | $80.0\% \pm 12.0\%$ | $0.00\text{ ms}$ | $2.83\text{ h}^*$ |
| Periodic Baseline ($k=12$, 2.5 Hz)| $91.7\% \pm 0.0\%$ | $50.0\% \pm 15.0\%$ | $0.00\text{ ms}$ | $2.96\text{ h}$ |
| Naive Frame-Diff ($\tau=0.08$) | $76.2\% \pm 3.1\%$ | $70.0\% \pm 14.0\%$ | $1.82 \pm 0.08\text{ ms}$ | $3.58\text{ h}$ |
| Exact Perceptual Hash ($\varepsilon=0.0$) | $0.0\% \pm 0.0\%$ | $100.0\% \pm 0.0\%$ | $2.14 \pm 0.11\text{ ms}$ | $2.83\text{ h}$ |
| **SightLine Default (Full Pipeline)** | **$84.2\% \pm 1.4\%$** | **$100.0\% \pm 0.0\%$** | **$2.41 \pm 0.12\text{ ms}$** | **$4.33\text{ h}$ (+53.0%)** |

*\*Note: Under periodic sampling at 5 Hz ($k=6$), GPU forward pass duration ($380\text{ ms}$) exceeds frame arrival period ($200\text{ ms}$), saturating the GPU at 100% duty cycle and preventing power reduction.*

---

## 🛠️ Installation & Reproduction

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA recommended for GPU profiling)
- Git

### Quickstart
```bash
# Clone the repository
git clone https://github.com/lovnishverma/sightline.git
cd sightline

# Install package and dependencies
pip install -e ".[dev]"
```

### Running the Test Suite
The repository includes a comprehensive 20-scenario automated test suite validating edge cases, watchdog timing, latch cooldown, and priority preemption:
```bash
pytest tests/ -v
```

### Running the Real-Video Ablation Benchmark
To reproduce the 12-configuration evaluation on egocentric video:
```bash
python scripts/run_ablation_real_video.py
```
Output files will be written to:
- `results/raw/ablation_real_video_raw.json`
- `results/processed/ablation_real_video_summary.csv`
- `results/metadata/environment_metadata.json`

---

## 📁 Repository Structure

```text
sightline/
├── configs/               # Hyperparameter and profile YAML configs
├── environment/           # Conda and pip dependency specifications
├── paper/                 # Research manuscript, BibTeX, and PDF
│   ├── manuscript_revised.docx
│   ├── manuscript_revised.pdf
│   └── references.bib
├── reports/               # Formal audit reports and technical appendices
│   ├── initial_audit.md
│   ├── experiment_protocol.md
│   ├── change_log.md
│   ├── results_summary.md
│   └── limitations.md
├── results/               # Raw and processed benchmark artifacts
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── scripts/               # Ablation and benchmark execution scripts
│   └── run_ablation_real_video.py
├── src/
│   ├── vision/            # PrioritySceneGate, dHash, motion, appearance
│   ├── speech/            # Preemptive priority audio queue & TTS
│   └── conversation/      # Contextual memory and token management
├── tests/                 # 20-scenario automated test suite
├── CITATION.cff           # Citation metadata
├── pyproject.toml         # Packaging configuration
└── README.md
```

---

## ⚠️ Limitations & Ethical Disclaimers

1. **Academic Research Prototype:** SightLine has **not** undergone medical device certification (ISO 13485 / IEC 62304) and must not be used as a primary mobility aid or substitute for a white cane or guide dog.
2. **Participant Validation Status:** Human-subject evaluation with blind and low-vision participants is currently documented as a **proposed evaluation protocol** awaiting Institutional Review Board (IRB) approval.
3. **Residual Risk Notice:** Low-contrast obstacles matching background reflectance along the direct optical axis may experience detection delays up to the nominal watchdog interval ($4.0\text{ s}$). Active depth sensing (LiDAR / ToF) is recommended for safety-critical deployments.

For full failure mode analyses, consult [`reports/limitations.md`](reports/limitations.md).

---

## 📄 Citation

```bibtex
@article{verma2026sightline,
  title={SightLine: Priority-Weighted Scene-Change Gating for Redundancy-Aware Assistive Vision-Language Models},
  author={Verma, Lovnish},
  journal={arXiv preprint},
  year={2026}
}
```
