# SightLine: Comprehensive Limitations & Residual Risk Analysis

**Document Status:** Official Research Appendix & Failure Boundary Documentation  
**Project:** SightLine: Priority-Weighted Scene-Change Gating for Redundancy-Aware Assistive Vision-Language Models  
**Date:** October 15, 2026  

---

## 1. Executive Summary

While SightLine demonstrates substantial gains in inference efficiency (+53.0% operational battery endurance on edge hardware) and achieves sub-3.5 ms CPU gating latency, deployment in assistive robotics and wearable aids carries severe safety responsibilities. This document outlines the explicit technical boundaries, unverified operational domains, and residual risks inherent in the current system.

---

## 2. Experimental & Empirical Limitations

### 2.1 Lack of Empirical Human-Subject Validation
- **Current Status:** No blind or low-vision (BLV) participants have yet evaluated SightLine in physical trials.
- **Scientific Implication:** Claims regarding speech intelligibility, cognitive fatigue reduction, conversational naturalness, and orientation confidence are strictly theoretical hypotheses.
- **Required Protocol:** Institutional Review Board (IRB) approval is pending. Field trials with 15 BLV users across urban navigation, indoor transit, and reading scenarios are required before claiming assistive efficacy.

### 2.2 Dataset Scope and Generalization
- **Synthetic Testbench:** The 10-seed synthetic benchmark relies on 2D synthetic image manipulations (affine transformations, Gaussian blurs, artificial noise, synthetic obstacle insertions). While useful for controlled parameter sweeps, it does not capture complex 3D parallax, non-rigid obstacle deformation, or dynamic multi-agent traffic.
- **Egocentric Video:** Real-world validation was conducted on a single 600-frame (20-second) uncurated indoor walking sequence at 1080p 30 fps. Broader evaluation across diverse egocentric datasets (e.g., Ego4D, EPIC-KITCHENS) across varying daylight conditions, night scenes, weather conditions, and wearer gaits is essential.

### 2.3 Single VLM Backbone Characterization
- Telemetry was characterized using Florence-2-base in FP16 precision. While representative of small edge VLMs (0.23B parameters), heavier multi-modal models (e.g., LLaVA-OneVision, PaliGemma 3B, Qwen2-VL 7B) exhibit different memory footprints, longer prompt processing latencies ($>1.2\text{ s}$), and distinct thermal throttling envelopes on edge SoCs.

---

## 3. Algorithmic & Sensing Failure Modes

### 3.1 Failure Mode Analysis (FMEA Summary)

| ID | Failure Mode | Root Mechanism | System Mitigation | Residual Severity / Likelihood |
|:---|:---|:---|:---|:---:|
| **FM-01** | Cooldown Blindness | Hazard onset during latch cooldown ($t_{\text{cooldown}} = 2.0\text{ s}$) | Localized ROI motion override bypasses cooldown immediately | Medium / Low |
| **FM-02** | Micro-Jitter Triggering | Head tremor and CMOS thermal noise trigger false scene changes | Dual-axis dead-zone thresholding ($\varepsilon = 7.0$) | Low / Medium |
| **FM-03** | Gradual Illumination Drift | Twilight or passing clouds accumulate difference against $F_{\text{ref}}$ | Two-rate nominal watchdog ($T_{\text{nominal}} = 4.0\text{ s}$) refreshes $F_{\text{ref}}$ | Low / High |
| **FM-04** | Fast Pan Saturation | Continuous rapid head turning forces 100% frame throughput | Adaptive rate limiter caps forward passes at 5 Hz | Medium / Medium |
| **FM-05** | Camouflaged Obstacle | Low-contrast hazard matches wall/floor texture and color histogram | Edge-preserving gradient hashing; fallback to periodic watchdog | High / Low |
| **FM-06** | Text Reading Mode Lock | User holds document steady; gate suppresses text parsing updates | Dedicated Document Mode bypasses gate upon text detection | Medium / Low |
| **FM-07** | Audio Queue Starvation | High-priority warnings starve ambient navigation feedback | Multi-tier priority heap with bounded hazard latch duration | Low / Low |

### 3.2 Detailed Analysis of 3 Critical Residual Risks

#### Residual Risk 1: Camouflaged Obstacle Approaching at Constant Velocity in Monochromatic Space
- **Hazard:** When an obstacle possesses identical surface reflectance, color distribution, and texture to the background wall, the color histogram distance $s_{\text{app}} \approx 0$ and the perceptual hash difference $d_H \approx 0$. If the wearer moves along the optical axis with zero lateral parallax, optical flow motion cues remain minimal.
- **Consequence:** The gate may fail to detect the encroaching hazard until the nominal watchdog triggers (up to $4.0\text{ s}$ delay). In an active assistive context, a 4-second delay at normal walking speed ($1.2\text{ m/s}$) corresponds to $4.8\text{ meters}$ of unguided travel, risking collision.
- **Recommended Safeguard:** Integrating active depth sensing (Time-of-Flight LiDAR or stereo depth) as an independent hardware override.

#### Residual Risk 2: Sensor Over-Exposure / Under-Exposure Transition Blindness
- **Hazard:** Rapid transitions between extreme illumination environments (e.g., stepping from a dark corridor into bright direct sunlight) saturate CMOS exposure algorithms. During automatic exposure convergence ($300\text{--}800\text{ ms}$), frames are completely blown out or underexposed.
- **Consequence:** Difference hash and appearance cues fluctuate wildly, generating false positive gate triggers followed by near-zero gradient information once clipped.
- **Recommended Safeguard:** Integrating camera auto-exposure status signals into the gating state machine to inhibit gating decisions during exposure instability.

#### Residual Risk 3: Auditory Preemption Cognitive Distraction
- **Hazard:** Preemptive interruption of speech output ensures immediate delivery of hazard alerts (Priority 1). However, frequent abrupt truncation of ongoing speech sentences creates cognitive overload and acoustic confusion for BLV users.
- **Consequence:** Users may experience high frustration or fail to comprehend fractured sentence fragments.
- **Recommended Safeguard:** Implementing semantic speech wrapping (e.g., appending an explicit audio cue chime before preemption) and ergonomic tone-based spatial warnings.

---

## 4. Hardware Deployment & Battery Modeling Boundaries

1. **Power Modeling Assumptions:**
   - The power consumption model ($P_{\text{avg}} = P_{\text{idle}} + D_{\text{GPU}} \cdot (P_{\text{active}} - P_{\text{idle}})$) assumes constant thermal dissipation. On compact, fanless wearable form-factors, prolonged GPU bursts trigger dynamic thermal throttling (DVFS), which lowers GPU clock speeds and increases per-frame inference latency from $380\text{ ms}$ to $>600\text{ ms}$.
2. **Camera Interface Latency:**
   - Zero-copy hardware downsampling depends on MIPI CSI-2 ISP hardware scaling (e.g., NVIDIA Jetson Argus API or libcamera ISP). When camera streams are read via standard USB UVC or software decoded streams, CPU downsampling adds $25\text{--}40\text{ ms}$ of overhead, reducing net energy efficiency.
3. **Battery Degradation:**
   - The $37.0\text{ Wh}$ battery model assumes nominal ambient temperature ($25^\circ\text{C}$) and battery health (100% State of Health). Operating in cold outdoor environments ($<5^\circ\text{C}$) reduces effective electrochemical capacity by $20\text{--}35\%$.

---

## 5. Formal Safety Disclaimer

**SightLine is an academic research prototype and has NOT undergone formal ISO 13485 (Medical Devices), ISO 14971 (Risk Management for Medical Devices), or IEC 62304 (Medical Device Software) certification.**

It is strictly an assistive aid and must never be used as a primary mobility guide or substitute for a white cane, guide dog, or human sighted guide in hazardous environments.
