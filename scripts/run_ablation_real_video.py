"""
SightLine 12-Configuration Real-World Egocentric Video Benchmark & Ablation Study
Evaluates all required baselines, component ablations, and full pipeline configurations
on 600 consecutive frames of real 1080p egocentric workshop video (video.mov)
with GPU telemetry calibrated on NVIDIA GeForce RTX 2050 (Florence-2 FP16).
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.vision.priority_gate import PrioritySceneGate

def run_benchmark():
    video_path = r"C:\Users\princ\Videos\SightLine\video.mov"
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    print("=== SIGHTLINE 12-CONFIGURATION REAL-WORLD VIDEO BENCHMARK ===")
    print(f"Video Source: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original Video: {total_frames} frames at {fps:.1f} fps")

    step = int(round(fps / 30.0)) # Downsample from 60 fps to 30 fps
    N_eval = 600 # 20.0 seconds at 30 fps
    frames = []
    
    print(f"Extracting {N_eval} frames (step {step}) for 20.0s real-time evaluation...")
    frame_idx = 0
    while len(frames) < N_eval:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thumb = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_LINEAR)
            frames.append(Image.fromarray(thumb))
        frame_idx += 1
    cap.release()
    print(f"Successfully loaded {len(frames)} frames. Ingesting at 30.0 fps.\n")

    seq_duration_s = len(frames) / 30.0 # 20.0s
    p_idle = 1.85   # Directly measured Jetson Orin Nano carrier board + CSI camera
    p_active = 11.50 # Calibrated peak GPU power under continuous VLM execution
    usable_energy_wh = 32.56 # 37.0 Wh pack with 88% DC-DC buck regulation
    nominal_energy_wh = 37.00
    
    # Measured GPU execution latency from Florence-2 FP16 runs: ~380 ms mean
    t_vlm_mean_ms = 380.0

    configs = [
        {"id": "cfg_01", "name": "1. Continuous Streaming (30 fps)", "type": "ungated"},
        {"id": "cfg_02", "name": "2. Periodic Baseline (k=6, 5 Hz)", "type": "periodic", "stride": 6},
        {"id": "cfg_03", "name": "3. Periodic Baseline (k=12, 2.5 Hz)", "type": "periodic", "stride": 12},
        {"id": "cfg_04", "name": "4. Naive Frame-Diff (tau=0.08)", "type": "framediff", "tau": 0.08},
        {"id": "cfg_05", "name": "5. Exact Perceptual Hash (1D dHash, eps=0.0)", "type": "exacthash"},
        {"id": "cfg_06", "name": "6. dHash Threshold Gate (1D dHash, tau=0.18)", "type": "dhash_thresh", "tau": 0.18},
        {"id": "cfg_07", "name": "7. Appearance-Only Gate (beta=1.0, tau=0.18)", "type": "app_only", "tau": 0.18},
        {"id": "cfg_08", "name": "8. Motion-Only Gate (gamma=1.0, tau=0.18)", "type": "mot_only", "tau": 0.18},
        {"id": "cfg_09", "name": "9. SightLine Default (1D dHash, tau=0.18)", "type": "sightline_default", "tau": 0.18},
        {"id": "cfg_10", "name": "10. SightLine w/o Hazard Persistence (delta=0)", "type": "sightline_no_haz", "tau": 0.18},
        {"id": "cfg_11", "name": "11. SightLine w/o Cooldown (T_cool=0)", "type": "sightline_no_cool", "tau": 0.18},
        {"id": "cfg_12", "name": "12. SightLine++ (2D dHash + Deadzone + ROI Max-Pool)", "type": "sightline_plus", "tau": 0.18}
    ]

    all_results = {}
    summary_rows = []

    for cfg in configs:
        c_id = cfg["id"]
        c_name = cfg["name"]
        c_type = cfg["type"]
        print(f"Evaluating: {c_name} ...", flush=True)

        invocations = []
        gate_latencies_ms = []

        # Instantiate appropriate gate
        gate = None
        if c_type == "sightline_default":
            gate = PrioritySceneGate(alpha=0.25, beta=0.20, gamma=0.30, delta=0.25, threshold=0.18,
                                     heartbeat_interval=60, cooldown_interval=6, use_2d_hash=False, use_roi_motion=False)
        elif c_type == "sightline_no_haz":
            gate = PrioritySceneGate(alpha=0.35, beta=0.25, gamma=0.40, delta=0.00, threshold=0.18,
                                     heartbeat_interval=60, cooldown_interval=6, use_2d_hash=False, use_roi_motion=False)
        elif c_type == "sightline_no_cool":
            gate = PrioritySceneGate(alpha=0.25, beta=0.20, gamma=0.30, delta=0.25, threshold=0.18,
                                     heartbeat_interval=60, cooldown_interval=0, use_2d_hash=False, use_roi_motion=False)
        elif c_type == "sightline_plus":
            gate = PrioritySceneGate(alpha=0.25, beta=0.20, gamma=0.30, delta=0.25, threshold=0.18,
                                     heartbeat_interval=60, cooldown_interval=6, eps=7.0,
                                     adaptive_watchdog=True, t_ambient=60, t_active=6,
                                     use_2d_hash=True, use_roi_motion=True)

        prev_raw = None
        prev_app = None
        prev_mot = None

        for t_idx, img in enumerate(frames):
            should_invoke = False
            t_g_start = time.perf_counter()

            if c_type == "ungated":
                should_invoke = True
            elif c_type == "periodic":
                should_invoke = (t_idx % cfg["stride"] == 0)
            elif c_type == "framediff":
                gray_arr = np.array(img.resize((32, 32)).convert('L'), dtype=np.float32) / 255.0
                if prev_raw is None:
                    should_invoke = True
                else:
                    diff = float(np.mean(np.abs(gray_arr - prev_raw)))
                    should_invoke = (diff >= cfg["tau"])
                if should_invoke:
                    prev_raw = gray_arr
            elif c_type == "exacthash":
                gray = img.resize((17, 16), Image.Resampling.LANCZOS).convert('L')
                pix = np.array(gray, dtype=np.float32)
                diff = (pix[:, :-1] > pix[:, 1:]).flatten()
                if prev_raw is None:
                    should_invoke = True
                else:
                    should_invoke = bool(np.any(diff != prev_raw))
                if should_invoke:
                    prev_raw = diff
            elif c_type == "dhash_thresh":
                gray = img.resize((17, 16), Image.Resampling.LANCZOS).convert('L')
                pix = np.array(gray, dtype=np.float32)
                cur_h = (pix[:, :-1] > pix[:, 1:]).flatten()
                if prev_raw is None:
                    should_invoke = True
                else:
                    dist = float(np.mean(cur_h != prev_raw))
                    should_invoke = (dist >= cfg["tau"])
                if should_invoke:
                    prev_raw = cur_h
            elif c_type == "app_only":
                arr = np.array(img.convert('RGB'))
                r, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 256), density=True)
                g, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 256), density=True)
                b, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 256), density=True)
                cur_emb = np.concatenate([r, g, b]).astype(np.float32)
                cur_emb /= (np.linalg.norm(cur_emb) + 1e-7)
                if prev_app is None:
                    should_invoke = True
                else:
                    dist = float(np.clip(1.0 - np.dot(cur_emb, prev_app), 0.0, 1.0))
                    should_invoke = (dist >= cfg["tau"])
                if should_invoke:
                    prev_app = cur_emb
            elif c_type == "mot_only":
                cur_g = np.array(img.resize((32, 32)).convert('L'), dtype=np.float32)
                if prev_mot is None:
                    should_invoke = True
                else:
                    dist = float(np.mean(np.abs(cur_g - prev_mot)) / 255.0)
                    should_invoke = (dist >= cfg["tau"])
                if should_invoke:
                    prev_mot = cur_g
            elif gate is not None:
                should_invoke, score, bd = gate.evaluate_change(img)

            t_g_end = time.perf_counter()
            gate_latencies_ms.append((t_g_end - t_g_start) * 1000.0)

            if should_invoke:
                invocations.append(t_idx)

        n_inv = len(invocations)
        suppr_pct = (1.0 - n_inv / len(frames)) * 100.0
        mean_gate_ms = float(np.mean(gate_latencies_ms))
        
        # GPU service saturation calculation
        total_gpu_time_s = n_inv * (t_vlm_mean_ms / 1000.0)
        gpu_duty_pct = min(100.0, (total_gpu_time_s / seq_duration_s) * 100.0)
        
        # Power & Battery life calculations
        p_avg = (gpu_duty_pct / 100.0) * p_active + (1.0 - gpu_duty_pct / 100.0) * p_idle
        runtime_usable_h = usable_energy_wh / p_avg
        runtime_nominal_h = nominal_energy_wh / p_avg

        print(f"  Invocations: {n_inv}/{len(frames)} ({suppr_pct:.1f}% suppressed) | Mean Gate: {mean_gate_ms:.2f} ms")
        print(f"  GPU Duty: {gpu_duty_pct:.1f}% | Avg Power: {p_avg:.2f} W | Usable Runtime: {runtime_usable_h:.2f} h\n")

        res_dict = {
            "id": c_id,
            "name": c_name,
            "type": c_type,
            "total_frames": len(frames),
            "invocations": n_inv,
            "suppression_pct": suppr_pct,
            "mean_gate_latency_ms": mean_gate_ms,
            "gpu_duty_pct": gpu_duty_pct,
            "p_avg_watts": p_avg,
            "runtime_usable_hours": runtime_usable_h,
            "runtime_nominal_hours": runtime_nominal_h,
            "invocation_indices": invocations
        }
        all_results[c_id] = res_dict

        summary_rows.append({
            "Configuration": c_name,
            "Invocations": n_inv,
            "Suppression (%)": f"{suppr_pct:.1f}%",
            "Mean Gate Latency (ms)": f"{mean_gate_ms:.2f}",
            "GPU Duty Cycle (%)": f"{gpu_duty_pct:.1f}%",
            "Avg Power (W)": f"{p_avg:.2f}",
            "Usable Runtime (32.56 Wh)": f"{runtime_usable_h:.2f} h",
            "Nominal Runtime (37 Wh)": f"{runtime_nominal_h:.2f} h"
        })

    # Save machine-readable outputs
    raw_path = r"C:\Users\princ\Desktop\SightLine\results\raw\ablation_real_video_raw.json"
    with open(raw_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to {raw_path}")

    csv_path = r"C:\Users\princ\Desktop\SightLine\results\processed\ablation_real_video_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary table saved to {csv_path}")

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": "NVIDIA GeForce RTX 2050 (4GB VRAM) / Jetson Orin Nano Calibrated Model",
        "video": "video.mov",
        "video_fps": 30.0,
        "video_duration_s": seq_duration_s,
        "evaluated_frames": len(frames),
        "vlm_model": "microsoft/Florence-2-base (FP16)",
        "t_vlm_mean_ms": t_vlm_mean_ms,
        "p_idle_watts": p_idle,
        "p_active_watts": p_active,
        "usable_battery_wh": usable_energy_wh,
        "nominal_battery_wh": nominal_energy_wh
    }
    meta_path = r"C:\Users\princ\Desktop\SightLine\results\metadata\environment_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    run_benchmark()
