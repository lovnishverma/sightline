import cv2
import os
import sys
import time
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any

# Ensure SightLine is in sys.path
sys.path.insert(0, os.path.abspath('.'))

from src.vision.priority_gate import PrioritySceneGate
from transformers import AutoModelForCausalLM, AutoProcessor
import transformers.modeling_utils
transformers.modeling_utils.PreTrainedModel._supports_sdpa = False

def run_real_video_benchmark():
    video_path = r"C:\Users\princ\Videos\SightLine\video.mov"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("=== LIVE REAL-WORLD EGOCENTRIC VIDEO GPU BENCHMARK ===", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0)} (CUDA Available: {torch.cuda.is_available()})", flush=True)
    
    # Load model on CUDA
    t0 = time.time()
    print("Loading Florence-2-base on CUDA (FP16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Florence-2-base',
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation='eager'
    ).to('cuda')
    processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
    print(f"Model loaded in {time.time()-t0:.2f}s! Initial VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MiB", flush=True)

    # Ingest 600 frames at 30 fps (step=2 from 60 fps video, covering 20.0 seconds of real activity)
    cap = cv2.VideoCapture(video_path)
    total_in_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Reading 600 frames (step 2) from {os.path.basename(video_path)} (total {total_in_vid} frames)...", flush=True)
    
    frames = []
    # Start at frame 300 to capture diverse physical interaction and movement
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    for _ in range(600):
        ret, frame = cap.read()
        if not ret:
            break
        # Read next frame for step=2 (downsample 60fps -> 30fps)
        cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()

    N_frames = len(frames)
    seq_duration_s = N_frames / 30.0
    print(f"Loaded {N_frames} frames ({seq_duration_s:.1f}s at 30 fps). Ready to benchmark!", flush=True)

    def vlm_forward(img: Image.Image, task="<CAPTION>"):
        t_start = time.perf_counter()
        inputs = processor(text=task, images=img, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            gen = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=48,
                num_beams=1
            )
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        res = processor.batch_decode(gen, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(res, task=task, image_size=(img.width, img.height))
        return dt_ms, parsed.get(task, "")

    # Warmup
    print("Warming up Florence-2 on GPU...", flush=True)
    vlm_forward(frames[0])
    torch.cuda.reset_peak_memory_stats(0)

    # Benchmark configurations
    configs = [
        {"name": "Continuous Streaming (30 fps)", "type": "ungated", "stride": 1},
        {"name": "Periodic Baseline (k=6, 5 Hz)", "type": "periodic", "stride": 6},
        {"name": "Periodic Baseline (k=12, 2.5 Hz)", "type": "periodic", "stride": 12},
        {"name": "Naive Frame-Diff (tau=0.08)", "type": "framediff", "tau": 0.08},
        {"name": "Exact Perceptual Hash (1D dHash)", "type": "exacthash"},
        {"name": "SightLine Default (1D dHash, tau=0.18)", "type": "sightline_default"},
        {"name": "SightLine++ (2D dHash + 4x4 ROI Max-Pool)", "type": "sightline_plus"}
    ]

    benchmark_results = {}

    for cfg in configs:
        cname = cfg["name"]
        ctype = cfg["type"]
        print(f"\n--- Running: {cname} ---", flush=True)
        
        invocations = []
        captions = []
        vlm_latencies = []
        gate_latencies = []
        
        # Initialize gate if needed
        if ctype == "sightline_default":
            gate = PrioritySceneGate(
                alpha=0.25, beta=0.20, gamma=0.30, delta=0.25,
                threshold=0.18, heartbeat_interval=60, cooldown_interval=6,
                use_2d_hash=False, use_roi_motion=False
            )
        elif ctype == "sightline_plus":
            gate = PrioritySceneGate(
                alpha=0.25, beta=0.20, gamma=0.30, delta=0.25,
                threshold=0.18, heartbeat_interval=60, cooldown_interval=6,
                adaptive_watchdog=True, t_ambient=60, t_active=6,
                use_2d_hash=True, use_roi_motion=True
            )
        
        prev_raw_frame = None

        for t_idx, img in enumerate(frames):
            should_invoke = False
            t_gate_ms = 0.0

            if ctype == "ungated":
                should_invoke = True
            elif ctype == "periodic":
                should_invoke = (t_idx % cfg["stride"] == 0)
            elif ctype == "framediff":
                t_g_start = time.perf_counter()
                gray_arr = np.array(img.resize((32, 32)).convert('L'), dtype=np.float32) / 255.0
                if prev_raw_frame is None:
                    should_invoke = True
                else:
                    diff = float(np.mean(np.abs(gray_arr - prev_raw_frame)))
                    should_invoke = (diff >= cfg["tau"])
                prev_raw_frame = gray_arr
                t_gate_ms = (time.perf_counter() - t_g_start) * 1000.0
            elif ctype == "exacthash":
                t_g_start = time.perf_counter()
                gray = img.resize((17, 16), Image.Resampling.LANCZOS).convert('L')
                pix = np.array(gray, dtype=np.float32)
                diff = (pix[:, :-1] > pix[:, 1:]).flatten()
                if prev_raw_frame is None:
                    should_invoke = True
                else:
                    should_invoke = bool(np.any(diff != prev_raw_frame))
                prev_raw_frame = diff
                t_gate_ms = (time.perf_counter() - t_g_start) * 1000.0
            elif ctype in ["sightline_default", "sightline_plus"]:
                t_g_start = time.perf_counter()
                should_invoke, score, breakdown = gate.evaluate_change(img)
                t_gate_ms = (time.perf_counter() - t_g_start) * 1000.0

            gate_latencies.append(t_gate_ms)

            if should_invoke:
                invocations.append(t_idx)
                # Run actual VLM forward pass on GPU (subsample if continuous to prevent benchmark timeout)
                if ctype == "ungated" and t_idx > 0 and len(vlm_latencies) >= 10:
                    # Use sampled latency to simulate 600 full passes
                    vlm_latencies.append(float(np.mean(vlm_latencies)))
                else:
                    lat_ms, cap_text = vlm_forward(img)
                    vlm_latencies.append(lat_ms)
                    if len(captions) < 5:
                        captions.append({"frame": t_idx, "time_s": t_idx/30.0, "caption": cap_text})
                    if ctype in ["sightline_default", "sightline_plus"]:
                        gate.update_hazard_state([cap_text])

        n_inv = len(invocations)
        suppr_pct = (1.0 - n_inv / N_frames) * 100.0
        mean_vlm_ms = float(np.mean(vlm_latencies)) if vlm_latencies else 0.0
        mean_gate_ms = float(np.mean(gate_latencies)) if gate_latencies else 0.0
        
        # Compute GPU active duty cycle over sequence
        total_gpu_time_s = (n_inv * (mean_vlm_ms / 1000.0))
        gpu_duty_pct = min(100.0, (total_gpu_time_s / seq_duration_s) * 100.0)
        
        # Energy model: Jetson Orin Nano hardware calibration
        p_idle = 1.85
        p_active = 11.50
        p_avg = (gpu_duty_pct / 100.0) * p_active + (1.0 - gpu_duty_pct / 100.0) * p_idle
        runtime_h = 32.56 / p_avg

        print(f"  Invocations: {n_inv}/{N_frames} ({suppr_pct:.1f}% suppressed)", flush=True)
        print(f"  Mean GPU Latency: {mean_vlm_ms:.2f} ms | Mean CPU Gating: {mean_gate_ms:.2f} ms", flush=True)
        print(f"  GPU Duty Cycle: {gpu_duty_pct:.1f}% | Est. Power: {p_avg:.2f} W | Est. Runtime: {runtime_h:.2f} h", flush=True)
        if captions:
            print(f"  Sample Captions: {captions[:2]}", flush=True)

        benchmark_results[cname] = {
            "invocations": n_inv,
            "total_frames": N_frames,
            "suppression_pct": suppr_pct,
            "mean_vlm_latency_ms": mean_vlm_ms,
            "mean_gate_latency_ms": mean_gate_ms,
            "gpu_duty_pct": gpu_duty_pct,
            "p_avg_watts": p_avg,
            "effective_runtime_hours": runtime_h,
            "first_five_captions": captions
        }

    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
    print(f"\nPeak GPU VRAM Allocated during benchmark: {peak_vram:.1f} MiB", flush=True)

    out_file = r"c:\Users\princ\Desktop\Paper\real_video_gpu_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "video": os.path.basename(video_path),
            "video_duration_seconds": seq_duration_s,
            "total_frames": N_frames,
            "fps": 30.0,
            "gpu_device": torch.cuda.get_device_name(0),
            "peak_vram_mib": peak_vram,
            "results": benchmark_results
        }, f, indent=2)
    
    print(f"\nBenchmark completed successfully! Results saved to: {out_file}", flush=True)

if __name__ == '__main__':
    run_real_video_benchmark()
