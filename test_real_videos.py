import cv2
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import transformers.modeling_utils
transformers.modeling_utils.PreTrainedModel._supports_sdpa = False

print("Loading Florence-2-base on CUDA...", flush=True)
model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True, torch_dtype=torch.float16, attn_implementation='eager').to('cuda')
processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
print("Model loaded!", flush=True)

test_videos = [
    (r"C:\Users\princ\Videos\SightLine\video.mov", [100, 600, 1200, 2400, 3600]),
    (r"C:\Users\princ\Videos\SightLine\SightLine.mp4", [300, 1500, 3000, 6000])
]

for p, frame_indices in test_videos:
    if not os.path.exists(p):
        print(f"File not found: {p}")
        continue
    cap = cv2.VideoCapture(p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n=== Video: {os.path.basename(p)} ({total} frames, {fps:.1f} fps) ===", flush=True)
    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            inputs = processor(text="<CAPTION>", images=img, return_tensors="pt").to("cuda", torch.float16)
            with torch.no_grad():
                gen = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=64, num_beams=1)
            res = processor.batch_decode(gen, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(res, task="<CAPTION>", image_size=(img.width, img.height))
            print(f"  Frame {f_idx} ({f_idx/fps:.1f}s): {parsed.get('<CAPTION>', '')}", flush=True)
    cap.release()
