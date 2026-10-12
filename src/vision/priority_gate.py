import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any

class PrioritySceneGate:
    """
    Priority-Weighted Scene-Change Gating for Redundancy-Aware Assistive VLMs.
    
    Causally couples perceptual difference hashing, color histogram dissimilarity,
    grayscale thumbnail motion proxy, and causally persisted hazard state with
    configurable watchdog heartbeat (T_max) and hazard latch cooldown (T_cool).
    """
    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 0.20,
        gamma: float = 0.30,
        delta: float = 0.25,
        threshold: float = 0.18,
        heartbeat_interval: Optional[int] = 15,
        cooldown_interval: int = 6,
        hazard_vocabulary: Optional[set] = None
    ):
        tot = alpha + beta + gamma + delta
        self.alpha = alpha / tot
        self.beta = beta / tot
        self.gamma = gamma / tot
        self.delta = delta / tot
        self.threshold = threshold
        self.heartbeat_interval = heartbeat_interval
        self.cooldown_interval = cooldown_interval
        
        self.hazard_vocabulary = hazard_vocabulary or {
            'car', 'vehicle', 'truck', 'bus', 'stairs', 'curb', 'hole', 'obstacle'
        }
        
        # Reference representations (most recently invoked frame)
        self.prev_hash: Optional[np.ndarray] = None
        self.prev_emb: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        
        # Causally persisted hazard state and temporal counters
        self.cached_hazard: float = 0.0
        self.frames_since_invocation: int = 0
        self.frames_since_hazard_alert: float = float('inf')

    def compute_dhash(self, img: Image.Image, size: int = 16) -> np.ndarray:
        """Compute difference hash (dHash) using Lanczos downsampling."""
        gray = img.resize((size + 1, size), Image.Resampling.LANCZOS).convert('L')
        pixels = np.array(gray, dtype=np.float32)
        diff = pixels[:, :-1] > pixels[:, 1:]
        return diff.flatten()

    def compute_appearance_proxy(self, img: Image.Image) -> np.ndarray:
        """
        Compute 48-bin concatenated RGB color histogram (L2-normalized)
        as an appearance-based semantic proxy for standalone validation.
        """
        arr = np.array(img.convert('RGB'))
        r_hist, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 256), density=True)
        g_hist, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 256), density=True)
        b_hist, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 256), density=True)
        emb = np.concatenate([r_hist, g_hist, b_hist])
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-7)

    def evaluate_change(
        self,
        img: Image.Image,
        detection_labels: Optional[List[str]] = None
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Evaluate whether the incoming frame warrants expensive VLM invocation.
        Returns (should_invoke, change_score, feature_breakdown).
        """
        cur_hash = self.compute_dhash(img)
        cur_emb = self.compute_appearance_proxy(img)
        cur_gray = np.array(img.resize((32, 32), Image.Resampling.BILINEAR).convert('L'), dtype=np.float32)

        # Startup unconditional pass at t = 0
        if self.prev_hash is None:
            self.prev_hash = cur_hash
            self.prev_emb = cur_emb
            self.prev_gray = cur_gray
            self.frames_since_invocation = 0
            if detection_labels is not None:
                has_haz = any(l.lower() in self.hazard_vocabulary for l in detection_labels)
                self.cached_hazard = 1.0 if has_haz else 0.0
                self.frames_since_hazard_alert = 0 if has_haz else float('inf')
            breakdown = {
                's_vis': 1.0,
                's_app': 1.0,
                's_mot': 1.0,
                's_haz': self.cached_hazard,
                'score': 1.0,
                'heartbeat_forced': False
            }
            return True, 1.0, breakdown

        # 1. Visual Hamming Distance
        s_vis = float(np.mean(cur_hash != self.prev_hash))
        
        # 2. Appearance Proxy Dissimilarity (Cosine)
        cos_sim = float(np.dot(cur_emb, self.prev_emb))
        s_app = float(np.clip(1.0 - cos_sim, 0.0, 1.0))
        
        # 3. Motion Magnitude Proxy (Normalized MAD on 32x32 thumbnail)
        s_mot = float(np.mean(np.abs(cur_gray - self.prev_gray)) / 255.0)
        
        # 4. Hazard state with latch cooldown
        self.frames_since_hazard_alert += 1
        if self.cooldown_interval > 0 and self.cached_hazard > 0:
            s_haz = 0.0 if self.frames_since_hazard_alert < self.cooldown_interval else float(self.cached_hazard)
        else:
            s_haz = float(self.cached_hazard)

        # Multi-factor change score fusion
        change_score = float(
            self.alpha * s_vis +
            self.beta * s_app +
            self.gamma * s_mot +
            self.delta * s_haz
        )
        
        self.frames_since_invocation += 1
        heartbeat_forced = bool(
            self.heartbeat_interval is not None and
            self.frames_since_invocation >= self.heartbeat_interval
        )
        should_invoke = bool(change_score >= self.threshold or heartbeat_forced)

        # Update reference state and hazard cache ONLY upon invocation
        if should_invoke:
            self.prev_hash = cur_hash
            self.prev_emb = cur_emb
            self.prev_gray = cur_gray
            self.frames_since_invocation = 0
            if detection_labels is not None:
                has_haz = any(l.lower() in self.hazard_vocabulary for l in detection_labels)
                if has_haz:
                    if self.cached_hazard == 0.0 or self.frames_since_hazard_alert >= self.cooldown_interval:
                        self.frames_since_hazard_alert = 0
                    self.cached_hazard = 1.0
                else:
                    self.cached_hazard = 0.0
                    self.frames_since_hazard_alert = float('inf')

        breakdown = {
            's_vis': s_vis,
            's_app': s_app,
            's_mot': s_mot,
            's_haz': s_haz,
            'score': change_score,
            'heartbeat_forced': heartbeat_forced
        }
        return should_invoke, change_score, breakdown

    def reset(self):
        """Reset internal temporal states."""
        self.prev_hash = None
        self.prev_emb = None
        self.prev_gray = None
        self.cached_hazard = 0.0
        self.frames_since_invocation = 0
        self.frames_since_hazard_alert = float('inf')
