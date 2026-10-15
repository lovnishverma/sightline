import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any

class PrioritySceneGate:
    """
    Priority-Weighted Scene-Change Gating for Redundancy-Aware Assistive VLMs.
    
    Causally couples perceptual difference hashing (1D or omni-directional 2D),
    color histogram dissimilarity, grayscale thumbnail motion proxy (with optional
    Region-of-Interest spatial block max-pooling), and causally persisted hazard state
    with configurable watchdog heartbeat (T_max), hazard latch cooldown (T_cool),
    and adaptive two-tier ambient/active rate switching.
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
        hazard_vocabulary: Optional[set] = None,
        eps: float = 0.0,
        theta_override: float = 0.02,
        theta_active: float = 0.02,
        t_ambient: Optional[int] = 60,
        t_active: int = 6,
        adaptive_watchdog: bool = False,
        quiescent_mode: bool = False,
        use_2d_hash: bool = False,
        use_roi_motion: bool = False,
        roi_weight: float = 0.50,
        theta_roi: float = 0.08,
    ):
        tot = alpha + beta + gamma + delta
        self.alpha = alpha / tot
        self.beta = beta / tot
        self.gamma = gamma / tot
        self.delta = delta / tot
        self.threshold = threshold
        self.heartbeat_interval = heartbeat_interval
        self.cooldown_interval = cooldown_interval
        self.eps = eps
        self.theta_override = theta_override
        self.theta_active = theta_active
        self.t_ambient = t_ambient
        self.t_active = t_active
        self.adaptive_watchdog = adaptive_watchdog
        self.quiescent_mode = quiescent_mode
        self.use_2d_hash = use_2d_hash
        self.use_roi_motion = use_roi_motion
        self.roi_weight = roi_weight
        self.theta_roi = theta_roi
        
        self.hazard_vocabulary = hazard_vocabulary or {
            'car', 'vehicle', 'truck', 'bus', 'bicycle', 'motorcycle', 'scooter',
            'pedestrian', 'person', 'stairs', 'curb', 'hole', 'obstacle', 'pole'
        }
        
        # Reference representations (most recently invoked frame)
        self.prev_hash: Optional[np.ndarray] = None
        self.prev_emb: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        
        # Causally persisted hazard state and temporal counters
        self.cached_hazard: float = 0.0
        self.frames_since_invocation: int = 0
        self.frames_since_hazard_alert: float = float('inf')

    def compute_dhash(self, img: Image.Image, size: int = 16, eps: Optional[float] = None) -> np.ndarray:
        """
        Compute difference hash (dHash) using Lanczos downsampling and dead-zone.
        If self.use_2d_hash is True, computes 2D omni-directional differences (horizontal + vertical).
        """
        e = self.eps if eps is None else eps
        if not self.use_2d_hash:
            # 1D Horizontal dHash
            gray = img.resize((size + 1, size), Image.Resampling.LANCZOS).convert('L')
            pixels = np.array(gray, dtype=np.float32)
            diff = pixels[:, :-1] - pixels[:, 1:]
            return (diff > e).flatten() if e > 0.0 else (diff > 0.0).flatten()
        else:
            # 2D Omni-directional dHash (Horizontal + Vertical gradients)
            gray_h = img.resize((size + 1, size), Image.Resampling.LANCZOS).convert('L')
            pix_h = np.array(gray_h, dtype=np.float32)
            diff_h = pix_h[:, :-1] - pix_h[:, 1:]
            bits_h = (diff_h > e).flatten() if e > 0.0 else (diff_h > 0.0).flatten()

            gray_v = img.resize((size, size + 1), Image.Resampling.LANCZOS).convert('L')
            pix_v = np.array(gray_v, dtype=np.float32)
            diff_v = pix_v[:-1, :] - pix_v[1:, :]
            bits_v = (diff_v > e).flatten() if e > 0.0 else (diff_v > 0.0).flatten()

            return np.concatenate([bits_h, bits_v])

    def compute_appearance_proxy(self, img: Image.Image) -> np.ndarray:
        """
        Compute 48-bin concatenated RGB color histogram (L2-normalized)
        as an appearance-based semantic proxy.
        """
        arr = np.array(img.convert('RGB'))
        r_hist, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 256), density=True)
        g_hist, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 256), density=True)
        b_hist, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 256), density=True)
        emb = np.concatenate([r_hist, g_hist, b_hist])
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-7)

    def compute_motion_proxy(self, cur_gray: np.ndarray, prev_gray: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute grayscale motion magnitude proxy against reference frame.
        Returns:
            s_mot_global: Normalized mean absolute difference across thumbnail.
            s_roi_max: Maximum localized difference across a 4x4 spatial block grid.
            cv_spatial: Coefficient of variation (relative spatial variance) of block differences.
        """
        diff = np.abs(cur_gray - prev_gray) / 255.0
        s_mot_global = float(np.mean(diff))

        # 4x4 spatial block grid (8x8 pixel blocks in 32x32 thumbnail)
        block_means = diff.reshape(4, 8, 4, 8).mean(axis=(1, 3))
        s_roi_max = float(block_means.max())
        mean_b = float(block_means.mean())
        std_b = float(block_means.std())
        cv_spatial = float(std_b / (mean_b + 1e-5))

        return s_mot_global, s_roi_max, cv_spatial

    def evaluate_change(
        self,
        img: Image.Image,
        detection_labels: Optional[List[str]] = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluate whether incoming frame warrants expensive VLM invocation.
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
                's_roi_max': 1.0,
                'cv_spatial': 0.0,
                's_haz': self.cached_hazard,
                'score': 1.0,
                'heartbeat_forced': False
            }
            return True, 1.0, breakdown

        # 1. Visual Hamming Distance
        s_vis = float(np.mean(cur_hash != self.prev_hash))
        
        # 2. Appearance Proxy Dissimilarity (Cosine)
        cos_sim = float(np.dot(cur_emb, self.prev_emb))
        if cos_sim >= 0.99999 or np.array_equal(cur_emb, self.prev_emb):
            s_app = 0.0
        else:
            s_app = float(np.clip(1.0 - cos_sim, 0.0, 1.0))
        
        # 3. Motion Magnitude Proxy (Global MAD + optional ROI Block Max)
        s_mot_global, s_roi_max, cv_spatial = self.compute_motion_proxy(cur_gray, self.prev_gray)
        if self.use_roi_motion:
            # Fuse global motion with localized ROI spike
            s_mot = float((1.0 - self.roi_weight) * s_mot_global + self.roi_weight * s_roi_max)
        else:
            s_mot = s_mot_global

        # 4. Adaptive Watchdog interval selection
        motion_active_trigger = (s_mot >= self.theta_active) or (self.use_roi_motion and s_roi_max >= self.theta_roi)
        if self.adaptive_watchdog:
            if self.quiescent_mode:
                # Pedometer-gated: 0 probes during rest unless motion detected
                self.heartbeat_interval = self.t_active if (motion_active_trigger or self.cached_hazard > 0) else None
            else:
                # Paced ambient: switches between T_ambient and T_active
                self.heartbeat_interval = self.t_active if (motion_active_trigger or self.cached_hazard > 0) else self.t_ambient

        # 5. Hazard state with latch cooldown and motion override
        self.frames_since_hazard_alert += 1
        motion_override_trigger = (s_mot >= self.theta_override) or (self.use_roi_motion and s_roi_max >= self.theta_roi)
        if self.cooldown_interval > 0 and self.cached_hazard > 0:
            s_haz = 0.0 if (self.frames_since_hazard_alert < self.cooldown_interval and not motion_override_trigger) else float(self.cached_hazard)
        else:
            s_haz = float(self.cached_hazard)

        # Multi-factor change score fusion
        change_score = float(
            self.alpha * s_vis +
            self.beta * s_app +
            self.gamma * s_mot +
            self.delta * s_haz
        )
        if change_score < 1e-5:
            change_score = 0.0
        
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
                    self.frames_since_hazard_alert = 0  # Unconditional reset on hazard detection
                    self.cached_hazard = 1.0
                else:
                    self.cached_hazard = 0.0
                    self.frames_since_hazard_alert = float('inf')

        breakdown = {
            's_vis': s_vis,
            's_app': s_app,
            's_mot': s_mot,
            's_mot_global': s_mot_global,
            's_roi_max': s_roi_max,
            'cv_spatial': cv_spatial,
            's_haz': s_haz,
            'score': change_score,
            'heartbeat_forced': heartbeat_forced
        }
        return should_invoke, change_score, breakdown

    def update_hazard_state(self, detection_labels: Optional[List[str]]):
        """Update causally persisted hazard state following VLM execution."""
        if detection_labels is None:
            # Fallback if downstream call failed or timed out: retain safe state
            return
        has_haz = any(l.lower() in self.hazard_vocabulary for l in detection_labels)
        if has_haz:
            self.frames_since_hazard_alert = 0
            self.cached_hazard = 1.0
        else:
            self.cached_hazard = 0.0
            self.frames_since_hazard_alert = float('inf')

    def reset(self):
        """Reset internal temporal states."""
        self.prev_hash = None
        self.prev_emb = None
        self.prev_gray = None
        self.cached_hazard = 0.0
        self.frames_since_invocation = 0
        self.frames_since_hazard_alert = float('inf')
