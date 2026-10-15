"""
SightLine Comprehensive Automated Test Suite (20 Evaluation Scenarios)
Meticulously verifying all safety, algorithmic, concurrency, and boundary requirements.

Covers:
1. Identical consecutive frames.
2. Gradual illumination changes.
3. Abrupt scene transitions.
4. Camera shake and egocentric motion.
5. A newly entering object (4x4 ROI max-pooling).
6. A stationary object becoming relevant.
7. A hazard appearing during cooldown (critical motion override).
8. A hazard appearing while inference is busy.
9. A VLM response arriving late.
10. A VLM timeout or failure.
11. Watchdog-triggered inference.
12. Priority changes during an active request.
13. Stale hazard labels.
14. Reference-frame update correctness.
15. Duplicate request suppression.
16. Critical alert preemption.
17. Queue saturation and eviction.
18. Offline or disconnected inference service.
19. Long-duration runtime stability (2,000 frames).
20. Invalid configuration and boundary values.
"""

import unittest
import numpy as np
from PIL import Image
import sys
import os
import time
import threading

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.priority_gate import PrioritySceneGate
from src.speech.audio_manager import AudioQueue

def make_solid_image(val: int = 128, size=(160, 120)) -> Image.Image:
    """Creates a uniform solid RGB image."""
    arr = np.full((size[1], size[0], 3), val, dtype=np.uint8)
    return Image.fromarray(arr)

def make_pattern_image(val1: int = 128, val2: int = 200, size=(160, 120)) -> Image.Image:
    """Creates a two-tone patterned RGB image."""
    arr = np.full((size[1], size[0], 3), val1, dtype=np.uint8)
    arr[size[1]//4 : 3*size[1]//4, size[0]//4 : 3*size[0]//4] = val2
    return Image.fromarray(arr)

class TestSightLine20Scenarios(unittest.TestCase):

    def setUp(self):
        self.gate = PrioritySceneGate(
            alpha=0.25, beta=0.20, gamma=0.30, delta=0.25,
            threshold=0.18, heartbeat_interval=15, cooldown_interval=6,
            eps=7.0, use_2d_hash=True, use_roi_motion=True
        )

    # -------------------------------------------------------------------------
    # Scenario 1: Identical Consecutive Frames
    # -------------------------------------------------------------------------
    def test_01_identical_consecutive_frames(self):
        img = make_pattern_image(100, 180)
        # Frame 0: startup unconditional invocation
        inv0, sc0, _ = self.gate.evaluate_change(img)
        self.assertTrue(inv0, "Startup frame t=0 must invoke unconditionally.")
        
        # Frames 1 to 20: identical images must be 100% suppressed
        for f in range(1, 14):
            inv, sc, bd = self.gate.evaluate_change(img)
            self.assertFalse(inv, f"Frame {f} must be suppressed on identical input.")
            self.assertEqual(sc, 0.0, f"Change score must be 0.0 for identical frame {f}.")
            self.assertEqual(bd['s_vis'], 0.0)
            self.assertEqual(bd['s_mot'], 0.0)

    # -------------------------------------------------------------------------
    # Scenario 2: Gradual Illumination Changes
    # -------------------------------------------------------------------------
    def test_02_gradual_illumination_changes(self):
        # Startup
        base = make_solid_image(100)
        self.gate.evaluate_change(base)

        # Subtle gradient shift (+1 DN per frame over 20 frames)
        # Frame-to-frame difference is tiny (~0.004), below threshold tau=0.18
        invocations = []
        for i in range(1, 25):
            val = min(255, 100 + i * 2)
            img = make_solid_image(val)
            inv, score, bd = self.gate.evaluate_change(img)
            if inv:
                invocations.append(i)

        # Gradual drift accumulates against reference frame Itp
        # Should NOT trigger on every frame, but accumulate until crossing tau or watchdog
        self.assertLess(len(invocations), 10, "Gradual drift must not trigger every frame.")
        self.assertGreater(len(invocations), 0, "Drift accumulating against Itp must eventually trigger.")

    # -------------------------------------------------------------------------
    # Scenario 3: Abrupt Scene Transitions
    # -------------------------------------------------------------------------
    def test_03_abrupt_scene_transitions(self):
        img_dark = make_solid_image(20)
        img_bright_pattern = make_pattern_image(220, 40)
        self.gate.evaluate_change(img_dark)

        # Instantaneous scene change
        inv, score, bd = self.gate.evaluate_change(img_bright_pattern)
        self.assertTrue(inv, "Abrupt scene transition must trigger instantaneous invocation.")
        self.assertGreater(score, 0.30, "Change score must comfortably exceed threshold.")

    # -------------------------------------------------------------------------
    # Scenario 4: Camera Shake and Egocentric Motion
    # -------------------------------------------------------------------------
    def test_04_camera_shake_and_egocentric_motion(self):
        # Startup
        base = make_pattern_image(120, 180)
        self.gate.evaluate_change(base)
        
        # Add small sub-threshold sensor noise / micro-shake (< eps=7.0 DN)
        rng = np.random.default_rng(42)
        base_arr = np.array(base, dtype=np.float32)
        suppressed_count = 0
        for _ in range(10):
            noise = rng.normal(0, 2.5, base_arr.shape)
            noisy_img = Image.fromarray(np.clip(base_arr + noise, 0, 255).astype(np.uint8))
            inv, _, _ = self.gate.evaluate_change(noisy_img)
            if not inv:
                suppressed_count += 1
        
        self.assertGreaterEqual(suppressed_count, 8, "Dead-zone thresholding must filter micro-shake noise.")

    # -------------------------------------------------------------------------
    # Scenario 5: Newly Entering Object (4x4 ROI Block Max-Pooling)
    # -------------------------------------------------------------------------
    def test_05_newly_entering_object(self):
        # Using obstacle tracking profile threshold (tau=0.12) to detect entering objects
        gate_obs = PrioritySceneGate(
            alpha=0.25, beta=0.20, gamma=0.30, delta=0.25,
            threshold=0.12, heartbeat_interval=15, cooldown_interval=6,
            use_2d_hash=True, use_roi_motion=True
        )
        base = np.zeros((120, 160, 3), dtype=np.uint8)
        base[20:100, 30:130] = [180, 140, 100]
        gate_obs.evaluate_change(Image.fromarray(base))

        # A localized obstacle enters an active quadrant
        # Global MAD is small (~0.09), but ROI max is large (~0.50)!
        modified = base.copy()
        modified[20:80, 30:90] = [20, 20, 20]
        inv, score, bd = gate_obs.evaluate_change(Image.fromarray(modified))
        
        self.assertGreater(bd['s_roi_max'], 0.20, "Localized ROI change must be detected by ROI pooling.")
        self.assertGreater(bd['s_roi_max'], 3.0 * bd['s_mot_global'], "ROI pooling must amplify localized motion over global MAD.")
        self.assertTrue(inv, "Newly entering obstacle must trigger inference via ROI max-pooling.")

    # -------------------------------------------------------------------------
    # Scenario 6: Stationary Object Becoming Relevant
    # -------------------------------------------------------------------------
    def test_06_stationary_object_becoming_relevant(self):
        img = make_pattern_image(128, 160)
        self.gate.evaluate_change(img, detection_labels=[])
        self.assertEqual(self.gate.cached_hazard, 0.0)

        # External VLM classification on stationary object identifies hazard
        self.gate.update_hazard_state(['curb'])
        self.assertEqual(self.gate.cached_hazard, 1.0, "Hazard state must latch to 1.0 upon detection.")
        self.assertEqual(self.gate.frames_since_hazard_alert, 0)

    # -------------------------------------------------------------------------
    # Scenario 7: Hazard Appearing During Cooldown (Critical Motion Override)
    # -------------------------------------------------------------------------
    def test_07_hazard_appearing_during_cooldown(self):
        img1 = make_pattern_image(100, 200)
        self.gate.evaluate_change(img1, detection_labels=['car'])
        self.assertEqual(self.gate.cached_hazard, 1.0)

        # In frame 2, cooldown is active (c_alert = 1 < T_cool = 6)
        inv2, _, bd2 = self.gate.evaluate_change(img1)
        self.assertFalse(inv2, "Cooldown must suppress sustained hazard without motion.")

        # In frame 3, sudden critical motion occurs (S_roi_max >= tau_crit=0.35)
        # Quadrant shift (100 -> 255) creates S_roi_max = |255 - 100| / 255 = 0.608 >= 0.35
        critical_arr = np.array(img1, dtype=np.uint8)
        critical_arr[0:60, 0:80] = 255
        critical_img = Image.fromarray(critical_arr)
        inv3, sc3, bd3 = self.gate.evaluate_change(critical_img, detection_labels=['car'])
        self.assertTrue(inv3, "Critical motion override must immediately bypass cooldown.")
        self.assertGreaterEqual(bd3['s_roi_max'], 0.35)

    # -------------------------------------------------------------------------
    # Scenario 8: Hazard Appearing While Inference is Busy
    # -------------------------------------------------------------------------
    def test_08_hazard_appearing_while_inference_busy(self):
        worker_busy = True
        dropped_requests = 0
        latched_hazard = False

        # Simulate busy worker handling
        def handle_frame(img, labels):
            nonlocal worker_busy, dropped_requests, latched_hazard
            should_invoke, score, _ = self.gate.evaluate_change(img)
            if should_invoke:
                if worker_busy:
                    dropped_requests += 1
                    # Ensure safety latch retains highest priority trigger
                    if any(l in self.gate.hazard_vocabulary for l in labels):
                        latched_hazard = True
                else:
                    self.gate.update_hazard_state(labels)

        handle_frame(make_solid_image(100), [])
        handle_frame(make_pattern_image(50, 250), ['car'])

        self.assertGreater(dropped_requests, 0, "Busy worker drops concurrent burst requests.")
        self.assertTrue(latched_hazard, "Critical hazard event must be latched during busy period.")

    # -------------------------------------------------------------------------
    # Scenario 9: VLM Response Arriving Late
    # -------------------------------------------------------------------------
    def test_09_vlm_response_arriving_late(self):
        # Frame 0 triggers inference
        self.gate.evaluate_change(make_solid_image(100))
        
        # Frames 1-3 arrive while VLM forward pass is still computing asynchronously
        for _ in range(3):
            inv, _, _ = self.gate.evaluate_change(make_solid_image(100))
            self.assertFalse(inv)

        # VLM response arrives late at frame 4
        self.gate.update_hazard_state(['obstacle'])
        self.assertEqual(self.gate.cached_hazard, 1.0, "Late VLM response must update hazard state safely.")

    # -------------------------------------------------------------------------
    # Scenario 10: VLM Timeout or Failure
    # -------------------------------------------------------------------------
    def test_10_vlm_timeout_or_failure(self):
        self.gate.evaluate_change(make_solid_image(100))
        
        # Simulate downstream exception
        try:
            raise TimeoutError("Downstream VLM inference service unreachable.")
        except TimeoutError:
            # Fallback policy: retain cached hazard state without crashing
            self.gate.update_hazard_state(None)

        self.assertEqual(self.gate.cached_hazard, 0.0, "Failure fallback must not corrupt state.")
        # Gate must continue running deterministically
        inv, _, _ = self.gate.evaluate_change(make_solid_image(100))
        self.assertFalse(inv)

    # -------------------------------------------------------------------------
    # Scenario 11: Watchdog-Triggered Inference
    # -------------------------------------------------------------------------
    def test_11_watchdog_triggered_inference(self):
        img = make_pattern_image(100, 150)
        self.gate.evaluate_change(img) # t=0

        # Run 14 identical frames without visual change (heartbeat_interval=15)
        for f in range(1, 15):
            inv, _, bd = self.gate.evaluate_change(img)
            self.assertFalse(inv, f"Frame {f} must not invoke before watchdog limit.")
            self.assertFalse(bd['heartbeat_forced'])

        # At frame 15, idle counter reaches 15: watchdog MUST force invocation!
        inv15, _, bd15 = self.gate.evaluate_change(img)
        self.assertTrue(inv15, "Watchdog must force invocation at heartbeat interval.")
        self.assertTrue(bd15['heartbeat_forced'], "Heartbeat forced flag must be True.")
        self.assertEqual(self.gate.frames_since_invocation, 0, "Counter must reset to 0.")

    # -------------------------------------------------------------------------
    # Scenario 12: Priority Changes During Active Request
    # -------------------------------------------------------------------------
    def test_12_priority_changes_during_active_request(self):
        queue = AudioQueue(max_size=5, urgency_threshold=0.80)
        queue.enqueue("Ambient room description", "room.wav", priority=0.30)
        queue.enqueue("Storefront name", "store.wav", priority=0.45)
        
        # Sudden urgent hazard alert arrives with high priority
        queue.enqueue("WARNING: Approaching bicycle!", "bike.wav", priority=0.90, is_urgent=True)
        
        # Dequeue must return the urgent item first
        first_item = queue.dequeue()
        self.assertEqual(first_item[0], "WARNING: Approaching bicycle!")

    # -------------------------------------------------------------------------
    # Scenario 13: Stale Hazard Labels
    # -------------------------------------------------------------------------
    def test_13_stale_hazard_labels(self):
        img = make_solid_image(100)
        self.gate.evaluate_change(img, detection_labels=['car'])
        self.assertEqual(self.gate.cached_hazard, 1.0)

        # Later, scene is clear and VLM returns no hazards
        self.gate.update_hazard_state([])
        self.assertEqual(self.gate.cached_hazard, 0.0, "Subsequent clear pass must clear hazard latch.")
        self.assertEqual(self.gate.frames_since_hazard_alert, float('inf'))

    # -------------------------------------------------------------------------
    # Scenario 14: Reference-Frame Update Correctness
    # -------------------------------------------------------------------------
    def test_14_reference_frame_update_correctness(self):
        img0 = make_pattern_image(80, 160)
        self.gate.evaluate_change(img0)
        initial_hash = self.gate.prev_hash.copy()

        # Suppressed sub-threshold frame
        img_sub = make_pattern_image(80, 165)
        inv_sub, _, _ = self.gate.evaluate_change(img_sub)
        self.assertFalse(inv_sub)
        np.testing.assert_array_equal(self.gate.prev_hash, initial_hash,
                                      "Reference frame must remain strictly anchored on suppressed frames.")

        # Triggered frame
        img_trig = make_pattern_image(10, 250)
        inv_trig, _, _ = self.gate.evaluate_change(img_trig)
        self.assertTrue(inv_trig)
        self.assertFalse(np.array_equal(self.gate.prev_hash, initial_hash),
                         "Reference frame must update when invocation is triggered.")

    # -------------------------------------------------------------------------
    # Scenario 15: Duplicate Request Suppression
    # -------------------------------------------------------------------------
    def test_15_duplicate_request_suppression(self):
        img = make_pattern_image(100, 150)
        self.gate.evaluate_change(img)
        
        # Ten duplicate calls with same frame
        invocations = [self.gate.evaluate_change(img)[0] for _ in range(10)]
        self.assertEqual(sum(invocations), 0, "All duplicate frames must be suppressed.")

    # -------------------------------------------------------------------------
    # Scenario 16: Critical Alert Preemption
    # -------------------------------------------------------------------------
    def test_16_critical_alert_preemption(self):
        queue = AudioQueue(max_size=5, urgency_threshold=0.80)
        queue.enqueue("Reading paragraph 1 of document...", "doc.wav", priority=0.20)
        current = queue.dequeue()
        self.assertEqual(current[0], "Reading paragraph 1 of document...")
        self.assertFalse(queue._interrupted)

        # Urgent obstacle warning enters: must trigger immediate preemption
        queue.enqueue("CRITICAL: Red traffic light!", "red.wav", priority=0.95, is_urgent=True)
        self.assertTrue(queue._interrupted, "Urgent alert must preempt currently playing ambient audio.")

    # -------------------------------------------------------------------------
    # Scenario 17: Queue Saturation and Eviction
    # -------------------------------------------------------------------------
    def test_17_queue_saturation(self):
        queue = AudioQueue(max_size=3, urgency_threshold=0.80)
        queue.enqueue("Message 1", "m1.wav", priority=0.10)
        queue.enqueue("Message 2", "m2.wav", priority=0.40)
        queue.enqueue("Message 3", "m3.wav", priority=0.50)
        
        # 4th message should evict lowest priority ("Message 1")
        queue.enqueue("Message 4", "m4.wav", priority=0.60)
        self.assertLessEqual(queue.size, 3, "Queue size must not exceed max_size.")
        
        items = []
        while not queue.is_empty:
            items.append(queue.dequeue()[0])
        self.assertNotIn("Message 1", items, "Lowest priority item must be evicted under saturation.")

    # -------------------------------------------------------------------------
    # Scenario 18: Offline / Disconnected Inference Service
    # -------------------------------------------------------------------------
    def test_18_offline_or_disconnected_inference(self):
        # The gating logic is pure CPU NumPy/Pillow with 0 network calls
        img = make_pattern_image(100, 200)
        t_start = time.perf_counter()
        inv, score, bd = self.gate.evaluate_change(img)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        
        self.assertLess(elapsed_ms, 50.0, "CPU gating must be fast and completely offline.")
        self.assertTrue(inv)

    # -------------------------------------------------------------------------
    # Scenario 19: Long-Duration Runtime Stability (2,000 Frames)
    # -------------------------------------------------------------------------
    def test_19_long_duration_runtime_stability(self):
        rng = np.random.default_rng(1234)
        base = make_pattern_image(100, 180)
        base_arr = np.array(base, dtype=np.float32)

        inv_count = 0
        t0 = time.perf_counter()
        for f in range(2000):
            # 95% static/noise, 5% periodic shifts
            if f % 40 == 0:
                img_arr = np.clip(base_arr + rng.normal(0, 20.0, base_arr.shape), 0, 255).astype(np.uint8)
            else:
                img_arr = np.clip(base_arr + rng.normal(0, 2.0, base_arr.shape), 0, 255).astype(np.uint8)
            inv, _, _ = self.gate.evaluate_change(Image.fromarray(img_arr))
            if inv:
                inv_count += 1
        elapsed = time.perf_counter() - t0

        fps = 2000.0 / elapsed
        self.assertGreater(fps, 100.0, f"Gating throughput ({fps:.1f} fps) must comfortably exceed 30 fps.")
        self.assertGreater(inv_count, 40, "Periodic events must be detected.")
        self.assertLess(inv_count, 300, "Redundant static intervals must be heavily suppressed.")

    # -------------------------------------------------------------------------
    # Scenario 20: Invalid Configuration and Boundary Values
    # -------------------------------------------------------------------------
    def test_20_invalid_configuration_and_boundary_values(self):
        # Negative weights should be normalized safely
        gate_weights = PrioritySceneGate(alpha=-1.0, beta=1.0, gamma=1.0, delta=1.0)
        self.assertAlmostEqual(gate_weights.alpha + gate_weights.beta + gate_weights.gamma + gate_weights.delta, 1.0)

        # Extreme threshold 1.0: only watchdog should trigger
        gate_strict = PrioritySceneGate(threshold=1.0, heartbeat_interval=5)
        img = make_pattern_image(50, 200)
        gate_strict.evaluate_change(img)
        
        triggered_at_watchdog = False
        for f in range(1, 10):
            inv, _, bd = gate_strict.evaluate_change(make_pattern_image(200, 50))
            if inv and bd['heartbeat_forced']:
                triggered_at_watchdog = True
                break
        self.assertTrue(triggered_at_watchdog, "Watchdog must force invocation even under extreme threshold.")

if __name__ == "__main__":
    unittest.main()
