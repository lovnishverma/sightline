"""
Unit test suite for SightLine Algorithm 1 (Hazard-State & Watchdog Gating).
Covers all six critical state transition cases required by the reviewer audit:
1. First-time hazard detection (z: 0 -> 1, c_alert reset to 0, cooldown engaged)
2. Hazard persistence across multiple inferences (throttled re-triggers every T_cool frames)
3. Hazard disappears and subsequently reappears (clear on false, re-arm on true)
4. New hazard appears during cooldown interval (cooldown reset on confirmed detection)
5. Watchdog fires while hazard is already latched (watchdog force, reference refresh)
6. Visual motion override fires during cooldown (S_mot >= tau bypasses cooldown immediately)
"""

import unittest
import numpy as np
from PIL import Image
import sys
import os

# Add SightLine src to path
sys.path.append(r"C:\Users\princ\Desktop\SightLine")
from src.vision.priority_gate import PrioritySceneGate

def make_dummy_image(val=128):
    arr = np.full((120, 160, 3), val, dtype=np.uint8)
    return Image.fromarray(arr)

class TestAlgorithm1StateTransitions(unittest.TestCase):
    def setUp(self):
        # Default profile: alpha=0.25, beta=0.20, gamma=0.30, delta=0.25, tau=0.18
        # Watchdog T_max=15, Cooldown T_cool=6
        self.gate = PrioritySceneGate(
            alpha=0.25,
            beta=0.20,
            gamma=0.30,
            delta=0.25,
            threshold=0.18,
            heartbeat_interval=15,
            cooldown_interval=6,
            hazard_vocabulary={'car', 'obstacle'}
        )
        # Process startup frame t=0 (unconditional invocation)
        img0 = make_dummy_image(128)
        invoke0, score0, _ = self.gate.evaluate_change(img0, detection_labels=[])
        self.assertTrue(invoke0, "Startup frame t=0 must invoke unconditionally")
        self.assertEqual(self.gate.cached_hazard, 0.0)
        self.assertEqual(self.gate.frames_since_hazard_alert, float('inf'))
        self.assertEqual(self.gate.frames_since_invocation, 0)

    def test_case_1_hazard_detected_for_first_time(self):
        """Case 1: A hazard is detected for the first time."""
        # Frame 1: Introduce visual change so gate fires and VLM detects hazard
        img1 = make_dummy_image(200) # Causes hash and motion change
        invoke, score, bd = self.gate.evaluate_change(img1, detection_labels=['car'])
        self.assertTrue(invoke, "Frame with significant change must invoke")
        self.assertEqual(self.gate.cached_hazard, 1.0, "Hazard state must be asserted (z=1)")
        self.assertEqual(self.gate.frames_since_hazard_alert, 0, "Alert counter must reset to 0 upon detection")
        
        # Frame 2: Identical image (no visual change). Cooldown must suppress hazard term
        invoke2, score2, bd2 = self.gate.evaluate_change(img1, detection_labels=None)
        self.assertEqual(self.gate.frames_since_hazard_alert, 1)
        self.assertEqual(bd2['s_haz'], 0.0, "Hazard score must be suppressed to 0 during cooldown")
        self.assertFalse(invoke2, "Frame 2 must be suppressed during cooldown")

    def test_case_2_hazard_remains_present_across_multiple_inferences(self):
        """Case 2: Hazard remains present across multiple inferences (throttled every T_cool frames)."""
        img = make_dummy_image(200)
        # Arm hazard at frame 1
        self.gate.evaluate_change(img, detection_labels=['car'])
        self.assertEqual(self.gate.cached_hazard, 1.0)
        
        # Frames 2 to 6 (5 frames of cooldown): must be suppressed
        for f in range(2, 7):
            inv, sc, bd = self.gate.evaluate_change(img, detection_labels=None)
            self.assertFalse(inv, f"Frame {f} should be suppressed (c_alert={self.gate.frames_since_hazard_alert})")
            self.assertEqual(bd['s_haz'], 0.0)
            
        # Frame 7: c_alert reaches 6 (= T_cool). Cooldown expires!
        # s_haz restores to 1.0. Since delta=0.25 >= tau=0.18, gate must trigger!
        inv7, sc7, bd7 = self.gate.evaluate_change(img, detection_labels=['car'])
        self.assertTrue(inv7, "Frame 7 must invoke as cooldown expires")
        self.assertEqual(bd7['s_haz'], 1.0, "Hazard term must be active when cooldown expires")
        self.assertEqual(self.gate.frames_since_hazard_alert, 0, "c_alert must restart at 0 for next cooldown cycle")
        
        # Frame 8: next cooldown cycle begins
        inv8, sc8, bd8 = self.gate.evaluate_change(img, detection_labels=None)
        self.assertFalse(inv8, "Frame 8 must be suppressed under new cooldown cycle")
        self.assertEqual(bd8['s_haz'], 0.0)

    def test_case_3_hazard_disappears_and_reappears(self):
        """Case 3: Hazard disappears and subsequently reappears."""
        img = make_dummy_image(200)
        # Arm hazard at frame 1
        self.gate.evaluate_change(img, detection_labels=['car'])
        self.assertEqual(self.gate.cached_hazard, 1.0)
        
        # Advance through cooldown to frame 7
        for _ in range(5):
            self.gate.evaluate_change(img, detection_labels=None)
            
        # At frame 7, VLM runs and reports NO hazard (car left scene)
        inv7, _, _ = self.gate.evaluate_change(img, detection_labels=[])
        self.assertTrue(inv7)
        self.assertEqual(self.gate.cached_hazard, 0.0, "Hazard flag must clear to 0.0")
        self.assertEqual(self.gate.frames_since_hazard_alert, float('inf'), "c_alert must reset to infinity")
        
        # Frame 8: No hazard, no change -> suppressed
        inv8, _, bd8 = self.gate.evaluate_change(img, detection_labels=None)
        self.assertFalse(inv8)
        self.assertEqual(bd8['s_haz'], 0.0)
        
        # Frame 9: New visual event occurs and VLM reports hazard reappearance
        img_new = make_dummy_image(50)
        inv9, _, _ = self.gate.evaluate_change(img_new, detection_labels=['car'])
        self.assertTrue(inv9)
        self.assertEqual(self.gate.cached_hazard, 1.0, "Hazard flag must re-arm to 1.0")
        self.assertEqual(self.gate.frames_since_hazard_alert, 0, "c_alert must reset to 0")

    def test_case_4_new_hazard_appears_during_cooldown(self):
        """Case 4: Visual event occurs during cooldown and VLM confirms hazard."""
        img1 = make_dummy_image(200)
        # Arm hazard at frame 1
        self.gate.evaluate_change(img1, detection_labels=['car'])
        
        # Frame 2: cooldown active (c_alert=1)
        self.gate.evaluate_change(img1, detection_labels=None)
        self.assertEqual(self.gate.frames_since_hazard_alert, 1)
        
        # Frame 3: Dramatic visual shift (e.g. second vehicle enters) causes visual cues to cross tau
        img2 = make_dummy_image(30)
        inv3, sc3, bd3 = self.gate.evaluate_change(img2, detection_labels=['obstacle'])
        self.assertTrue(inv3, "Dramatic visual change must trigger inference even during cooldown")
        self.assertEqual(self.gate.cached_hazard, 1.0)
        # Alert counter should reset to 0 upon confirmed detection
        self.assertEqual(self.gate.frames_since_hazard_alert, 0, "c_alert must reset to 0 after confirmed VLM pass")

    def test_case_5_watchdog_fires_while_hazard_latched(self):
        """Case 5: Watchdog fires while hazard is already latched."""
        # Use T_max = 8, T_cool = 12 (so watchdog fires BEFORE cooldown expires)
        custom_gate = PrioritySceneGate(
            alpha=0.25, beta=0.20, gamma=0.30, delta=0.25, threshold=0.18,
            heartbeat_interval=8, cooldown_interval=12
        )
        img = make_dummy_image(128)
        custom_gate.evaluate_change(img, detection_labels=['car'])
        
        # Advance 7 frames without visual change
        for _ in range(7):
            inv, _, bd = custom_gate.evaluate_change(img, detection_labels=None)
            self.assertFalse(inv)
            self.assertEqual(bd['s_haz'], 0.0)
            
        # At 8th frame, c_idle = 8 reaches T_max=8. Watchdog MUST force inference!
        inv_wd, _, bd_wd = custom_gate.evaluate_change(img, detection_labels=['car'])
        self.assertTrue(inv_wd, "Watchdog heartbeat must force invocation")
        self.assertTrue(bd_wd['heartbeat_forced'], "Heartbeat flag must be True")
        self.assertEqual(custom_gate.frames_since_invocation, 0, "Idle frame counter must reset to 0")

    def test_case_6_motion_override_during_cooldown(self):
        """Case 6: Visual motion override fires during cooldown (S_mot >= tau)."""
        img1 = make_dummy_image(100)
        self.gate.evaluate_change(img1, detection_labels=['car'])
        self.assertEqual(self.gate.cached_hazard, 1.0)
        
        # Frame 2: small jitter (S_mot < tau), cooldown suppresses
        img_jitter = make_dummy_image(105)
        inv2, _, bd2 = self.gate.evaluate_change(img_jitter, detection_labels=None)
        self.assertFalse(inv2)
        self.assertEqual(bd2['s_haz'], 0.0)
        
        # Frame 3: Rapid obstacle closing creates large motion (S_mot >= tau=0.18)
        # Gray value jump from 100 to 180 on thumbnail: |180 - 100| / 255 = 0.313 >= 0.18
        img_motion = make_dummy_image(180)
        inv3, sc3, bd3 = self.gate.evaluate_change(img_motion, detection_labels=['car'])
        self.assertTrue(inv3, "Motion override must trigger immediate invocation")
        self.assertGreaterEqual(bd3['s_mot'], 0.18, "Motion score must cross threshold")
        self.assertEqual(bd3['s_haz'], 1.0, "Motion override must bypass cooldown and restore s_haz=1.0")

if __name__ == '__main__':
    unittest.main()
