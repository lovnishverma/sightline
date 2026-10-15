import unittest
import sys
import os
import time

sys.path.append(r"C:\Users\princ\Desktop\SightLine")
from src.speech.audio_manager import AudioQueue

class TestPriorityAudioQueue(unittest.TestCase):
    def setUp(self):
        self.queue = AudioQueue(max_size=5, urgency_threshold=0.80)

    def test_priority_ordering(self):
        """Lower priority items should yield to higher priority items."""
        self.queue.enqueue("OCR Text", "path_ocr.wav", priority=0.30)
        self.queue.enqueue("General Scenery", "path_scene.wav", priority=0.55)
        self.queue.enqueue("Approaching Bicycle", "path_bike.wav", priority=0.75)
        
        # Dequeue should return highest priority first (bike -> scene -> ocr)
        t1, _ = self.queue.dequeue()
        self.assertEqual(t1, "Approaching Bicycle")
        t2, _ = self.queue.dequeue()
        self.assertEqual(t2, "General Scenery")
        t3, _ = self.queue.dequeue()
        self.assertEqual(t3, "OCR Text")

    def test_urgent_hazard_preemption(self):
        """Urgent hazard alert (priority >= 0.80) should preempt lower-priority playback."""
        self.queue.enqueue("Reading long document paragraph...", "doc.wav", priority=0.25)
        item = self.queue.dequeue()
        self.assertEqual(item[0], "Reading long document paragraph...")
        self.assertEqual(self.queue._current_audio, "doc.wav")
        self.assertFalse(self.queue._interrupted)
        
        # Now an urgent obstacle enters
        self.queue.enqueue("WARNING: Fast-moving vehicle approaching!", "car.wav", priority=0.95, is_urgent=True)
        self.assertTrue(self.queue._interrupted, "Urgent alert must preempt current lower-priority playback")
        
        # Dequeue must immediately deliver the urgent alert
        urgent_item = self.queue.dequeue()
        self.assertEqual(urgent_item[0], "WARNING: Fast-moving vehicle approaching!")

    def test_priority_calculation_eq11(self):
        """Verify priority calculation adheres to Equation 11 weights [0.50, 0.20, 0.15, 0.15]."""
        p_hazard = AudioQueue.calculate_priority(is_hazard=True, proximity_score=0.9, novelty_score=1.0, urgency_score=0.9)
        # 0.50*1.0 + 0.20*0.9 + 0.15*1.0 + 0.15*0.9 = 0.50 + 0.18 + 0.15 + 0.135 = 0.965
        self.assertAlmostEqual(p_hazard, 0.965, places=3)
        self.assertGreaterEqual(p_hazard, 0.80)

        p_ocr = AudioQueue.calculate_priority(is_hazard=False, proximity_score=0.2, novelty_score=0.5, urgency_score=0.1)
        # 0.50*0.20 + 0.20*0.2 + 0.15*0.5 + 0.15*0.1 = 0.10 + 0.04 + 0.075 + 0.015 = 0.230
        self.assertAlmostEqual(p_ocr, 0.230, places=3)
        self.assertLess(p_ocr, 0.50)

if __name__ == '__main__':
    unittest.main()
