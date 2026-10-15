from src.vision.florence import FlorenceVisionEngine
from src.conversation.context import CONTEXT
from src.speech.tts import text_to_speech
from src.speech.audio_manager import AudioQueue
from src.vision.scene_change import compute_hash, hash_distance
from src.vision.priority_gate import PrioritySceneGate
from src.config import CONFIG
from PIL import Image

AUDIO_QUEUE = AudioQueue(max_size=CONFIG.MAX_QUEUE_SIZE)

class SightLineAssistant:
    """Orchestrates vision, context, and speech."""
    
    def __init__(self):
        self.vision = FlorenceVisionEngine()
        self.audio_finish_time = 0.0
        # Initialize multi-factor PrioritySceneGate with adaptive two-tier watchdog & ROI hazard awareness
        self.gate = PrioritySceneGate(
            alpha=0.25, beta=0.20, gamma=0.30, delta=0.25,
            threshold=0.18, heartbeat_interval=60,
            cooldown_interval=6,
            adaptive_watchdog=True,
            t_ambient=60, t_active=6,
            use_2d_hash=True,
            use_roi_motion=True
        )
        
    def initialize(self):
        self.vision.load()
        
    def process_image(self, image, task: str, voice_name: str, force: bool = False, question: str = ""):
        """Process an image and generate a response."""
        if self.vision.model is None:
            return "Model is initializing, please wait...", None, "Model Loading..."
            
        import time
        if not force and time.time() < getattr(self, "audio_finish_time", 0.0):
            return None, None, "Speaking..."
            
        import numpy as np
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        img_hash = compute_hash(image)
        
        # Map human-readable tasks to internal tokens
        task_map = {
            "Quick Glance": "<CAPTION>",
            "Detailed Scene": "<DETAILED_CAPTION>",
            "Immersive Description": "<MORE_DETAILED_CAPTION>",
            "Read Text": "<OCR>",
            "Ask Question": "<VQA>"
        }
        internal_task = task_map.get(task, "<DETAILED_CAPTION>")
        
        # Multi-factor PrioritySceneGate evaluation
        should_invoke, change_score, breakdown = self.gate.evaluate_change(image)
        
        # Debounce/Gating check if not forced
        if not force and not should_invoke and internal_task != "<VQA>" and CONTEXT.last_text != "":
            text, audio = CONTEXT.get_last()
            return text, audio, f"Gate suppressed (change: {change_score:.3f})"
            
        # Inference based on task
        if internal_task == "<OCR>":
            response = self.vision.read_text(image)
        elif internal_task == "<VQA>":
            if not question or not question.strip():
                response = "Please type a question in the box to use this mode."
            else:
                response = self.vision.ask_question(image, question.strip())
        elif internal_task == "<MORE_DETAILED_CAPTION>":
            response = self.vision.describe_scene(image, detailed=True)
        else:
            response = self.vision.describe_scene(image, detailed=False)
        # Hyper-Local Translation for Hindi voices
        if "Hindi" in voice_name:
            try:
                from deep_translator import GoogleTranslator, MyMemoryTranslator
                translated = GoogleTranslator(source='auto', target='hi').translate(response)
                
                if not translated or "Error 500" in translated or "Server Error" in translated:
                    # Fallback to MyMemory API if Google blocks the scraper
                    translated = MyMemoryTranslator(source='en-US', target='hi-IN').translate(response)

                if translated and "Error 500" not in translated:
                    response = translated
                else:
                    print("Translation API failed or rate-limited. Falling back to English.")
                    voice_name = "English (US) - Aria"
            except Exception as e:
                print(f"Translation error: {e}. Falling back to English.")
                voice_name = "English (US) - Aria"

        # TTS
        audio_path = text_to_speech(response, voice_name)
        
        if audio_path:
            import time
            try:
                from pydub import AudioSegment
                duration = AudioSegment.from_file(audio_path).duration_seconds
            except Exception as e:
                print(f"Duration error: {e}")
                duration = len(response) / 15.0  # Fallback rough estimate
            self.audio_finish_time = time.time() + duration

        # Extract detected words/entities to update causal hazard state
        import re
        words = re.findall(r'\b[a-zA-Z]+\b', response.lower())
        self.gate.update_hazard_state(words)
        has_hazard = any(w in self.gate.hazard_vocabulary for w in words)

        # Update context
        CONTEXT.update(img_hash, task, response, audio_path)
        
        # Enqueue audio with priority scoring and preemption (Eq. 11)
        if audio_path:
            p_score = AUDIO_QUEUE.calculate_priority(
                is_hazard=has_hazard,
                proximity_score=0.90 if has_hazard else 0.40,
                urgency_score=0.95 if has_hazard else 0.30
            )
            AUDIO_QUEUE.enqueue(response, audio_path, priority=p_score, is_urgent=has_hazard)
            
        return response, audio_path, f"Processed task: {task}"
