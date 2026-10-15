import threading
import os
import time
import heapq
from collections import deque
from typing import Tuple, Optional, List, Dict, Any

class AudioQueue:
    """
    Thread-safe Priority-Weighted Audio Queue with Urgent Interruption and Preemption Support.
    
    Implements Eq. (11) from the SightLine manuscript:
        Priority(o_i) = w1 * f_class(o_i) + w2 * f_prox(o_i) + w3 * f_nov(o_i) + w4 * f_urg(o_i)
    
    When an urgent collision hazard arrives (Priority >= 0.80 or is_urgent=True),
    the queue triggers immediate preemption (interruption) of lower-priority descriptive
    speech, bounding alert delivery latency under sensory contention.
    """

    def __init__(self, max_size: int = 5, urgency_threshold: float = 0.80):
        self._lock = threading.Lock()
        self._counter = 0
        self._max_size = max_size
        self.urgency_threshold = urgency_threshold
        self._current_audio: Optional[str] = None
        self._current_priority: float = 0.0
        self._interrupted: bool = False
        
        # Priority heap: elements are (-priority, counter, text, audio_path, is_urgent, timestamp)
        self._priority_heap: List[Tuple[float, int, str, str, bool, float]] = []
        
        # Standard FIFO deque for backward compatibility
        self._queue: deque[Tuple[str, str]] = deque()

    @staticmethod
    def calculate_priority(
        is_hazard: bool = False,
        proximity_score: float = 0.5,
        novelty_score: float = 1.0,
        urgency_score: float = 0.5,
        weights: Tuple[float, float, float, float] = (0.50, 0.20, 0.15, 0.15)
    ) -> float:
        """
        Calculate scalar priority score using Equation 11.
        w = [0.50, 0.20, 0.15, 0.15] for [Class, Proximity, Novelty, Urgency].
        """
        w1, w2, w3, w4 = weights
        f_class = 1.0 if is_hazard else 0.20
        p = w1 * f_class + w2 * proximity_score + w3 * novelty_score + w4 * urgency_score
        return float(np.clip(p, 0.0, 1.0)) if 'np' in globals() else max(0.0, min(1.0, p))

    def enqueue(
        self,
        text: str,
        audio_path: str,
        priority: float = 0.50,
        is_urgent: bool = False
    ) -> Optional[str]:
        """
        Add audio item with priority scoring. If item is urgent, preempts lower-priority items.
        """
        with self._lock:
            now = time.time()
            self._counter += 1
            
            # Urgent Preemption Logic
            if is_urgent or priority >= self.urgency_threshold:
                if self._current_priority < priority and self._current_audio:
                    # Preempt currently playing speech
                    self._interrupted = True
                    self._safe_delete(self._current_audio)
                    self._current_audio = None
                
                # Push with maximum negative priority to head of queue
                heapq.heappush(self._priority_heap, (-priority, self._counter, text, audio_path, is_urgent, now))
                self._queue.appendleft((text, audio_path))
            else:
                if len(self._priority_heap) >= self._max_size:
                    # Pop lowest priority item
                    # In python heapq, largest (-priority) has lowest actual priority
                    self._evict_lowest_priority()
                    
                heapq.heappush(self._priority_heap, (-priority, self._counter, text, audio_path, is_urgent, now))
                self._queue.append((text, audio_path))
                
            return audio_path

    def dequeue(self) -> Optional[Tuple[str, str]]:
        """Retrieve highest-priority audio item."""
        with self._lock:
            if self._priority_heap:
                neg_p, cnt, text, audio_path, is_urgent, ts = heapq.heappop(self._priority_heap)
                self._current_audio = audio_path
                self._current_priority = -neg_p
                self._interrupted = False
                
                # Sync standard queue representation
                self._queue = deque([(t, a) for _, _, t, a, _, _ in sorted(self._priority_heap)])
                return (text, audio_path)
            elif self._queue:
                item = self._queue.popleft()
                self._current_audio = item[1]
                return item
            return None

    def _evict_lowest_priority(self):
        """Evicts the lowest priority item when queue capacity is reached."""
        if not self._priority_heap:
            return
        # Find item with smallest priority (largest neg_p)
        worst_idx = max(range(len(self._priority_heap)), key=lambda i: self._priority_heap[i][0])
        evicted = self._priority_heap.pop(worst_idx)
        heapq.heapify(self._priority_heap)
        self._safe_delete(evicted[3])
        self._queue = deque([(t, a) for _, _, t, a, _, _ in sorted(self._priority_heap)])

    def clear(self):
        """Clear all queued audio and delete temporary files."""
        with self._lock:
            for _, _, _, path, _, _ in self._priority_heap:
                self._safe_delete(path)
            self._priority_heap.clear()
            self._queue.clear()
            self._current_audio = None
            self._current_priority = 0.0
            self._interrupted = False

    def interrupt(self):
        """Interrupt current playback immediately and purge non-urgent queue."""
        self.clear()

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._priority_heap) == 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._priority_heap)

    @staticmethod
    def _safe_delete(path: str):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
