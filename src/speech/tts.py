import asyncio
import os
import threading
import tempfile
from typing import Optional
import edge_tts
from src.config import CONFIG

# Voice mappings for supported locales
VOICE_MAP = {
    "English (US) - Aria": "en-US-AriaNeural",
    "English (US) - Guy": "en-US-GuyNeural",
    "English (UK) - Sonia": "en-GB-SoniaNeural",
    "Hindi - Swara": "hi-IN-SwaraNeural",
    "Hindi - Madhur": "hi-IN-MadhurNeural",
}

def init_tts_loop() -> asyncio.AbstractEventLoop:
    """Create a dedicated event loop for TTS in a background thread."""
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_run, daemon=True).start()
    return loop

_TTS_LOOP = init_tts_loop()

def _generate_offline_tts(
    text: str,
    voice_name: str = "English (US) - Aria",
    fallback_text: Optional[str] = None
) -> Optional[str]:
    """Offline fallback using local system TTS (Windows SAPI5 / pyttsx3)."""
    try:
        import pyttsx3
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            path = f.name

        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        selected_voice = None
        has_hindi_voice = False

        if voices:
            selected_voice = voices[0].id
            for v in voices:
                v_name = (v.name or "").lower()
                v_id = (v.id or "").lower()
                if "hindi" in v_name or "hindi" in v_id or "kalpana" in v_name or "hemant" in v_name:
                    selected_voice = v.id
                    has_hindi_voice = True
                    break
                elif "guy" in voice_name.lower() or "david" in voice_name.lower():
                    if "david" in v_name:
                        selected_voice = v.id
                elif "sonia" in voice_name.lower() or "hazel" in voice_name.lower():
                    if "hazel" in v_name or "gb" in v_id:
                        selected_voice = v.id
                elif "aria" in voice_name.lower() or "zira" in voice_name.lower():
                    if "zira" in v_name:
                        selected_voice = v.id

            if selected_voice:
                engine.setProperty("voice", selected_voice)

        # If text is in Devanagari/non-ASCII but no Hindi SAPI voice is installed locally,
        # use fallback English text so SAPI doesn't produce an empty silent file.
        text_to_speak = text.strip()
        is_non_ascii = any(ord(c) > 127 for c in text_to_speak)
        if is_non_ascii and not has_hindi_voice and fallback_text and fallback_text.strip():
            text_to_speak = fallback_text.strip()

        engine.setProperty("rate", 175)
        engine.save_to_file(text_to_speak, path)
        engine.runAndWait()

        if os.path.exists(path) and os.path.getsize(path) > 200:
            return path
        return None
    except Exception as offline_err:
        print(f"Offline TTS fallback error: {offline_err}")
        return None

def text_to_speech(
    text: str,
    voice_name: str = "English (US) - Aria",
    fallback_text: Optional[str] = None
) -> Optional[str]:
    """Convert text to speech, returning the audio file path with automatic offline fallback."""
    if not text or not text.strip():
        return None

    try:
        voice_id = VOICE_MAP.get(voice_name, "en-US-AriaNeural")

        async def _generate():
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{CONFIG.AUDIO_FORMAT}") as f:
                path = f.name
            communicate = edge_tts.Communicate(
                text.strip(),
                voice=voice_id,
                rate=CONFIG.TTS_RATE,
            )
            await communicate.save(path)
            return path

        coro = _generate()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, _TTS_LOOP)
            result = future.result(timeout=CONFIG.TTS_TIMEOUT)
            if result and os.path.exists(result) and os.path.getsize(result) > 0:
                return result
            raise RuntimeError("Edge-TTS produced empty audio")
        except Exception:
            try:
                coro.close()
            except Exception:
                pass
            raise
    except Exception as e:
        print(f"Online TTS unavailable or failed ({e}). Falling back to local offline TTS...")
        return _generate_offline_tts(text, voice_name, fallback_text=fallback_text)
