import urllib.request
import urllib.parse
import json

def translate_to_hindi(text: str) -> str:
    """
    Translates text to Hindi using multiple robust fallback strategies:
    1. Direct Google Translate API (fast, reliable, no scraping blocks)
    2. MyMemoryTranslator (deep_translator)
    3. GoogleTranslator (deep_translator)
    """
    if not text or not text.strip():
        return text

    clean_text = text.strip()

    # Tier 1: Direct Google Translate API (Googleapis GTX endpoint)
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=hi&dt=t&q=" + urllib.parse.quote(clean_text)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = "".join([item[0] for item in data[0] if item and item[0]])
            if translated and translated.strip():
                return translated.strip()
    except Exception as e:
        # Fall back to next tier
        pass

    # Tier 2: MyMemoryTranslator
    try:
        from deep_translator import MyMemoryTranslator
        translated = MyMemoryTranslator(source="en-US", target="hi-IN").translate(clean_text)
        if translated and "error" not in translated.lower() and "500" not in translated:
            return translated.strip()
    except Exception:
        pass

    # Tier 3: GoogleTranslator (deep-translator)
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="hi").translate(clean_text)
        if translated and "error" not in translated.lower() and "500" not in translated:
            return translated.strip()
    except Exception:
        pass

    return clean_text
