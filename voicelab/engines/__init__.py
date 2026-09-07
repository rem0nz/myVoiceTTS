from .base import Engine, TTSEngine, VCEngine, Audio
from .system import SystemEngine
from .kokoro import KokoroEngine
from .piper import PiperEngine

__all__ = ["Engine", "TTSEngine", "VCEngine", "Audio",
           "SystemEngine", "KokoroEngine", "PiperEngine", "tts_engines"]


def tts_engines() -> dict:
    return {e.name: e for e in (KokoroEngine(), PiperEngine(), SystemEngine())}
