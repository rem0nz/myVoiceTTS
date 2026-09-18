from .base import Engine, TTSEngine, VCEngine, Audio
from .system import SystemEngine
from .kokoro import KokoroEngine
from .piper import PiperEngine
from .f5 import F5Engine

__all__ = ["Engine", "TTSEngine", "VCEngine", "Audio",
           "SystemEngine", "KokoroEngine", "PiperEngine", "F5Engine", "tts_engines"]


def tts_engines() -> dict:
    return {e.name: e for e in (KokoroEngine(), F5Engine(), PiperEngine(), SystemEngine())}
