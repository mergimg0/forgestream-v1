"""SystemAudioSource -- capture system audio via BlackHole virtual device."""

from __future__ import annotations

from typing import Any

from .microphone import MicrophoneSource

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore
    HAS_SOUNDDEVICE = False


class SystemAudioSource(MicrophoneSource):
    """Captures system audio output via BlackHole virtual audio device.

    BlackHole must be installed: brew install blackhole-2ch

    For full meeting capture (your voice + remote participants):
    1. Open Audio MIDI Setup
    2. Create Aggregate Device combining mic + BlackHole
    3. Use that aggregate device as the input
    """

    BLACKHOLE_NAMES = ["BlackHole 2ch", "BlackHole 16ch", "BlackHole"]

    def __init__(self, device: int | None = None) -> None:
        if device is None:
            device = self._find_blackhole_device()
        super().__init__(device=device)

    @classmethod
    def _find_blackhole_device(cls) -> int | None:
        """Find the BlackHole device index."""
        if not HAS_SOUNDDEVICE:
            return None
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                for name in cls.BLACKHOLE_NAMES:
                    if name.lower() in d["name"].lower():
                        return i
        return None

    @classmethod
    def is_available(cls) -> bool:
        """Check if BlackHole is installed and available."""
        return cls._find_blackhole_device() is not None

    @classmethod
    def get_device_info(cls) -> dict[str, Any] | None:
        """Get BlackHole device information."""
        if not HAS_SOUNDDEVICE:
            return None
        idx = cls._find_blackhole_device()
        if idx is not None:
            info = sd.query_devices(idx)
            return {"index": idx, "name": info["name"], "channels": info["max_input_channels"]}
        return None
