"""Test emotion pipeline configuration fields."""

from forgestream.config import ForgeStreamConfig


def test_emotion_config_defaults():
    config = ForgeStreamConfig()
    assert config.emotion_enabled is True
    assert config.emotion_window_seconds == 3.0
    assert config.emotion_stride_seconds == 1.0
    assert config.emotion_ml_enabled is False
    assert config.emotion_buffer_seconds == 30.0
