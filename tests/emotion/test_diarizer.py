"""Tests for SpeakerDiarizer — mocked pyannote-audio."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from forgestream.emotion.diarizer import SpeakerDiarizer


class TestSpeakerDiarizer:
    def test_label_silence_returns_unknown(self):
        diarizer = SpeakerDiarizer(huggingface_token="fake")
        # No chunks added, no diarization run
        assert diarizer.get_current_speaker() == "unknown"

    def test_get_speaker_at_without_diarization(self):
        diarizer = SpeakerDiarizer(huggingface_token="fake")
        assert diarizer.get_speaker_at(1.0) == "unknown"

    def test_add_chunk_accumulates_audio(self):
        diarizer = SpeakerDiarizer(huggingface_token="fake")
        chunk = b"\x00\x01" * 8000  # 0.5s of PCM
        diarizer.add_chunk(chunk)
        assert diarizer._write_pos == 8000  # 8000 float32 samples from 16000 bytes

    def test_buffer_wraps_correctly(self):
        diarizer = SpeakerDiarizer(
            huggingface_token="fake", window_seconds=1.0
        )
        # Buffer holds 16000 float32 samples (1 second at 16kHz)
        chunk = b"\x00\x01" * 8000  # 0.5s = 8000 samples
        diarizer.add_chunk(chunk)
        diarizer.add_chunk(chunk)
        diarizer.add_chunk(chunk)  # third chunk should cause wrap
        assert diarizer._total_samples == 24000

    def test_is_available_static(self):
        # Just checks the interface exists
        result = SpeakerDiarizer.is_available()
        assert isinstance(result, bool)

    @patch("forgestream.emotion.diarizer.SpeakerDiarizer._ensure_pipeline")
    def test_update_diarization_with_mock_pipeline(self, mock_ensure):
        # Mock torch at module level
        import sys
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.float32 = "float32"
        sys.modules["torch"] = mock_torch

        try:
            diarizer = SpeakerDiarizer(huggingface_token="fake")

            # Mock the pipeline to return segments
            mock_pipeline = MagicMock()
            mock_turn_1 = MagicMock()
            mock_turn_1.start = 0.0
            mock_turn_1.end = 2.5
            mock_turn_2 = MagicMock()
            mock_turn_2.start = 2.5
            mock_turn_2.end = 5.0
            mock_pipeline.return_value.itertracks.return_value = [
                (mock_turn_1, None, "SPEAKER_00"),
                (mock_turn_2, None, "SPEAKER_01"),
            ]
            diarizer._pipeline = mock_pipeline

            # Add enough audio
            chunk = b"\x00\x01" * 8000
            for _ in range(10):
                diarizer.add_chunk(chunk)

            diarizer.update_diarization()
            assert len(diarizer._segments) == 2
            assert diarizer._segments[0][2] == "SPEAKER_00"
            assert diarizer._segments[1][2] == "SPEAKER_01"
        finally:
            del sys.modules["torch"]

    @patch("forgestream.emotion.diarizer.SpeakerDiarizer._ensure_pipeline")
    def test_get_speaker_at_returns_correct_speaker(self, mock_ensure):
        diarizer = SpeakerDiarizer(huggingface_token="fake")
        # Manually set segments (simulating diarization result)
        diarizer._segments = [
            (0.0, 2.5, "SPEAKER_00"),
            (2.5, 5.0, "SPEAKER_01"),
        ]
        diarizer._total_samples = 80000  # 5 seconds
        diarizer._write_pos = 80000

        assert diarizer.get_speaker_at(1.0) == "SPEAKER_00"
        assert diarizer.get_speaker_at(3.0) == "SPEAKER_01"
        assert diarizer.get_speaker_at(6.0) == "unknown"

    @patch("forgestream.emotion.diarizer.SpeakerDiarizer._ensure_pipeline")
    def test_get_current_speaker(self, mock_ensure):
        diarizer = SpeakerDiarizer(huggingface_token="fake")
        diarizer._segments = [
            (0.0, 2.5, "SPEAKER_00"),
            (2.5, 5.0, "SPEAKER_01"),
        ]
        assert diarizer.get_current_speaker() == "SPEAKER_01"
