from unittest.mock import patch, MagicMock

from forgestream.audio.microphone import MicrophoneSource
from forgestream.audio.system_audio import SystemAudioSource
from forgestream.audio.source import AudioSource


class TestMicrophoneSource:
    def test_is_audio_source(self):
        source = MicrophoneSource()
        assert isinstance(source, AudioSource)

    def test_default_device(self):
        source = MicrophoneSource()
        assert source.device is None

    def test_custom_device(self):
        source = MicrophoneSource(device=3)
        assert source.device == 3

    @patch("forgestream.audio.microphone.sd")
    def test_list_devices(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 1},
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        devices = MicrophoneSource.list_input_devices()
        assert len(devices) == 2
        assert devices[0]["name"] == "Built-in Microphone"

    @patch("forgestream.audio.microphone.sd")
    async def test_start_opens_stream(self, mock_sd):
        source = MicrophoneSource()
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        await source.start()
        assert source.is_active is True
        mock_sd.InputStream.assert_called_once()

    @patch("forgestream.audio.microphone.sd")
    async def test_stop_closes_stream(self, mock_sd):
        source = MicrophoneSource()
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        await source.start()
        await source.stop()
        assert source.is_active is False
        mock_stream.stop.assert_called_once()


class TestSystemAudioSource:
    def test_is_microphone_source(self):
        source = SystemAudioSource(device=0)
        assert isinstance(source, MicrophoneSource)

    @patch("forgestream.audio.system_audio.sd")
    def test_find_blackhole(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 1},
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        idx = SystemAudioSource._find_blackhole_device()
        assert idx == 1

    @patch("forgestream.audio.system_audio.sd")
    def test_is_available(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        assert SystemAudioSource.is_available() is True

    @patch("forgestream.audio.system_audio.sd")
    def test_not_available_without_blackhole(self, mock_sd):
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 1},
        ]
        assert SystemAudioSource.is_available() is False
