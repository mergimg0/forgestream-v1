# BlackHole + Aggregate Device Setup Guide

ForgeStream uses BlackHole to capture system audio (Zoom/Meet/Teams calls) for real-time meeting intelligence.

## Step 1: Install BlackHole

```bash
brew install blackhole-2ch
```

Restart your Mac after installation (or at minimum, restart any audio applications).

## Step 2: Create an Aggregate Device

This lets you hear audio AND capture it simultaneously.

1. Open **Audio MIDI Setup** (search in Spotlight or find in `/Applications/Utilities/`)
2. Click the **+** button in the bottom-left corner
3. Select **Create Aggregate Device**
4. Check both:
   - **Built-in Microphone** (your mic — captures your voice)
   - **BlackHole 2ch** (virtual device — captures system audio)
5. Rename it to **ForgeStream Input** (double-click the name)

## Step 3: Create a Multi-Output Device

This routes audio to both your speakers AND BlackHole.

1. In Audio MIDI Setup, click **+** again
2. Select **Create Multi-Output Device**
3. Check both:
   - **Built-in Output** (your speakers/headphones)
   - **BlackHole 2ch**
4. Rename it to **ForgeStream Output**
5. Make sure **Built-in Output** is the **Master Device** (click the dropdown)

## Step 4: Configure Your Meeting App

For Zoom/Meet/Teams:
1. In the meeting app's audio settings:
   - **Speaker/Output**: Select **ForgeStream Output**
   - **Microphone/Input**: Keep your normal microphone
2. This routes remote participants' audio through BlackHole while you hear it normally

## Step 5: Run ForgeStream

```bash
cd ~/projects/forgestream
python3 -m forgestream.runner /path/to/audio --mode collaborative
```

ForgeStream automatically detects BlackHole and captures from it. To verify:

```python
from forgestream.audio.system_audio import SystemAudioSource
print("BlackHole available:", SystemAudioSource.is_available())
print("Device info:", SystemAudioSource.get_device_info())
```

## Step 6: For Full Meeting Capture (Both Sides)

To capture BOTH your voice AND remote participants in one stream:
- Set ForgeStream to use the **ForgeStream Input** aggregate device
- This combines your mic + BlackHole into one input

```python
from forgestream.audio.microphone import MicrophoneSource
devices = MicrophoneSource.list_input_devices()
# Find "ForgeStream Input" in the list and use its index
```

## Troubleshooting

- **No audio captured**: Make sure the meeting app output is set to ForgeStream Output
- **Echo/feedback**: Don't set the meeting app's mic to BlackHole — only the output
- **BlackHole not listed**: Restart your Mac after installation
- **Low volume**: In Audio MIDI Setup, check that BlackHole 2ch volume isn't muted
