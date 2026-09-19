import pyaudio


def create_stream(
    pyaudio_manager: pyaudio.PyAudio,
    sample_rate: int = 16_000,
    frames_per_buffer: int = 160,
) -> pyaudio.Stream:
    return pyaudio_manager.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )


def record_audio(stream: pyaudio.Stream, duration_seconds: float = 0.1) -> bytes:
    frames_to_read = int(stream._rate * duration_seconds)
    byte_frames = stream.read(frames_to_read)
    return byte_frames
