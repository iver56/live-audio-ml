"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""
import math
import struct
import wave

import pyaudio

SAMPLES_PER_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 50
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

BYTES_PER_SAMPLE = p.get_sample_size(FORMAT)
BIT_DEPTH = BYTES_PER_SAMPLE * 8

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=SAMPLES_PER_CHUNK
)

frames = []
for i in range(0, int(RATE / SAMPLES_PER_CHUNK * RECORD_SECONDS)):
    data = stream.read(SAMPLES_PER_CHUNK)

    values = []
    for j in range(SAMPLES_PER_CHUNK):
        byte_index = j * BYTES_PER_SAMPLE
        those_bytes = data[byte_index:byte_index + BYTES_PER_SAMPLE]
        unpacked_int = struct.unpack("<h", those_bytes)[0]
        value = unpacked_int / 32767  # between -1 and 1
        values.append(value)

    rms = math.sqrt(
        sum(value ** 2 for value in values) / SAMPLES_PER_CHUNK
    )
    b = (" " + "#" * int(rms * 79)).ljust(80)
    print(b, end="\r")

    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(BYTES_PER_SAMPLE)
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
