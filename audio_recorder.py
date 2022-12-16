import pyaudio
import wave
import numpy as np

class AudioRecorder():
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.frames = []  # buffer to store recording
        self.channels = 1
        self.sr = 22050
        self.frames_per_buffer = 882

    def clear(self):
        self.frames = []

    def start(self, seconds):
        if len(self.frames) != 0:
            self.clear()
        print('Recording...')
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=self.channels,
                             rate=self.sr,
                             frames_per_buffer=self.frames_per_buffer,
                             input=True)
        k = int(self.sr * seconds / self.frames_per_buffer)
        for i in range(k):
            data = stream.read(self.frames_per_buffer)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()
        print('Done recording')

    def save(self, filename):
        if len(self.frames) == 0:
            return
        print('Saving...')
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sr)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print('File %s saved' % filename)

    def get_npdata(self):
        return np.frombuffer(b''.join(self.frames), dtype=np.int16)
