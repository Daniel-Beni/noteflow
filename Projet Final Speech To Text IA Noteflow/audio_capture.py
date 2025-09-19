import pyaudio
import pkg_resources
import webrtcvad
import queue
from collections import deque
import numpy as np

# ==============================
# Paramètres généraux
# ==============================
RATE = 16000                  # fréquence d'échantillonnage (16 kHz = requis par Whisper)
CHANNELS = 1                  # mono
FORMAT = pyaudio.paInt16      # PCM 16 bits
FRAME_MS = 30                 # taille des frames analysées par VAD (10, 20 ou 30 ms)
FRAME_BYTES = int(RATE * (FRAME_MS / 1000.0)) * 2  # nombre d’octets par frame (int16 → 2 octets)

# ==============================
# Paramètres VAD (Voice Activity Detection)
# ==============================
AGGRESSIVENESS = 1            # 0 = tolérant, 3 = très strict (on coupe vite le bruit)
RING_MS = 500                 # fenêtre de décision (ms)
RING_FRAMES = RING_MS // FRAME_MS

START_SPEECH_RATIO = 0.6      # proportion de frames "voix" dans la fenêtre → déclenche un segment
STOP_SILENCE_RATIO = 0.95     # proportion de frames "non-voix" dans la fenêtre → termine un segment

# Taille minimale d’un segment valide avant de l’envoyer à Whisper
MIN_SEGMENT_SIZE = 4000       # ~0.25s d’audio à 16kHz (16k * 0.25 * 2 bytes ≈ 8000, ici en bytes bruts)


# ==============================
# Capture micro (PyAudio)
# ==============================
class MicStream:
    """
    Capture des frames audio depuis le micro en temps réel.
    Utilise PyAudio en mode callback pour remplir une queue.
    """
    def __init__(self, rate=RATE, channels=CHANNELS, frames_per_buffer=int(RATE * FRAME_MS / 1000.0)):
        self.pa = pyaudio.PyAudio()
        self.rate = rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.stream = None
        self.q = queue.Queue()

    def start(self, device_index=None):
        """Démarre le flux micro en callback → remplit la queue"""
        def _callback(in_data, frame_count, time_info, status):
            self.q.put(in_data)
            return (None, pyaudio.paContinue)

        self.stream = self.pa.open(format=FORMAT,
                                channels=self.channels,
                                rate=self.rate,
                                input=True,
                                input_device_index=device_index,
                                frames_per_buffer=self.frames_per_buffer,
                                stream_callback=_callback)
        self.stream.start_stream()
        return self

    def frames(self):
        """Itérateur qui fournit des frames audio brutes depuis la queue"""
        while True:
            try:
                yield self.q.get(timeout=0.1)
            except queue.Empty:
                continue

    def stop(self):
        """Arrête proprement le micro"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


# ==============================
# Détection de segments de parole (VAD)
# ==============================
class VADSegmenter:
    """
    Détecte la parole dans le flux audio en utilisant WebRTC VAD.
    Retourne des segments audio complets (paroles entre deux silences).
    """
    def __init__(self, rate=RATE, aggressiveness=AGGRESSIVENESS):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.rate = rate
        self.ring = deque(maxlen=RING_FRAMES)  # tampon circulaire pour décider start/stop
        self.triggered = False                 # état : en segment ou pas
        self.voiced_buffer = []                # stockage temporaire des frames en cours

    def is_speech(self, frame_bytes):
        """Retourne True si la frame contient de la parole, False sinon"""
        return self.vad.is_speech(frame_bytes, self.rate)

    def process(self, frames_iter):
        """
        Générateur qui yield des chunks audio (bytes PCM16) correspondant à des segments parlés.
        """
        for frame in frames_iter:
            speech = self.is_speech(frame)
            self.ring.append((frame, speech))

            if not self.triggered:
                # 🔹 Phase attente : déclencher si beaucoup de frames "voix"
                if len(self.ring) == self.ring.maxlen:
                    voiced_count = sum(1 for _, s in self.ring if s)
                    if voiced_count >= START_SPEECH_RATIO * len(self.ring):
                        self.triggered = True
                        # inclure pré-roll (début stocké dans ring)
                        self.voiced_buffer.extend(f for f, _ in self.ring)
                        self.ring.clear()
            else:
                # 🔹 Phase active : on accumule les frames
                self.voiced_buffer.append(frame)

                if len(self.ring) == self.ring.maxlen:
                    non_voiced = sum(1 for _, s in self.ring if not s)
                    if non_voiced >= STOP_SILENCE_RATIO * len(self.ring):
                        # fin du segment détectée
                        chunk = b"".join(self.voiced_buffer)

                        if len(chunk) >= MIN_SEGMENT_SIZE:
                            yield chunk
                        else:
                            print(f"[DEBUG] Segment ignoré, trop court ({len(chunk)} bytes)")

                        # reset état
                        self.voiced_buffer.clear()
                        self.ring.clear()
                        self.triggered = False

        # Flush si on termine pendant qu'on est déclenché
        if self.voiced_buffer:
            chunk = b"".join(self.voiced_buffer)
            if len(chunk) >= MIN_SEGMENT_SIZE:
                yield chunk
            else:
                print(f"[DEBUG] Segment ignoré, trop court ({len(chunk)} bytes)")
            self.voiced_buffer.clear()


# ==============================
# Utilitaires
# ==============================
def pcm16_bytes_to_float32(pcm_bytes):
    """
    Convertit bytes PCM16 mono en numpy float32 [-1, 1]
    → format attendu par faster-whisper
    """
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
