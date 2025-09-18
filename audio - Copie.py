import sounddevice as sd
import queue
import numpy as np
import torch
from pyannote.audio import Inference
from huggingface_hub import login
from sklearn.cluster import KMeans
from faster_whisper import WhisperModel

# ==========================
# 1. Authentification Hugging Face
# ==========================
login("je met pas le key sur git les gars")  
# Charger modèle d'embedding speaker
spk_embedder = Inference("pyannote/embedding", window="whole", device=torch.device("cpu"))

# ==========================
# 2. Config
# ==========================
sr = 16000                 # Whisper fonctionne en 16kHz
chunk_duration = 5         # secondes, plus long pour plus de contexte
model = WhisperModel("small", device="cpu", compute_type="int8")  # rapide et léger

audio_q = queue.Queue()

# ==========================
# 3. Capture micro
# ==========================
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

# Choisir le bon périphérique micro
print(sd.query_devices())  # repère ton micro
stream = sd.InputStream(samplerate=sr, channels=1, callback=callback)
stream.start()

print("🎤 Enregistrement en cours... (CTRL+C pour quitter)")

# ==========================
# 4. Boucle temps réel
# ==========================
buffer = np.zeros(int(sr * chunk_duration))
embeddings = []

try:
    while True:
        data = audio_q.get()
        data = data[:, 0]  # mono
        buffer = np.roll(buffer, -len(data))
        buffer[-len(data):] = data

        # Normalisation
        audio_float32 = buffer.astype(np.float32)
        max_val = np.max(np.abs(audio_float32)) + 1e-9
        audio_float32 /= max_val

        # Filtrer le silence strictement
        rms = np.sqrt(np.mean(audio_float32**2))
        if rms < 0.01:  # seuil à ajuster selon ton micro
            continue  # ignorer le chunk trop silencieux

        # ========== Transcription ==========
        segments, _ = model.transcribe(audio_float32, beam_size=5, language="fr")  # "en" pour anglais
        text = " ".join([seg.text for seg in segments]).strip()

        if not text:
            continue

        # ========== Embedding speaker ==========
        waveform = torch.tensor(audio_float32).unsqueeze(0)  # [1, time]
        emb = spk_embedder({"waveform": waveform, "sample_rate": sr})

        # Conversion en numpy
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy().squeeze()
        else:
            emb_np = np.array(emb).squeeze()

        embeddings.append(emb_np)
        if len(embeddings) > 30:
            embeddings = embeddings[-30:]

        # ========== Clustering ==========
        speaker = 0
        if len(embeddings) > 5:
            km = KMeans(n_clusters=2, n_init=10).fit(embeddings)
            speaker = km.labels_[-1]

        # ========== Affichage + log ==========
        print(f"SPEAKER_{speaker}: {text}")
        with open("conversation_log.txt", "a", encoding="utf-8") as f:
            f.write(f"SPEAKER_{speaker}: {text}\n")

except KeyboardInterrupt:
    print("\n🛑 Arrêt du programme demandé par l’utilisateur.")
    stream.stop()
    stream.close()
