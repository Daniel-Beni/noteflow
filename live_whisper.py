import sounddevice as sd
from scipy.io.wavfile import write
from transformers import pipeline

# Paramètres audio
samplerate = 16000
duration = 10  # phrases un peu plus longues
filename = "live_audio.wav"

# Charger Whisper Small
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

print("🎤 Parlez dans le micro...")
audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()

write(filename, samplerate, audio)
print(f"💾 Audio enregistré sous {filename}")

# Transcrire avec Whisper
result = asr(filename)
print("📝 Transcription :", result["text"])

