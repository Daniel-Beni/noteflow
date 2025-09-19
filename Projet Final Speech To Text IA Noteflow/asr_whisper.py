from faster_whisper import WhisperModel

MODEL_SIZE = "small.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

class WhisperASR:
    def __init__(self, model_size=MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_f32):
        segments, info = self.model.transcribe(
            audio_f32,
            language="en",
            vad_filter=False,
            beam_size=1,
            temperature=0
        )
        return " ".join(s.text.strip() for s in segments)
