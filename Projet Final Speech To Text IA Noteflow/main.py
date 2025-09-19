import numpy as np
from audio_capture import MicStream, VADSegmenter, pcm16_bytes_to_float32
from asr_whisper import WhisperASR
from sauvegarde import sauvegarder_transcription
from summarize_transcript import generer_reponse_ollama, creer_pdf_avec_reponse

def main():
    
    # 1. Initialiser micro, VAD et Whisper
    mic = MicStream().start()
    vad = VADSegmenter()
    asr = WhisperASR()
    full_transcript = []
    
    print("Parlez dans le micro (Ctrl+C pour quitter)\n")

    try:
        # 2. Boucle principale : du micro → VAD → Whisper
        for chunk in vad.process(mic.frames()):
            # chunk = un segment complet en PCM16 (bytes)

            # 3. Convertir bytes PCM16 → float32 numpy
            audio_f32 = pcm16_bytes_to_float32(chunk)

            # 4. Transcrire avec Whisper
            text = asr.transcribe(audio_f32)

            # 5. Afficher le texte reconnu
            print(f"Transcript : {text}")
            full_transcript.append(text)

    except KeyboardInterrupt:
        print(" Arrêt demandé par l’utilisateur")
    finally:
        mic.stop()
        full_transcript_text = " ".join(full_transcript)
        if full_transcript_text:
            print("\nTranscription complète enregistrée.")
        
            # Appel de la fonction Ollama
            reponse_ollama = generer_reponse_ollama(full_transcript_text)
        
            # Définir le nom du fichier PDF
            pdf_filename = "resume_transcription.pdf"
        
            # Créer le rapport PDF|||
            creer_pdf_avec_reponse(full_transcript_text, reponse_ollama, pdf_filename)
        
        else:
            print("Aucune transcription à traiter.")

if __name__ == "__main__":
    main()
