import os

def sauvegarder_transcription(transcription_parts, nom_fichier="transcription.txt"):
    if not transcription_parts:
        print("\nAucun texte n'a été transcrit, aucun fichier généré.")
        return

    print(f"\nSauvegarde de la transcription en cours...")
    
    final_text = " ".join(transcription_parts)
    
    try:
        with open(nom_fichier, "w", encoding="utf-8") as f:
            f.write(final_text)
        
        print(f"Transcription sauvegardée dans le fichier : {os.path.abspath(nom_fichier)}")

    except IOError as e:
        print(f" Erreur lors de la sauvegarde du fichier : {e}")