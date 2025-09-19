import ollama
from fpdf import FPDF
import os

# ==============================
# Paramètres et configuration
# ==============================

MODEL = "gemma3:4b" 
OLLAMA_HOST = "127.0.0.1:11434"
PROMPT_PREFIX = """Internally:

Clean and correct the transcript (semantic corrections, coherence, fluidity).

Reformulate slightly if necessary to make it clearer and more readable.

Detect repetitions and parallels (when ideas are repeated or recalled later).

Do not output the corrected transcript or any explanations.

Final Output (the only thing you produce):
Provide a structured summary formatted exactly like this:

Main ideas
idea 1
idea 2
…
Secondary points
point 1
point 2
…
Recurring ideas and reminders
recurring idea 1
recurring idea 2
…
Keywords
keyword 1
keyword 2
…
Conclusions / Recommendations
conclusion 1
conclusion 2
…

Your answer must only contain this summary in the above format. No introduction, no commentary, no explanations."""


# ==============================
# Génération de la réponse via Ollama
# ==============================
def generer_reponse_ollama(text, model=MODEL, prompt_prefix=PROMPT_PREFIX):
    client = ollama.Client(host=OLLAMA_HOST)
    
    prompt_complet = f"{prompt_prefix}\n{text}"

    try:
        response = client.generate(model=model, keep_alive=0, prompt=prompt_complet)
        return response.get('response', 'Aucune réponse trouvée.')
    except Exception as e:
        print(f"Erreur lors de l'appel à Ollama : {e}")
        return f"Erreur de génération : {e}"

# ==============================
# Création du PDF
# ==============================
def creer_pdf_avec_reponse(prompt_content, ollama_output, output_filename):
    try:
        pdf = FPDF()
        
        pdf.add_font("dejavu-sans", "", "DejaVuSansCondensed.ttf", uni=True)
        pdf.add_font("dejavu-sans-bold", "B", "DejaVuSansCondensed-Bold.ttf", uni=True)
        
        pdf.add_page()
        pdf.set_font("dejavu-sans", size=12)

        pdf.multi_cell(0, 5, txt=f"Transcription:\n{prompt_content}")
        pdf.ln()

        pdf.set_font("dejavu-sans-bold", "B", 12)
        pdf.write(5, txt="Résumé Structuré:\n")
        pdf.set_font("dejavu-sans", size=12)
        
        lignes = ollama_output.strip().split('\n')
        for ligne in lignes:
            ligne = ligne.strip()
            if not ligne:
                continue
            
            est_titre = any(ligne.lower().startswith(h) for h in ["main ideas", "secondary points", "recurring ideas", "keywords", "conclusions"])
            
            if est_titre:
                pdf.set_font("dejavu-sans-bold", "B", 12)
                pdf.write(5, txt=ligne + '\n')
                pdf.set_font("dejavu-sans", size=12)
            else:
                pdf.write(5, txt=ligne + '\n')

        pdf.output(output_filename)
        print(f"Rapport exporté vers : {output_filename}")
    except Exception as e:
        print(f"Erreur lors de la création du PDF : {e}")