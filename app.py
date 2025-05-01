import streamlit as st
from openai import OpenAI
import pandas as pd
from rapidfuzz import fuzz
from PyPDF2 import PdfReader
from docx import Document
import requests
import os
import io
import re
from fpdf import FPDF

# --- Page configuration
st.set_page_config(
    page_title="CraftMyJob – by Job Seekers Hub France", layout="centered"
)
st.title("✨ CraftMyJob")
st.caption("by Job Seekers Hub France 🇫🇷")

# --- Load métier reference
@st.cache_data
def load_metiers():
    return pd.read_csv("referentiel_metiers_craftmyjob_final.csv")
df_metiers = load_metiers()

# --- Load communes for autocomplete
use_autocomplete = False
communes_df = None
if os.path.exists("datacommunes.csv"):
    @st.cache_data
    def load_communes():
        df = pd.read_csv("datacommunes.csv", dtype=str)
        df["ville_cp"] = df["nom_commune"].str.strip() + " (" + df["code_postal"].str.strip() + ")"
        return df
    communes_df = load_communes()
    use_autocomplete = True

# --- OpenAI helper (v1 interface)

def get_gpt_response(prompt, api_key):
    # On utilise l'interface v1 sans passer de proxies
    client = OpenAI(api_key=api_key)
    # Veille à ne PAS fournir de paramètre proxies ici
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Tu es un expert en recrutement et en personal branding."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1200,
    )
    return response.choices[0].message.content


# --- France Travail API helpers

def fetch_ft_token(client_id, client_secret):
    url = (
        "https://entreprise.francetravail.fr/connexion/oauth2/access_token"
        "?realm=/partenaire"
    )
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "api_offresdemploiv2 o2dsoffre",
    }
    res = requests.post(url, data=data)
    res.raise_for_status()
    return res.json()["access_token"]


def get_ft_offers(token, keywords, localisation, limit=7):
    api_url = "https://api.francetravail.io/api/offres/v2/offres/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"motsCles": keywords, "localisation": localisation, "range": f"0-{limit-1}"}
    res = requests.get(api_url, headers=headers, params=params)
    if res.status_code != 200:
        return []
    return res.json().get("resultats", [])

# --- ROME/ESCO matching (pondération améliorée)

def scorer_metier(inputs, df):
    def score_row(row):
        s_title = fuzz.token_set_ratio(row.get('Metier',''), inputs['job_title'])
        s_act   = fuzz.token_set_ratio(row.get('Activites',''), inputs['missions'])
        s_comp  = fuzz.token_set_ratio(row.get('Competences',''), inputs['skills'])
        # pondération : 30% titre, 50% activités, 20% compétences
        return 0.3 * s_title + 0.5 * s_act + 0.2 * s_comp
    df['score'] = df.apply(score_row, axis=1)
    return df.nlargest(6, 'score')

# --- Prompt templates using display labels
TEMPLATES = {
    "📄 Bio LinkedIn": "Rédige une bio LinkedIn engageante, professionnelle.",
    "✉️ Mail de candidature spontanée": "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV": "Génère un mini-CV 5-7 lignes, sans astérisques, souligne deux mots-clés avec underscores.",
    "🧩 CV optimisé IA": "Rédige un CV optimisé, sans astérisques, souligne deux mots-clés avec underscores.",
}

def generate_prompt(template_label, inputs, cv_text):
    profile = (
        f"Profil :\n"
        f"- Poste : {inputs['job_title']}\n"
        f"- Missions : {inputs['missions']}\n"
        f"- Compétences : {inputs['skills']}\n"
        f"- Valeurs : {inputs['values']}\n"
        f"- Localisation : {', '.join(inputs['location']) if isinstance(inputs['location'], list) else inputs['location']}\n"
        f"- Expérience : {inputs['experience_level']}\n"
        f"- Contrat : {inputs['contract_type']}\n"
        f"- Télétravail : {'Oui' if inputs['remote'] else 'Non'}\n"
    )
    if cv_text:
        profile += f"- Extrait CV (500c) : {cv_text[:500]}...\n"
    return profile + "\n" + TEMPLATES.get(template_label, '')

# --- PDF generation with fpdf2
class PDFGenerator:
    @staticmethod
    def make_pdf(text: str):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 8, line)
        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return buf

# --- UI: inputs
st.subheader("🛠️ Décris ton projet")
file_u = st.file_uploader("Télécharge ton CV (optionnel)", type=["pdf","docx","txt"])
cv_text = ""
if file_u:
    ext = file_u.name.split('.')[-1].lower()
    if ext == 'pdf':
        cv_text = ' '.join(p.extract_text() or '' for p in PdfReader(file_u).pages)
    elif ext == 'docx':
        cv_text = ' '.join(p.text for p in Document(file_u).paragraphs)
    else:
        cv_text = file_u.read().decode()

# Données saisies
inputs = {
    'job_title': st.text_input("🔤 Poste recherché"),
    'missions': st.text_area("📋 Missions souhaitées"),
    'values': st.text_area("🏢 Valeurs (opt)"),
    'skills': st.text_area("🧠 Compétences clés"),
}
# Localisation
loc_input = st.text_input("📍 Localisation (villes, séparées par ,)")
if use_autocomplete:
    opts = communes_df[communes_df['nom_commune'].str.contains(loc_input, case=False, na=False)]['ville_cp'][:10].tolist()
    st.write("Suggestions :", opts)
    inputs['location'] = st.multiselect("Choisis villes", opts)
else:
    inputs['location'] = [x.strip() for x in loc_input.split(',') if x.strip()]
# Expérience, contrat, remote
inputs['experience_level'] = st.radio("🎯 Expérience", ["Débutant(e)","Expérimenté(e)","Senior"])
inputs['contract_type']   = st.selectbox("📄 Contrat", ["CDI","Freelance","CDD","Stage"])
inputs['remote']          = st.checkbox("🏠 Remote")

# Clés API
st.subheader("🔑 Clés API")
key_o      = st.text_input("OpenAI Key",           type='password')
key_id     = st.text_input("FT Client ID",        type='password')
key_secret = st.text_input("FT Client Secret",    type='password')

# Choix de génération
st.subheader("⚙️ Génération")
choices = st.multiselect(
    "Que veux-tu générer ?",
    list(TEMPLATES.keys()),
    default=["📄 Bio LinkedIn", "✉️ Mail de candidature spontanée"]
)

# Bouton d’action
results = {}
off_main = []
if st.button("🚀 Générer & Chercher"):
    if not key_o:
        st.error("Clé OpenAI requise.")
        st.stop()
    # Génération IA avec gestion d’erreur
    for label in choices:
        try:
            prompt = generate_prompt(label, inputs, cv_text)
            results[label] = get_gpt_response(prompt, key_o)
        except Exception as e:
            st.error(f"Erreur IA ({label}) : {e}")
    # Recherche offres FT
    if key_id and key_secret and inputs['location']:
        token = fetch_ft_token(key_id, key_secret)
        postal_codes = [m.group(1) if (m := re.search(r"\((\d{5})\)", loc)) else loc for loc in inputs['location']]
        for pc in postal_codes:
            off_main += get_ft_offers(token, f"{inputs['job_title']} {inputs['skills']}", pc)
        # dédoublonnage
        seen = set(); unique = []
        for o in off_main:
            url = o.get('contact',{}).get('urlOrigine','')
            if url and url not in seen:
                seen.add(url); unique.append(o)
        off_main = unique

# Affichage des résultats IA
for label, txt in results.items():
    st.subheader(label)
    st.markdown(txt)
    if label == '🧩 CV optimisé IA':
        pdf_bytes = PDFGenerator.make_pdf(txt)
        st.download_button(
            '📥 Télécharger PDF', data=pdf_bytes,
            file_name='CV_optimise.pdf', mime='application/pdf'
        )

# Affichage offres
if off_main:
    st.header("🔎 Offres Pôle Emploi")
    for o in off_main:
        t = o.get('intitule','—')
        e = o.get('entreprise',{}).get('nomEntreprise','—')
        l = o.get('lieuTravail',{}).get('libelle','—')
        u = o.get('contact',{}).get('urlOrigine','#')
        st.markdown(f"**{t}**  \n{e} – {l}  \n[Voir]({u})\n---")
else:
    st.info("Aucune offre trouvée.")

# SIS – Matching métiers
st.subheader("🧠 SIS – Matching métiers")
if inputs['job_title'] and inputs['missions'] and inputs['skills']:
    top6df = scorer_metier(
        {'job_title': inputs['job_title'], 'missions': inputs['missions'], 'skills': inputs['skills']},
        df_metiers.copy()
    )
    st.success("Top 6 des métiers ROME/ESCO (pondération améliorée)")
    for _, r in top6df.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
    # Recherche offres pour le premier métier
    if key_id and key_secret:
        token = fetch_ft_token(key_id, key_secret)
        off2 = []
        for pc in postal_codes:
           off2 += get_ft_offers(token, top6df.iloc[0]['Metier'], pc)
        seen2 = set(); uniq2 = []
        for o in off2:
            url = o.get('contact',{}).get('urlOrigine','')
            if url and url not in seen2: seen2.add(url); uniq2.append(o)
        st.subheader(f"🔎 Offres pour {top6df.iloc[0]['Metet']}")
        if uniq2:
            for o in uniq2:
                t = o.get('intitule','—'); e = o.get('entreprise',{}).get('nomEntreprise','—'); l = o.get('lieuTravail',{}).get('libelle','—'); u = o.get('contact',{}).get('urlOrigine','#')
                st.markdown(f"**{t}** – {e} ({l})  \n[Voir]({u})")
        else:
            st.info("Aucune offre trouvée pour ce métier.")
else:
    st.info("Renseignez Intitulé, Missions et Compétences.")
