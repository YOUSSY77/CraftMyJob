
import streamlit as st
import openai
import pandas as pd
from rapidfuzz import fuzz
from PyPDF2 import PdfReader
from docx import Document
import requests
import os
import io
import re

# Optional PDF generation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    canvas = None
    A4 = None

# --- Page configuration
st.set_page_config(page_title="CraftMyJob – by Job Seekers Hub France", layout="centered")
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

# --- OpenAI query helper

def get_gpt_response(prompt, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Tu es un expert en recrutement et en personal branding."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1200
    )
    return response.choices[0].message.content

# --- France Travail API helpers

def fetch_ft_token(client_id, client_secret):
    url = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "api_offresdemploiv2 o2dsoffre"
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

# --- ROME/ESCO matching

def scorer_metier(inputs, df):
    def score_row(row):
        s1 = fuzz.token_set_ratio(row.get('Metier',''), inputs['job_title'])
        s2 = fuzz.token_set_ratio(row.get('Activites',''), inputs['missions'])
        s3 = fuzz.token_set_ratio(row.get('Competences',''), inputs['skills'])
        return (s1 + s2 + s3) / 3
    df['score'] = df.apply(score_row, axis=1)
    return df.nlargest(6, 'score')

# --- Prompt templates

def generate_prompt(template, inputs, cv_text):
    profile = (
        f"Profil :
"
        f"- Poste : {inputs['job_title']}
"
        f"- Missions : {inputs['missions']}
"
        f"- Compétences : {inputs['skills']}
"
        f"- Valeurs : {inputs['values']}
"
        f"- Localisation : {', '.join(inputs['location']) if isinstance(inputs['location'], list) else inputs['location']}
"
        f"- Expérience : {inputs['experience_level']}
"
        f"- Contrat : {inputs['contract_type']}
"
        f"- Télétravail : {'Oui' if inputs['remote'] else 'Non'}
"
    )
    if cv_text:
        profile += f"- Extrait CV (500c) : {cv_text[:500]}...
"
    texts = {
        "Bio": "Rédige une bio LinkedIn engageante, professionnelle.",
        "Mail": "Écris un mail de candidature spontanée clair et convaincant.",
        "MiniCV": "Génère un mini-CV 5-7 lignes, souligne deux mots-clés avec underscores.",
        "CVopt": "Rédige un CV optimisé, souligne deux mots-clés avec underscores."
    }
    return profile + "
" + texts.get(template, '')

# --- PDF generation

def make_pdf(text):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4; y = h - 40
    for line in text.split('
'):
        c.drawString(40, y, line); y -= 14
        if y < 40:
            c.showPage(); y = h - 40
    c.save(); buf.seek(0)
    return buf

# --- UI: inputs
st.subheader("🛠️ Décris ton projet")
file_u = st.file_uploader("Télécharge ton CV (optionnel)", type=["pdf","docx","txt"])
cv_text = ""
if file_u:
    ext = file_u.name.split('.')[-1].lower()
    if ext == 'pdf': cv_text = ' '.join(p.extract_text() or '' for p in PdfReader(file_u).pages)
    elif ext == 'docx': cv_text = ' '.join(p.text for p in Document(file_u).paragraphs)
    else: cv_text = file_u.read().decode()

job = st.text_input("🔤 Poste recherché")
missions = st.text_area("📋 Missions souhaitées")
values = st.text_area("🏢 Valeurs (opt)")
skills = st.text_area("🧠 Compétences clés")

# localisation multi
loc_input = st.text_input("📍 Localisation (villes, séparées par ,)")
if use_autocomplete:
    opts = communes_df[communes_df['nom_commune'].str.contains(loc_input, case=False, na=False)]['ville_cp'][:10].tolist()
    st.write("Suggestions:", opts)
    locations = st.multiselect("Choisis villes", opts)
else:
    locations = [x.strip() for x in loc_input.split(',') if x.strip()]
postal_codes = [re.search(r"\((\d{5})\)", loc).group(1) if re.search(r"\((\d{5})\)", loc) else loc for loc in locations]

exp = st.radio("🎯 Expérience", ["Débutant(e)","Expérimenté(e)","Senior"])
ctr = st.selectbox("📄 Contrat", ["CDI","Freelance","CDD","Stage"])
remote = st.checkbox("🏠 Remote")

# API keys
st.subheader("🔑 Clés API")
key_o = st.text_input("OpenAI Key", type='password')
key_id = st.text_input("FT Client ID", type='password')
key_secret = st.text_input("FT Client Secret", type='password')

# choix de génération
st.subheader("⚙️ Génération")
choices = st.multiselect("", ["Bio","Mail","MiniCV","CVopt"], default=["Bio"])

# action
results = {}
off_main = []
if st.button("🚀 Générer & Chercher"):
    if not key_o: st.error("OpenAI Key requise"); st.stop()
    inp = {'job_title': job, 'missions': missions, 'values': values, 'skills': skills, 'location': locations, 'experience_level': exp, 'contract_type': ctr, 'remote': remote}
    for ch in choices:
        results[ch] = get_gpt_response(generate_prompt(ch, inp, cv_text), key_o)
    if key_id and key_secret and postal_codes:
        token = fetch_ft_token(key_id, key_secret)
        for pc in postal_codes:
            off_main = off_main + get_ft_offers(token, f"{job} {skills}", pc)
        # dédupe
        uniq = []
        seen = set()
        for o in off_main:
            url = o.get('contact',{}).get('urlOrigine','')
            if url and url not in seen: seen.add(url); uniq.append(o)
        off_main = uniq

# affichage IA
for ch, txt in results.items():
    st.subheader(ch)
    st.markdown(txt)
    if ch == 'CVopt':
        pdf = make_pdf(txt)
        if pdf: st.download_button('📥 Télécharger PDF', data=pdf, file_name='CV.pdf', mime='application/pdf')

# affichage offres
if 'off_main' in locals():
    st.header("🔎 Offres Pôle Emploi")
    if off_main:
        for o in off_main:
            t = o.get('intitule','—'); e = o.get('entreprise',{}).get('nomEntreprise','—'); l = o.get('lieuTravail',{}).get('libelle','—'); u = o.get('contact',{}).get('urlOrigine','#')
            st.markdown(f"**{t}**  
{e} – {l}  
[Voir]({u})
---")
    else:
        st.info("Aucune offre trouvée.")

# SIS
st.subheader("🧠 SIS – Matching métiers")
if job and missions and skills:
    top6 = scorer_metier({'job_title': job, 'missions': missions, 'skills': skills}, df_metiers.copy())
    st.success("Top 6 métiers ROME/ESCO")
    for _, r in top6.iterrows(): st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
    if key_id and key_secret and postal_codes:
        token = fetch_ft_token(key_id, key_secret)
        off2 = []
        for pc in postal_codes:
            off2 = off2 + get_ft_offers(token, top6.iloc[0]['Metier'], pc)
        seen2 = set(); un2 = []
        for o in off2:
            url = o.get('contact',{}).get('urlOrigine','')
            if url and url not in seen2: seen2.add(url); un2.append(o)
        st.subheader(f"🔎 Offres pour {top6.iloc[0]['Metier']}")
        if un2:
            for o in un2:
                t = o.get('intitule','—'); e = o.get('entreprise',{}).get('nomEntreprise','—'); l = o.get('lieuTravail',{}).get('libelle','—'); u = o.get('contact',{}).get('urlOrigine','#')
                st.markdown(f"**{t}** – {e} ({l})  
[Voir]({u})")
        else: st.info("Aucune offre trouvée.")
else:
    st.info("Renseignez Intitulé, Missions et Compétences.")
