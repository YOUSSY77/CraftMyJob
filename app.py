# -*- coding: utf-8 -*-
import os
# ── 0) Supprimer les vars de proxy hérités
for v in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(v, None)

import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document
import io
from fpdf import FPDF
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Initialisation session_state pour la sélection des villes
if "locations" not in st.session_state:
    st.session_state.locations = []

# ── Config page
st.set_page_config(
    page_title="CraftMyJob – by Job Seekers Hub France",
    layout="centered"
)
st.title("✨ CraftMyJob")
st.caption("by Job Seekers Hub France 🇫🇷")

# ── Chargement référentiel métiers
@st.cache_data
def load_metiers() -> pd.DataFrame:
    return pd.read_csv("referentiel_metiers_craftmyjob_final.csv", dtype=str)

df_metiers = load_metiers()

# ── Construction TF-IDF pour SIS
@st.cache_data
def build_tfidf(df: pd.DataFrame):
    corpus = (
        df["Activites"].fillna("") + " " +
        df["Competences"].fillna("") + " " +
        df["Metier"].fillna("")
    ).tolist()
    vect = TfidfVectorizer(max_features=2000)
    X_ref = vect.fit_transform(corpus)
    return vect, X_ref

vect, X_ref = build_tfidf(df_metiers)

# ── Helpers

def get_gpt_response(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Tu es un expert en recrutement et en personal branding."},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

class PDFGen:
    @staticmethod
    def to_pdf(text: str) -> io.BytesIO:
        buf = io.BytesIO()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 8, line)
        pdf.output(buf)
        buf.seek(0)
        return buf

# ── Autocomplétion multi-villes via geo.api.gouv.fr

def search_communes(query: str, limit: int = 10) -> list[str]:
    url = "https://geo.api.gouv.fr/communes"
    params = {"nom": query, "fields": "nom,codesPostaux", "boost": "population", "limit": limit}
    r = requests.get(url, params=params, timeout=5)
    r.raise_for_status()
    résultats = []
    for c in r.json():
        cp = c.get("codesPostaux", ["00000"])[0]
        résultats.append(f"{c['nom']} ({cp})")
    return résultats

# ── 1️⃣ Que souhaites-tu faire dans la vie ?
st.header("1️⃣ Que souhaites-tu faire dans la vie ?")
# CV uploader + texte
uploaded_cv = st.file_uploader("📂 Optionnel : ton CV", type=["pdf","docx","txt"])
cv_text = ""
if uploaded_cv:
    ext = uploaded_cv.name.rsplit(".",1)[-1].lower()
    if ext == "pdf":
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(uploaded_cv).pages)
    elif ext == "docx":
        cv_text = " ".join(p.text for p in Document(uploaded_cv).paragraphs)
    else:
        cv_text = uploaded_cv.read().decode()

job_title = st.text_input("🔤 Intitulé du poste souhaité")
missions  = st.text_area("📋 Missions principales")
values    = st.text_area("🏢 Valeurs (facultatif)")
skills    = st.text_area("🧠 Compétences clés")

# Autocomplete multi-villes
typed = st.text_input("📍 Commencez à taper une ville…")
raw_suggestions = search_communes(typed) if typed else []
options = list(dict.fromkeys(st.session_state.locations + raw_suggestions))
st.session_state.locations = st.multiselect(
    "Sélectionnez une ou plusieurs villes", options=options,
    default=st.session_state.locations, key="locations"
)
# Extraction CP
postal_codes = [m.group(1) for loc in st.session_state.locations
                for m in [re.search(r"\((\d{5})\)", loc)] if m]

# ── 2️⃣ Générations IA
st.header("2️⃣ Générations IA")
templates = {
    "📄 Bio LinkedIn":        "Rédige une bio LinkedIn engageante et professionnelle.",
    "✉️ Mail de candidature": "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV":             "Génère un mini-CV (5-7 lignes), souligne deux mots-clés.",
    "🧩 CV optimisé IA":      "Rédige un CV optimisé, souligne deux mots-clés."
}
choices = st.multiselect("Choisis ce que tu veux générer", list(templates), default=list(templates)[:2])

# ── 3️⃣ Tes clés API
st.header("3️⃣ Tes clés API")
openai_key   = st.text_input("🔑 OpenAI API Key", type="password")
ft_client_id = st.text_input("🔑 Pôle-Emploi Client ID", type="password")
ft_secret    = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

# ── 4️⃣ Paramètres et bouton
# Expérience en années plutôt que simple étiquette
experience_map = {
    "0-2 ans": "Débutant(e)",
    "3-5 ans": "Expérimenté(e)",
    ">=6 ans": "Senior"
}
st.radio("🎯 Ton expérience (en années)", list(experience_map.keys()), format_func=lambda x: f"{x} ({experience_map[x]})")
contract_type = st.selectbox("📄 Type de contrat", ["CDI","Freelance","CDD","Stage"])
remote        = st.checkbox("🏠 Full remote")

if st.button("🚀 Lancer tout"):
    # vérifications
    if not openai_key:
        st.error("🔑 Clé OpenAI requise"); st.stop()
    if not (ft_client_id and ft_secret and postal_codes):
        st.warning("🔑 Pôle-Emploi + au moins une ville requis"); st.stop()
    inp = dict(
        job_title=job_title, missions=missions, values=values,
        skills=skills, locations=st.session_state.locations,
        experience_level=st.session_state["selected_experience"], # récupère clé radio
        contract_type=contract_type, remote=remote
    )
    # IA
    for lbl in choices:
        out = get_gpt_response(
            generate_prompt(lbl, inp, cv_text), openai_key
        )
        st.subheader(lbl)
        st.markdown(out)
        if lbl == "🧩 CV optimisé IA":
            pdf = PDFGen.to_pdf(out)
            st.download_button("📥 Télécharger CV", data=pdf,
                               file_name="CV_optimise.pdf", mime="application/pdf")
    
    # Matching & Offres
    token = fetch_ft_token(ft_client_id, ft_secret)
    st.header("🔎 Top 5 offres pour ton profil")
    mots = ",".join(re.findall(r"\w{2,}", job_title + " " + skills)[:7])
    offres = []
    for cp in postal_codes:
        offres += search_offres(token, mots, cp, limit=5)
    seen, uniq = set(), []
    for o in offres:
        url = o.get('contact',{}).get('urlOrigine','')
        if url and url not in seen:
            seen.add(url); uniq.append(o)
    if uniq:
        for o in uniq[:5]:
            st.markdown(f"**{o['intitule']}** – {o['lieuTravail']['libelle']}  \n"  
                        f"[Voir]({o['contact']['urlOrigine']})")
    else:
        st.info("🔍 Aucune offre trouvée pour ce poste.")
    
    # SIS
    st.header("🧠 SIS – Métiers correspondants (matching métier)")
    st.caption("Le score est basé sur cosinus TF-IDF (activités+compétences+intitulé)")
    top6 = scorer_metier(inp, df_metiers, top_k=6)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        # offres par métier
        subs = []
        for cp in postal_codes:
            subs += search_offres(token, r['Metier'], cp, limit=3)
        seen2, uniq2 = set(), []
        for o in subs:
            u2 = o['contact'].get('urlPostulation') or o['contact'].get('urlOrigine','')
            if u2 and u2 not in seen2:
                seen2.add(u2); uniq2.append(o)
        for o in uniq2[:3]:
            date = o.get('dateCreation','')[:10]
            lien = o['contact'].get('urlPostulation') or o['contact'].get('urlOrigine','#')
            desc = (o.get('description','') or '').replace("\n"," ")[:100] + '…'
            st.markdown(
                f"• **{o['intitule']}** – {o['lieuTravail']['libelle']} ({date})  \n" 
                f"  {desc}  \n" 
                f"  [Voir/Poster]({lien})"
            )


