# -*- coding: utf-8 -*-
"""
CraftMyJob – Streamlit app for smart job suggestions
"""
import os
import io
import re
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from fpdf import FPDF
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ── 0) CLEAN ENVIRONMENT ─────────────────────────────────────────────────
for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(var, None)

# ── 1) STREAMLIT CONFIGURATION & STYLING ──────────────────────────────────
st.set_page_config(page_title="CraftMyJob – Job Seekers Hub France", layout="centered")
st.markdown("""
<style>
  .stButton>button { background-color:#2E86C1; color:white; border-radius:4px; }
  .section-header { font-size:1.2rem; font-weight:600; margin-top:1.5em; color:#2E86C1; }
  h1, h2, h3 { color:#2E86C1; }
</style>
""", unsafe_allow_html=True)

# Logo
try:
    logo = Image.open("logo_jobseekers.PNG")
    st.image(logo, width=120)
except FileNotFoundError:
    pass

st.title("✨ CraftMyJob – Votre assistant emploi intelligent")

# ── 2) DATA LOADING & MODEL PREPARATION ─────────────────────────────────
@st.cache_data
def load_referentiel(path: str = "referentiel_metiers_craftmyjob_final.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    return df

@st.cache_data
def build_tfidf_model(df: pd.DataFrame, max_features: int = 2000):
    corpus = df["Activites"] + " " + df["Competences"] + " " + df["Metier"]
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

referentiel = load_referentiel()
vecteur, tfidf_matrix = build_tfidf_model(referentiel)

# ── 3) HELPER FUNCTIONS ─────────────────────────────────────────────────

def build_keywords(texts: list[str], max_terms: int = 7) -> str:
    combined = " ".join(texts).lower()
    tokens = re.findall(r"\w{2,}", combined, flags=re.UNICODE)
    stop_words = {"et","ou","la","le","les","de","des","du","un","une",
                  "à","en","pour","par","avec","sans","sur","dans","au","aux"}
    seen, keywords = set(), []
    for tok in tokens:
        if tok in stop_words or tok in seen:
            continue
        seen.add(tok)
        keywords.append(tok)
        if len(keywords) >= max_terms:
            break
    return ",".join(keywords)


def get_gpt_response(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Tu es un expert en recrutement et personal branding."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    res = requests.post(url, json=payload, headers=headers, timeout=30)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


class PDFGen:
    @staticmethod
    def to_pdf(text: str) -> io.BytesIO:
        buffer = io.BytesIO()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 8, line)
        pdf.output(buffer)
        buffer.seek(0)
        return buffer


def fetch_ftoken(client_id: str, client_secret: str) -> str:
    auth_url = (
        "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    )
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "api_offresdemploiv2 o2dsoffre"
    }
    resp = requests.post(auth_url, data=data, timeout=10)
    resp.raise_for_status()
    return resp.json().get("access_token", "")


def search_offres(token: str, mots: str, localisation: str, limit: int = 5) -> list:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"motsCles": mots, "localisation": localisation, "range": f"0-{limit-1}"}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code == 204:
        return []
    if resp.status_code not in (200, 206):
        st.error(f"FT API {resp.status_code}: {resp.text}")
        return []
    return resp.json().get("resultats", [])


def scorer_metiers(profile: dict, df: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    document = " ".join([profile["missions"], profile["skills"], profile["job_title"], profile.get("desired_skills", "")])
    vec_user = vecteur.transform([document])
    cos_scores = cosine_similarity(vec_user, tfidf_matrix).flatten()
    df_scores = df.copy()
    df_scores["cosine"] = cos_scores
    df_scores["fz_title"] = df_scores["Metier"].apply(lambda m: fuzz.token_set_ratio(m, profile["job_title"]) / 100)
    df_scores["fz_missions"] = df_scores["Activites"].apply(lambda a: fuzz.token_set_ratio(a, profile["missions"]) / 100)
    df_scores["fz_comp"] = df_scores["Competences"].apply(lambda c: fuzz.token_set_ratio(c, profile["skills"]) / 100)
    df_scores["score"] = (0.5 * df_scores["cosine"] + 0.2 * df_scores["fz_title"] + 
                           0.15 * df_scores["fz_missions"] + 0.15 * df_scores["fz_comp"]) * 100
    return df_scores.nlargest(top_k, "score").reset_index(drop=True)


def search_territoires(query: str, limit: int = 10) -> list[str]:
    results = []
    if re.fullmatch(r"\d{2}", query):
        resp = requests.get(
            f"https://geo.api.gouv.fr/departements/{query}/communes", 
            params={"fields": "nom,codesPostaux", "limit": limit}, timeout=5
        )
        resp.raise_for_status()
        for entry in resp.json():
            cp = entry["codesPostaux"][0] if entry["codesPostaux"] else "00000"
            results.append(f"{entry['nom']} ({cp})")
        results.append(f"Departement {query}")
    else:
        resp = requests.get(
            "https://geo.api.gouv.fr/communes", 
            params={"nom": query, "fields": "nom,codesPostaux", "limit": limit}, timeout=5
        )
        if resp.status_code == 200:
            for entry in resp.json():
                cp = entry["codesPostaux"][0] if entry["codesPostaux"] else "00000"
                results.append(f"{entry['nom']} ({cp})")
        resp2 = requests.get(
            "https://geo.api.gouv.fr/regions",
            params={"nom": query, "fields": "nom,code"}, timeout=5
        )
        if resp2.status_code == 200:
            for rg in resp2.json():
                results.append(f"{rg['nom']} (region:{rg['code']})")
    return list(dict.fromkeys(results))

# ── 4) USER PROFILE FORM ─────────────────────────────────────────────────
st.header("1️⃣ Votre profil & vos préférences")
uploaded_cv = st.file_uploader("📂 Optionnel : Téléchargez votre CV", type=["pdf", "docx", "txt"])
cv_text = ""
if uploaded_cv:
    ext = uploaded_cv.name.split('.')[-1].lower()
    if ext == 'pdf':
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(uploaded_cv).pages)
    elif ext == 'docx':
        cv_text = " ".join(p.text for p in Document(uploaded_cv).paragraphs)
    else:
        cv_text = uploaded_cv.read().decode(errors='ignore')
job_title = st.text_input("🔤 Intitulé du poste souhaité")
missions = st.text_area("📋 Missions principales (ex : gestion de projet, recrutement, etc.")
values = st.text_area("🏢 Valeurs (facultatif)")
skills = st.text_area("🧠 Vos compétences clés")
desired_skills = st.text_area("✨ Compétences métier ciblées (facultatif)")

# Territoires
st.markdown("""<div class='section-header'>🌍 Où voulez-vous travailler ?</div>""", unsafe_allow_html=True)
typed = st.text_input("Commencez à taper une commune, département ou région…")
options = search_territoires(typed) if typed else []
default = st.session_state.get('locations', [])
selections = st.multiselect("Sélectionnez vos territoires", options=(default + options), default=default)
st.session_state.locations = selections

# Experience & contracts
exp_level = st.radio("🎯 Niveau d'expérience", ["Débutant (0-2 ans)", "Expérimenté (2-5 ans)", "Senior (5+ ans)"])
contract = st.selectbox("📄 Type de contrat", ["CDI","CDD","Freelance","Stage"])
remote = st.checkbox("🏠 Full remote available")

# ── 5) API KEYS & IA TEMPLATES ────────────────────────────────────────────
st.header("2️⃣ Clés API & génération IA")
openai_key   = st.text_input("🔑 Clé OpenAI API", type="password")
pe_client_id = st.text_input("🔑 Pôle-Emploi Client ID", type="password")
pe_secret    = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

templates = {
    "Bio LinkedIn": "Rédige une bio LinkedIn engageante et professionnelle.",
    "Mail de candidature": "Écris un mail de candidature spontanée clair et convaincant.",
    "Mini CV": "Générez un mini-CV de 5-7 lignes avec deux mots-clés.",
    "CV optimisé IA": "Rédigez un CV optimisé avec deux mots-clés.",
}
choices = st.multiselect("Que voulez-vous générer ?", list(templates.keys()), default=list(templates.keys())[:2])

# ── 6) ACTION BUTTON ─────────────────────────────────────────────────────
if st.button("🚀 Lancer l'analyse"):
    if not openai_key:
        st.error("🔑 Merci de renseigner votre clé OpenAI.")
        st.stop()
    if not (pe_client_id and pe_secret and selections):
        st.error("🔑 Merci de saisir vos identifiants Pôle-Emploi et au moins un territoire.")
        st.stop()

    profile = {
        "job_title": job_title,
        "missions": missions,
        "values": values,
        "skills": skills,
        "desired_skills": desired_skills,
        "territories": selections,
        "exp_level": exp_level,
        "contract": contract,
        "remote": remote
    }

    # Générations IA
    st.header("🧠 Résultats Génération IA")
    for name in choices:
        prompt = (
            f"Poste: {profile['job_title']}\n"
            f"Missions: {profile['missions']}\n"
            f"Compétences: {profile['skills']}\n"
            f"Compétences ciblées: {profile['desired_skills']}\n"
            f"Valeurs: {profile['values']}\n"
            f"Localités: {', '.join(selections)}\n"
            f"Expérience: {profile['exp_level']}\n"
            f"Contrat: {profile['contract']}\n"
            f"Télétravail: {'Oui' if profile['remote'] else 'Non'}\n\n"
            f"{templates[name]}"
        )
        try:
            result = get_gpt_response(prompt, openai_key)
            st.subheader(name)
            st.markdown(result)
            if name == "CV optimisé IA":
                pdf_buf = PDFGen.to_pdf(result)
