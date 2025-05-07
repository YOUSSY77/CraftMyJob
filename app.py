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

# ── Initialisation session_state
if "locations" not in st.session_state:
    st.session_state.locations = []
if "departements" not in st.session_state:
    st.session_state.departements = []

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

# ── Helpers IA / PDF
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

# ── Recherche géographique (communes + départements)
def search_communes(query: str, limit: int = 10) -> list[str]:
    url = "https://geo.api.gouv.fr/communes"
    params = {"nom": query, "fields": "nom,codesPostaux,codeDepartement", "boost": "population", "limit": limit}
    r = requests.get(url, params=params, timeout=5)
    r.raise_for_status()
    suggestions = []
    for c in r.json():
        cp = c.get("codesPostaux", ["00000"])[0]
        dep = c.get("codeDepartement")
        suggestions.append(f"{c['nom']} ({cp})")
        if dep and dep not in st.session_state.departements:
            st.session_state.departements.append(dep)
    return suggestions

# ── Recherche offres Pôle-Emploi
def fetch_ft_token(cid: str, sec: str) -> str:
    auth_url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {"grant_type":"client_credentials","client_id":cid,"client_secret":sec,"scope":"api_offresdemploiv2 o2dsoffre"}
    r = requests.post(auth_url, data=data, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]

def search_offres(token: str, mots: str, loc_param: str, limit: int = 5) -> list:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"motsCles": mots, "localisation": loc_param, "range": f"0-{limit-1}"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code == 204:
        return []
    if r.status_code not in (200,206):
        st.error(f"FT API {r.status_code} : {r.text}")
        return []
    return r.json().get("resultats", [])

# ── Scorer métier (Cosinus)
def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    user_doc = " ".join([inp['missions'], inp['skills'], inp['job_title']])
    v_user = vect.transform([user_doc])
    cosines = cosine_similarity(v_user, X_ref).flatten()
    df2 = df.copy()
    df2['score'] = (cosines * 100).round(1)
    return df2.nlargest(top_k, 'score')

# ── 1️⃣ Inputs projet pro
st.header("1️⃣ Ta recherche")
uploaded_cv = st.file_uploader("📂 Optionnel : ton CV", type=["pdf","docx","txt"])
cv_text = ""
if uploaded_cv:
    ext = uploaded_cv.name.rsplit(".",1)[-1].lower()
    if ext == "pdf":
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(uploaded_cv).pages)
    elif ext == "docx":
        cv_text = " ".join(p.text for p in Document(uploaded_cv).paragraphs)
    else:
        cv_text = uploaded_cv.read().decode()

job_title = st.text_input("🔤 Intitulé souhaité")
missions  = st.text_area("📋 Missions principales")
skills    = st.text_area("🧠 Compétences clés")

# Géolocalisation multi-sélection : communes
typed = st.text_input("🔍 Ville(s) ou commune(s)")
suggestions = search_communes(typed) if typed else []
options = list(dict.fromkeys(st.session_state.locations + suggestions))
st.session_state.locations = st.multiselect("Sélectionnez villes/communes", options=options, default=st.session_state.locations)
# codes postaux + départements
postal_codes = [re.search(r"\((\d{5})\)", loc).group(1) for loc in st.session_state.locations if re.search(r"\((\d{5})\)", loc)]
dep_codes = st.session_state.departements

# ── 2️⃣ Générations IA
templates = {
    "📄 Bio LinkedIn":        "Rédige une bio LinkedIn engageante et professionnelle.",
    "✉️ Mail de candidature": "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV":             "Génère un mini-CV (5-7 lignes), souligne deux mots-clés.",
    "🧩 CV optimisé IA":      "Rédige un CV optimisé, souligne deux mots-clés."
}
choices = st.multiselect("3️⃣ Générations IA – Choisis ce que tu veux générer", list(templates), default=list(templates)[:2])

# ── 3️⃣ Clés API openai + pole-emploi
openai_key   = st.text_input("🔑 OpenAI Key", type="password")
ft_id        = st.text_input("🔑 Pôle-Emploi ID", type="password")
ft_secret    = st.text_input("🔑 Pôle-Emploi Secret", type="password")

# ── 4️⃣ Lancer tout
if st.button("🚀 Lancer tout"):
    if not openai_key:
        st.error("Clé OpenAI requise"); st.stop()
    if not (ft_id and ft_secret and (postal_codes or dep_codes)):
        st.error("Indique Pôle-Emploi + zones (CP ou dép)"); st.stop()

    inp = dict(job_title=job_title, missions=missions, skills=skills)

    # — IA
    for lbl in choices:
        out = get_gpt_response(generate_prompt(lbl, inp, cv_text), openai_key)
        st.subheader(lbl); st.markdown(out)

    # — Offres top 5 filtrées géographiquement
    token = fetch_ft_token(ft_id, ft_secret)
    mots = ",".join(re.findall(r"\w{2,}", job_title + " " + skills)[:7])
    raw = []
    for cp in postal_codes + dep_codes:
        raw += search_offres(token, mots, cp, limit=5)
    filtres = []
    for o in raw:
        cpo = o['lieuTravail']['codePostal']
        if cpo in postal_codes or any(cpo.startswith(d) for d in dep_codes):
            filtres.append(o)
    seen, uniq = set(), []
    for o in filtres:
        url = o['contact'].get('urlOrigine','')
        if url and url not in seen:
            seen.add(url); uniq.append(o)
    st.subheader("🔎 Offres Geo-filtées")
    if uniq:
        for o in uniq[:5]:
            st.markdown(f"**{o['intitule']}** — {o['lieuTravail']['libelle']}  \n[Voir]({o['contact']['urlOrigine']})")
    else:
        st.info("Aucune offre dans tes zones sélectionnées.")

    # — SIS métiers
    st.subheader("🧠 SIS – Matching métier")
    top6 = scorer_metier(inp, df_metiers)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** — {int(r['score'])}%")


