# -*- coding: utf-8 -*-
import os
# ── 1) On nettoie les vars de proxy
for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(v, None)

import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document
import io
from fpdf import FPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ── Config page
st.set_page_config(
    page_title="CraftMyJob – by Job Seekers Hub France",
    layout="centered"
)
st.title("✨ CraftMyJob")
st.caption("by Job Seekers Hub France 🇫🇷")

# ── CONSTANTES
REGIONS_MAP = {
    "ile de france": ["75","77","78","91","92","93","94","95"]
}

# ── 1) Que souhaites-tu faire dans la vie ?  
st.header("1️⃣ Que souhaites-tu faire dans la vie ?")
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

job_title        = st.text_input("🔤 Intitulé du poste souhaité")
missions         = st.text_area("📋 Missions principales")
values           = st.text_area("🏢 Valeurs (facultatif)")
skills           = st.text_area("🧠 Compétences clés")
loc_input        = st.text_input("📍 Localisation (villes ou région)")
locations        = [v.strip() for v in re.split("[,;]", loc_input) if v.strip()]
experience_level = st.radio("🎯 Niveau d'expérience",
                            ["Débutant(e)","Expérimenté(e)","Senior"])
contract_type    = st.selectbox("📄 Type de contrat",
                                ["CDI","Freelance","CDD","Stage"])
remote           = st.checkbox("🏠 Full remote")

# ── Normalisation localisation → postal_codes
postal_codes = []
for loc in locations:
    key = loc.lower()
    # région
    if key in REGIONS_MAP:
        postal_codes += REGIONS_MAP[key]
        continue
    # CP entre parenthèses
    m = re.search(r"\((\d{5})\)", loc)
    if m:
        postal_codes.append(m.group(1))
        continue
    # chiffres seuls 2 ou 5 → cp ou département
    clean = re.sub(r"\D","", loc)
    if clean.isdigit() and len(clean) in (2,5):
        postal_codes.append(clean)
        continue
    # sinon on tente la commune brute
    postal_codes.append(loc)

# ── 2) Clés API
st.header("2️⃣ Tes clés API")
openai_key   = st.text_input("🔑 OpenAI API Key", type="password")
ft_client_id = st.text_input("🔑 Pôle-Emploi Client ID", type="password")
ft_secret    = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

# ── 3) Générations IA
st.header("3️⃣ Générations IA")
templates = {
    "📄 Bio LinkedIn":           "Rédige une bio LinkedIn engageante et professionnelle.",
    "✉️ Mail de candidature":    "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV":                "Génère un mini-CV (5-7 lignes), souligne deux mots-clés.",
    "🧩 CV optimisé IA":         "Rédige un CV optimisé, souligne deux mots-clés."
}
choices = st.multiselect(
    "Choisis les éléments à générer",
    list(templates.keys()),
    default=list(templates.keys())[:2]
)

def get_gpt_response(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": 
                "Tu es un expert en recrutement et en personal branding."},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def generate_prompt(label: str, inp: dict, cv: str) -> str:
    base = (
        f"Poste: {inp['job_title']}\n"
        f"Missions: {inp['missions']}\n"
        f"Compétences: {inp['skills']}\n"
        f"Valeurs: {inp['values']}\n"
        f"Localisation: {', '.join(inp['locations'])}\n"
        f"Expérience: {inp['experience_level']}\n"
        f"Contrat: {inp['contract_type']}\n"
        f"Télétravail: {'Oui' if inp['remote'] else 'Non'}\n"
    )
    if cv:
        base += f"CV extrait: {cv[:300]}...\n"
    return base + "\n" + templates[label]

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

# ── 4) Matching & Offres
st.header("4️⃣ Matching et Offres")

def fetch_ft_token(cid: str, sec: str) -> str:
    auth_url = (
        "https://entreprise.pole-emploi.fr"
        "/connexion/oauth2/access_token?realm=/partenaire"
    )
    data = {
        "grant_type":    "client_credentials",
        "client_id":     cid,
        "client_secret": sec,
        "scope":         "api_offresdemploiv2 o2dsoffre"
    }
    resp = requests.post(auth_url, data=data, timeout=10)
    resp.raise_for_status()
    return resp.json()["access_token"]

def search_offres(token: str, mots: str, loc: str, limit: int = 5) -> list:
    url = (
        "https://api.francetravail.io"
        "/partenaire/offresdemploi/v2/offres/search"
    )
    headers = {"Authorization": f"Bearer {token}"}
    params = {"motsCles": mots, "localisation": loc, "range": f"0-{limit-1}"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code == 204:
        return []
    if r.status_code not in (200,206):
        st.error(f"FT API {r.status_code} : {r.text}")
        return []
    return r.json().get("resultats", [])

# ── 5) SIS – Dis-moi ce que tu sais faire
@st.cache_data
def load_ref() -> pd.DataFrame:
    return pd.read_csv("referentiel_metiers_craftmyjob_final.csv")

df_metiers = load_ref()

@st.cache_data(show_spinner=False)
def build_tfidf(df):
    corpus = (
        df["Activites"].fillna("") + " " +
        df["Competences"].fillna("") + " " +
        df["Metier"].fillna("")
    )
    vect = TfidfVectorizer(stop_words="french", max_features=2000)
    X = vect.fit_transform(corpus)
    return vect, X

vect, X_ref = build_tfidf(df_metiers)

def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    user_doc = " ".join([inp["missions"], inp["skills"], inp["job_title"]])
    v_user = vect.transform([user_doc])
    cosines = cosine_similarity(v_user, X_ref).flatten()
    df2 = df.copy()
    df2["score"] = (cosines * 100).round(1)
    return df2.nlargest(top_k, "score")

# ── ACTION lorsqu'on clique
if st.button("🚀 Lancer tout"):
    if not openai_key:
        st.error("🔑 Ta clé OpenAI est requise"); st.stop()
    if not (ft_client_id and ft_secret):
        st.error("🔑 Tes identifiants Pole-Emploi sont requis"); st.stop()

    inp = {
        "job_title": job_title, "missions": missions,
        "values": values,     "skills": skills,
        "locations": locations, "experience_level": experience_level,
        "contract_type": contract_type, "remote": remote
    }

    # — IA
    for lbl in choices:
        try:
            out = get_gpt_response(generate_prompt(lbl, inp, cv_text), openai_key)
            st.subheader(lbl)
            st.markdown(out)
            if lbl == "🧩 CV optimisé IA":
                pdf = PDFGen.to_pdf(out)
                st.download_button("📥 Télécharger PDF", data=pdf,
                                   file_name="CV_optimise.pdf",
                                   mime="application/pdf")
        except Exception as e:
            st.error(f"Erreur IA ({lbl}) : {e}")

    # — Offres France Travail
    token = fetch_ft_token(ft_client_id, ft_secret)
    st.subheader("🔎 Top 5 offres correspondant à ton profil")
    mots = f"{job_title} {skills}"
    offres_all = []
    for pc in postal_codes:
        offres_all += search_offres(token, mots, pc, limit=5)
    # dédupe
    seen = set(); uniq = []
    for o in offres_all:
        url = o.get("contact",{}).get("urlOrigine","")
        if url and url not in seen:
            seen.add(url); uniq.append(o)
    for o in uniq[:5]:
        st.markdown(f"**{o['intitule']}** – {o['lieuTravail']['libelle']}  \n"
                    f"[Voir]({o['contact']['urlOrigine']})\n---")

    # — SIS
    st.subheader("🧠 SIS – Les métiers qui te correspondent")
    top6 = scorer_metier(inp, df_metiers.copy(), top_k=6)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        # offres par métier
        offres2 = search_offres(token, r["Metet"] if "Metet" in r else r["Metier"], postal_codes[0], limit=3)
        for o in offres2:
            st.write(f"  • {o['intitule']} ({o['lieuTravail']['libelle']})")

