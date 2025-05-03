# -*- coding: utf-8 -*-
import os
# ── 1) On nettoie les vars de proxy du système
for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(v, None)

import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document
import io
from fpdf import FPDF
import pandas as pd
from rapidfuzz import fuzz

# ── Config page
st.set_page_config(page_title="CraftMyJob – by Job Seekers Hub France", layout="centered")
st.title("✨ CraftMyJob")
st.caption("by Job Seekers Hub France 🇫🇷")

# ── 1) Formulaire
st.subheader("🛠️ Décris ton projet professionnel")
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
locations_input  = st.text_input("📍 Ville(s) (séparées par ,)")
locations        = [v.strip() for v in locations_input.split(",") if v.strip()]
experience_level = st.radio("🎯 Expérience", ["Débutant(e)","Expérimenté(e)","Senior"])
contract_type    = st.selectbox("📄 Type de contrat", ["CDI","Freelance","CDD","Stage"])
remote           = st.checkbox("🏠 Full remote")

# ── 2) Tes clés API
st.subheader("🔑 Clés API")
openai_key   = st.text_input("OpenAI API Key", type="password")
ft_client_id = st.text_input("Pôle-Emploi Client ID", type="password")
ft_secret    = st.text_input("Pôle-Emploi Client Secret", type="password")

# ── 3) Templates IA
templates = {
    "📄 Bio LinkedIn":    "Rédige une bio LinkedIn engageante et professionnelle.",
    "✉️ Mail de candidature": "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV":         "Génère un mini-CV (5-7 lignes), souligne deux mots-clés.",
    "🧩 CV optimisé IA":  "Rédige un CV optimisé, souligne deux mots-clés."
}
choices = st.multiselect("🛠️ Génération IA", list(templates.keys()), default=list(templates.keys())[:2])

# ── Helper : appel direct à l'API ChatCompletion via HTTP
def get_gpt_response(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Tu es un expert en recrutement et en personal branding."},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
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

# ── PDF via fpdf2
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

# ── Pôle-Emploi OAuth2 & recherche
def fetch_ft_token(cid: str, secret: str) -> str:
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "grant_type":    "client_credentials",
        "client_id":     cid,
        "client_secret": secret,
        "scope":         "api_offresdemploiv2 o2dsoffre"
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(url, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

def search_offres(token: str, mots: str, loc: str, limit: int = 7) -> list:
    url     = "https://api.pole-emploi.io/partenaire/offresdemploi/v2/offres/search"
    headers = {"Authorization": f"Bearer {token}"}
    params  = {"motsCles": mots, "localisation": loc, "range": f"0-{limit-1}"}

    r = requests.get(url, headers=headers, params=params, timeout=30)
    st.write(f"Recherche FT pour « {loc} » → HTTP {r.status_code}")

    # On accepte 200, 206 (partial) et 204 (no content) comme succès
    if r.status_code == 204:
        st.info(f"Aucune offre trouvée pour {loc} (204 No Content)")
        return []

    if r.status_code not in (200, 206):
        st.error(f"❌ Erreur FT API {r.status_code} : {r.text}")
        return []

    data = r.json().get("resultats", [])
    st.write(f"  • Nombre d’offres reçues pour {loc} : {len(data)}")
    return data


# ── Chargement référentiel métiers + scoring
@st.cache_data
def load_ref() -> pd.DataFrame:
    return pd.read_csv("referentiel_metiers_craftmyjob_final.csv")

df_metiers = load_ref()

def scorer_metier(inp: dict, df: pd.DataFrame) -> pd.DataFrame:
    def score(r):
        s1 = fuzz.token_set_ratio(r["Metier"],      inp["job_title"])
        s2 = fuzz.token_set_ratio(r["Activites"],   inp["missions"])
        s3 = fuzz.token_set_ratio(r["Competences"], inp["skills"])
        return 0.3*s1 + 0.2*s2 + 0.5*s3
    df["score"] = df.apply(score, axis=1)
    return df.nlargest(6, "score")

# ── Bouton principal
if st.button("🚀 Générer & Chercher"):
    if not openai_key:
        st.error("❌ Clé OpenAI requise")
        st.stop()

    inp = {
        "job_title":        job_title,
        "missions":         missions,
        "values":           values,
        "skills":           skills,
        "locations":        locations,
        "experience_level": experience_level,
        "contract_type":    contract_type,
        "remote":           remote
    }

    # — Générations IA
    for lbl in choices:
        try:
            pr  = generate_prompt(lbl, inp, cv_text)
            out = get_gpt_response(pr, openai_key)
            st.subheader(lbl)
            st.markdown(out)
            if lbl == "🧩 CV optimisé IA":
                pdf = PDFGen.to_pdf(out)
                st.download_button("📥 Télécharger CV (PDF)", data=pdf,
                                   file_name="CV_optimise.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"❌ Erreur IA ({lbl}) : {e}")

    # — Offres Pôle-Emploi
    if ft_client_id and ft_secret and locations:
        try:
            token = fetch_ft_token(ft_client_id, ft_secret)
            st.subheader("🔎 Offres Pôle-Emploi")
            for loc in locations:
                offres = search_offres(token, f"{job_title} {skills}", loc)
                if offres:
                    for o in offres:
                        st.markdown(
                            f"**{o['intitule']}** – {o['entreprise']['nomEntreprise']} – "
                            f"{o['lieuTravail']['libelle']}  \n"
                            f"[Voir l'offre]({o['contact']['urlOrigine']})\n---"
                        )
                else:
                    st.info(f"Aucune offre pour {loc}")
        except Exception as e:
            st.error(f"❌ Erreur Pôle-Emploi : {e}")

    # — Matching métiers
    st.subheader("🧠 SIS – Matching métiers ROME/ESCO")
    top6 = scorer_metier(inp, df_metiers.copy())
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
