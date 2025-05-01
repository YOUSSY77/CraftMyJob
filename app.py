# -*- coding: utf-8 -*-
# 1️⃣ On vide d’emblée tout proxy pour ne PAS les transmettre à OpenAI.Client
import os
for v in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(v, None)

import streamlit as st
from openai import OpenAI
import requests
from PyPDF2 import PdfReader
from docx import Document
import io
from fpdf import FPDF
import pandas as pd
from rapidfuzz import fuzz

# --- Config de la page
st.set_page_config(page_title="CraftMyJob – by Job Seekers Hub France", layout="centered")
st.title("✨ CraftMyJob")
st.caption("by Job Seekers Hub France 🇫🇷")

# --- 1) Formulaire de saisie
st.subheader("🛠️ Décris ton projet professionnel")
uploaded_cv = st.file_uploader("📂 Optionnel : ton CV", type=["pdf", "docx", "txt"])
cv_text = ""
if uploaded_cv:
    ext = uploaded_cv.name.rsplit(".", 1)[-1].lower()
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
experience_level = st.radio("🎯 Expérience", ["Débutant(e)", "Expérimenté(e)", "Senior"])
contract_type    = st.selectbox("📄 Type de contrat", ["CDI", "Freelance", "CDD", "Stage"])
remote           = st.checkbox("🏠 Full remote")

# --- 2) Tes clés API
st.subheader("🔑 Clés API")
openai_key   = st.text_input("🔑 OpenAI API Key", type="password")
ft_client_id = st.text_input("🔑 Pôle-Emploi Client ID", type="password")
ft_secret    = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

# --- 3) Choix des contenus à générer
templates = {
    "📄 Bio LinkedIn":                  "Rédige une bio LinkedIn engageante et professionnelle.",
    "✉️ Mail de candidature":           "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV":                       "Génère un mini-CV (5-7 lignes), souligne deux mots-clés.",
    "🧩 CV optimisé IA":                "Rédige un CV optimisé, souligne deux mots-clés.",
}
choices = st.multiselect("🛠️ Génération IA", list(templates.keys()), default=list(templates.keys())[:2])

# --- Helper OpenAI (v1 client, sans proxy)
def get_gpt_response(prompt: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Tu es un expert en recrutement et en personal branding."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return resp.choices[0].message.content

def generate_prompt(label: str, inputs: dict, cv: str) -> str:
    prof = (
        f"Poste: {inputs['job_title']}\n"
        f"Missions: {inputs['missions']}\n"
        f"Compétences: {inputs['skills']}\n"
        f"Valeurs: {inputs['values']}\n"
        f"Localisation: {', '.join(inputs['locations'])}\n"
        f"Expérience: {inputs['experience_level']}\n"
        f"Contrat: {inputs['contract_type']}\n"
        f"Télétravail: {'Oui' if inputs['remote'] else 'Non'}\n"
    )
    if cv:
        prof += f"CV excerpt: {cv[:300]}...\n"
    return prof + "\n" + templates[label]

# --- Générateur PDF (fpdf2)
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

# --- Pôle-Emploi OAuth2 & recherche
def fetch_ft_token(cid: str, secret: str) -> str:
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "grant_type":    "client_credentials",
        "client_id":     cid,
        "client_secret": secret,
        "scope":         "api_offresdemploiv2 o2dsoffre"
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

def search_offres(token: str, mots: str, loc: str, limit: int = 7) -> list:
    url = "https://api.pole-emploi.io/partenaire/offresdemploi/v2/offres/search"
    headers = {"Authorization": f"Bearer {token}"}
    params  = {"motsCles": mots, "localisation": loc, "range": f"0-{limit-1}"}
    r = requests.get(url, headers=headers, params=params)
    return r.json().get("resultats", []) if r.status_code == 200 else []

# --- Chargement référentiel métiers & scoring 30/20/50
@st.cache_data
def load_ref():
    return pd.read_csv("referentiel_metiers_craftmyjob_final.csv")

df_metiers = load_ref()

def scorer_metier(inputs: dict, df: pd.DataFrame) -> pd.DataFrame:
    def sc(r):
        t = fuzz.token_set_ratio(r["Metier"],      inputs["job_title"])
        a = fuzz.token_set_ratio(r["Activites"],   inputs["missions"])
        c = fuzz.token_set_ratio(r["Competences"], inputs["skills"])
        return 0.3*t + 0.2*a + 0.5*c
    df["score"] = df.apply(sc, axis=1)
    return df.nlargest(6, "score")

# --- Bouton principal
if st.button("🚀 Générer & Chercher"):
    if not openai_key:
        st.error("👉 Ta clé OpenAI est requise")
        st.stop()

    inputs = {
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
    for label in choices:
        try:
            prompt = generate_prompt(label, inputs, cv_text)
            out    = get_gpt_response(prompt, openai_key)
            st.subheader(label)
            st.markdown(out)
            if label == "🧩 CV optimisé IA":
                pdf = PDFGen.to_pdf(out)
                st.download_button(
                    "📥 Télécharger CV (PDF)",
                    data=pdf,
                    file_name="CV_optimise.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"Erreur IA ({label}) : {e}")

    # — Recherche d’offres Pôle-Emploi
    if ft_client_id and ft_secret and locations:
        try:
            token = fetch_ft_token(ft_client_id, ft_secret)
            st.subheader("🔎 Offres Pôle-Emploi")
            for loc in locations:
                offres = search_offres(token, f"{job_title} {skills}", loc)
                for o in offres:
                    st.markdown(
                        f"**{o['intitule']}** – {o['entreprise']['nomEntreprise']} – "
                        f"{o['lieuTravail']['libelle']}  \n"
                        f"[Voir l'offre]({o['contact']['urlOrigine']})\n---"
                    )
        except Exception as e:
            st.error(f"Erreur FT API : {e}")

    # — Matching métiers
    st.subheader("🧠 SIS – Matching métiers ROME/ESCO")
    top6 = scorer_metier(inputs, df_metiers.copy())
    for _, row in top6.iterrows():
        st.markdown(f"**{row['Metier']}** – {int(row['score'])}%")
