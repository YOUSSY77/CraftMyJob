import os
# On vide les proxies pour que la v1 de OpenAI.Client n’en hérite pas
for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
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
import os
# Supprime les proxies hérités de l'environnement pour ne pas polluer OpenAI.Client
for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(v, None)


# --- Page config
st.set_page_config(page_title="CraftMyJob – by Job Seekers Hub France", layout="centered")
st.title("✨ CraftMyJob")
st.caption("by Job Seekers Hub France 🇫🇷")

# --- 1) Formulaire
st.subheader("🛠️ Décris ton projet professionnel")
uploaded_cv = st.file_uploader("📂 Optionnel : ton CV", type=["pdf","docx","txt"])
cv_text = ""
if uploaded_cv:
    ext = uploaded_cv.name.rsplit(".",1)[-1].lower()
    if ext=="pdf":
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(uploaded_cv).pages)
    elif ext=="docx":
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

# --- 2) Tes clés
st.subheader("🔑 Clés API")
openai_key   = st.text_input("OpenAI API Key", type="password")
ft_client_id = st.text_input("Pole-Emploi Client ID", type="password")
ft_secret    = st.text_input("Pole-Emploi Client Secret", type="password")

# --- 3) Options IA
templates = {
    "📄 Bio LinkedIn":                   "Rédige une bio LinkedIn engageante, professionnelle.",
    "✉️ Mail de candidature spontanée":  "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV":                        "Génère un mini-CV 5-7 lignes, souligne deux mots-clés.",
    "🧩 CV optimisé IA":                 "Rédige un CV optimisé, souligne deux mots-clés.",
}
choices = st.multiselect("🛠️ Génération IA", list(templates.keys()), default=list(templates.keys())[:2])

# --- Helpers IA
def get_gpt_response(prompt, api_key):
    client = OpenAI(api_key=api_key)          # nouvelle interface v1
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":"Tu es un expert en recrutement."},
            {"role":"user",   "content":prompt}
        ]
    )
    return res.choices[0].message.content

def generate_prompt(label, inputs, cv):
    profil = (
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
        profil += f"CV excerpt: {cv[:300]}...\n"
    return profil + "\n" + templates[label]

# --- PDF generator
class PDFGen:
    @staticmethod
    def to_pdf(txt: str):
        buf = io.BytesIO()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in txt.split("\n"):
            pdf.multi_cell(0, 8, line)
        pdf.output(buf)
        buf.seek(0)
        return buf

# --- Pole-Emploi OAuth2 & recherche
def fetch_ft_token(cid, sec):
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
      "grant_type":"client_credentials",
      "client_id":cid,
      "client_secret":sec,
      "scope":"api_offresdemploiv2 o2dsoffre"
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

def search_offres(token, mots, loc, limit=7):
    url = "https://api.pole-emploi.io/partenaire/offresdemploi/v2/offres/search"
    h = {"Authorization":f"Bearer {token}"}
    p = {"motsCles":mots,"localisation":loc,"range":f"0-{limit-1}"}
    r = requests.get(url, headers=h, params=p)
    return r.json().get("resultats",[]) if r.status_code==200 else []

# --- Matching ROME/ESCO (pondération 30/30/40)
@st.cache_data
def load_métier_ref():
    return pd.read_csv("referentiel_metiers_craftmyjob_final.csv")

df_ref = load_métier_ref()
def scorer_metier(inputs, df):
    def score_row(row):
        s1 = fuzz.token_set_ratio(row["Metier"],     inputs["job_title"])
        s2 = fuzz.token_set_ratio(row["Activites"],  inputs["missions"])
        s3 = fuzz.token_set_ratio(row["Competences"],inputs["skills"])
        # nouveau poids : 30% poste / 20% activités / 50% compétences
        return 0.3*s1 + 0.2*s2 + 0.5*s3
    df["score"] = df.apply(score_row, axis=1)
    return df.nlargest(6, "score")

# --- Bouton principal
if st.button("🚀 Générer & Chercher"):
    if not openai_key:
        st.error("🔑 Ta clé OpenAI est requise")
    else:
        inputs = {
            "job_title":job_title, "missions":missions,
            "values":values,     "skills":skills,
            "locations":locations, "experience_level":experience_level,
            "contract_type":contract_type, "remote":remote
        }
        # Générations IA
        for lbl in choices:
            try:
                pr = generate_prompt(lbl, inputs, cv_text)
                out = get_gpt_response(pr, openai_key)
                st.subheader(lbl); st.markdown(out)
                if lbl=="🧩 CV optimisé IA":
                    pdf = PDFGen.to_pdf(out)
                    st.download_button("📥 Télécharger CV", data=pdf,
                                        file_name="CV_optimise.pdf",
                                        mime="application/pdf")
            except Exception as e:
                st.error(f"Erreur IA ({lbl}) : {e}")

        # Recherche offres
        if ft_client_id and ft_secret and locations:
            tok = fetch_ft_token(ft_client_id, ft_secret)
            st.subheader("🔎 Offres trouvées")
            for loc in locations:
                rs = search_offres(tok, f"{job_title} {skills}", loc)
                for o in rs:
                    st.markdown(f"**{o['intitule']}** – {o['entreprise']['nomEntreprise']} – {o['lieuTravail']['libelle']}  \n[Voir]({o['contact']['urlOrigine']})\n---")

        # Matching ROME/ESCO
        st.subheader("🧠 SIS – Matching métiers")
        top6 = scorer_metier(inputs, df_metiers.copy())
        for _,r in top6.iterrows():
            st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
