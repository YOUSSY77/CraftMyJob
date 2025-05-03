# -*- coding: utf-8 -*-
import os
# ── 0) On retire tout proxy du contexte
for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
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

# ── Page config
st.set_page_config(
    page_title="CraftMyJob – by Job Seekers Hub France",
    layout="centered"
)
st.title("✨ CraftMyJob")
st.caption("by Job Seekers Hub France 🇫🇷")

# ── Autocomplétion via Geo API du Gouvernement ──────────
def search_communes(query: str, limit: int = 10) -> list[str]:
    """
    Renvoie jusqu'à `limit` suggestions "Nom (CP)" pour la chaîne `query`.
    """
    url = "https://geo.api.gouv.fr/communes"
    params = {
        "nom":      query,
        "fields":   "nom,codesPostaux",
        "boost":    "population",
        "limit":    limit
    }
    r = requests.get(url, params=params, timeout=5)
    r.raise_for_status()
    communes = []
    for c in r.json():
        # on prend le premier code postal
        cp = c["codesPostaux"][0] if c["codesPostaux"] else "00000"
        communes.append(f"{c['nom']} ({cp})")
    return communes

# --- UI : champ texte + multiselect
typed = st.text_input("📍 Commencez à taper une ville…")
suggestions = []
if typed:
    try:
        suggestions = search_communes(typed)
    except Exception:
        suggestions = []
locations = st.multiselect("Sélectionnez une ou plusieurs villes", options=suggestions)
# extraire les CP
postal_codes = [re.search(r"\((\d{5})\)", loc).group(1) 
                for loc in locations if re.search(r"\((\d{5})\)", loc)]


# ── 3) Construction TF-IDF pour matching métier
@st.cache_data
def build_tfidf(df):
    corpus = (
        df["Activites"].fillna("") + " " +
        df["Competences"].fillna("") + " " +
        df["Metier"].fillna("")
    ).tolist()
    vect = TfidfVectorizer(max_features=2000)
    X = vect.fit_transform(corpus)
    return vect, X

vect, X_ref = build_tfidf(df_metiers)

# ── UTILITAIRES ─────────────────────────────────────────

def get_gpt_response(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role":"system", "content":"Tu es un expert en recrutement et en personal branding."},
            {"role":"user",   "content":prompt}
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

def fetch_ft_token(cid: str, sec: str) -> str:
    auth_url = (
        "https://entreprise.pole-emploi.fr"
        "/connexion/oauth2/access_token?realm=/partenaire"
    )
    data = {
        "grant_type":"client_credentials",
        "client_id":cid,
        "client_secret":sec,
        "scope":"api_offresdemploiv2 o2dsoffre"
    }
    r = requests.post(auth_url, data=data, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]

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

def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    user_doc = " ".join([inp["missions"], inp["skills"], inp["job_title"]])
    v_user   = vect.transform([user_doc])
    cosines  = cosine_similarity(v_user, X_ref).flatten()
    df2      = df.copy()
    df2["score"] = (cosines * 100).round(1)
    return df2.nlargest(top_k, "score")

# ── 1️⃣ Que souhaites-tu faire dans la vie ? ─────────────

st.header("1️⃣ Que souhaites-tu faire dans la vie ?")
uploaded_cv     = st.file_uploader("📂 Optionnel : ton CV", type=["pdf","docx","txt"])
cv_text         = ""
if uploaded_cv:
    ext = uploaded_cv.name.rsplit(".",1)[-1].lower()
    if ext=="pdf":
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(uploaded_cv).pages)
    elif ext=="docx":
        cv_text = " ".join(p.text for p in Document(uploaded_cv).paragraphs)
    else:
        cv_text = uploaded_cv.read().decode()

job_title       = st.text_input("🔤 Intitulé du poste souhaité")
missions        = st.text_area( "📋 Missions principales")
values          = st.text_area( "🏢 Valeurs (facultatif)")
skills          = st.text_area( "🧠 Compétences clés")

# autocomplete villes
typed = st.text_input("📍 Commencez à taper une ville…")
suggestions = communes_df[
    communes_df["ville_cp"].str.contains(typed, case=False, na=False)
]["ville_cp"].tolist()[:10]
locations = st.multiselect(
    "Sélectionnez une ou plusieurs villes",
    options=suggestions
)

# extraction CP
postal_codes = []
for loc in locations:
    m = re.search(r"\((\d{5})\)", loc)
    if m: postal_codes.append(m.group(1))

experience_level = st.radio("🎯 Niveau d'expérience", ["Débutant(e)","Expérimenté(e)","Senior"])
contract_type    = st.selectbox("📄 Type de contrat", ["CDI","Freelance","CDD","Stage"])
remote           = st.checkbox("🏠 Full remote")

# ── 2️⃣ Tes clés API ────────────────────────────────────

st.header("2️⃣ Tes clés API")
openai_key   = st.text_input("🔑 OpenAI API Key", type="password")
ft_client_id = st.text_input("🔑 Pôle-Emploi Client ID", type="password")
ft_secret    = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

# ── 3️⃣ Générations IA ──────────────────────────────────

st.header("3️⃣ Générations IA")
templates = {
    "📄 Bio LinkedIn":          "Rédige une bio LinkedIn engageante et professionnelle.",
    "✉️ Mail de candidature":   "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV":               "Génère un mini-CV (5-7 lignes), souligne deux mots-clés.",
    "🧩 CV optimisé IA":        "Rédige un CV optimisé, souligne deux mots-clés."
}
choices = st.multiselect(
    "Choisis les éléments à générer",
    list(templates.keys()),
    default=list(templates.keys())[:2]
)

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

# ── 4️⃣ Matching & Offres ───────────────────────────────

st.header("4️⃣ Matching & Offres")

if st.button("🚀 Lancer tout"):
    # validations
    if not openai_key:
        st.error("🔑 Clé OpenAI requise"); st.stop()
    if not (ft_client_id and ft_secret and postal_codes):
        st.error("🔑 Identifiants Pôle-Emploi + villes requis"); st.stop()

    inp = {
        "job_title": job_title,
        "missions":  missions,
        "values":    values,
        "skills":    skills,
        "locations": locations,
        "experience_level": experience_level,
        "contract_type":    contract_type,
        "remote":           remote
    }

    # — IA
    for lbl in choices:
        try:
            prompt = generate_prompt(lbl, inp, cv_text)
            out    = get_gpt_response(prompt, openai_key)
            st.subheader(lbl)
            st.markdown(out)
            if lbl == "🧩 CV optimisé IA":
                pdf = PDFGen.to_pdf(out)
                st.download_button(
                    "📥 Télécharger PDF", data=pdf,
                    file_name="CV_optimise.pdf", mime="application/pdf"
                )
        except Exception as e:
            st.error(f"❌ Erreur IA ({lbl}) : {e}")

    # — Offres pour le poste saisi (top 5)
    token     = fetch_ft_token(ft_client_id, ft_secret)
    mots_cles = f"{job_title} {skills}"
    st.subheader(f"🔎 Top 5 offres pour « {job_title} »")
    all_of = []
    for cp in postal_codes:
        all_of += search_offres(token, mots_cles, cp, limit=5)
    seen, uniq = set(), []
    for o in all_of:
        url = o.get("contact",{}).get("urlOrigine","")
        if url and url not in seen:
            seen.add(url); uniq.append(o)
    if uniq:
        for o in uniq[:5]:
            st.markdown(
                f"**{o['intitule']}** – {o['lieuTravail']['libelle']}  \n"
                f"[Voir]({o['contact']['urlOrigine']})\n---"
            )
    else:
        st.info("🔍 Aucune offre trouvée pour ce poste.")

    # — SIS : Top 6 métiers + Top 3 offres par métier
    st.subheader("🧠 SIS – Les métiers qui te correspondent")
    top6 = scorer_metier(inp, df_metiers.copy(), top_k=6)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        sub_of = search_offres(token, r["Metier"], postal_codes[0], limit=3)
        for s in sub_of:
            st.write(f"  • {s['intitule']} ({s['lieuTravail']['libelle']})")
