# -*- coding: utf-8 -*-
import os
import re
import io
import streamlit as st
import requests
import pandas as pd
from fpdf import FPDF
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── 0) Supprimer les proxies hérités ────────────────────────────────────────────
for v in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(v, None)

# ── Initialisation session_state pour la sélection des villes ─────────────────
if "locations" not in st.session_state:
    st.session_state.locations = []

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CraftMyJob – by Job Seekers Hub France", layout="centered")
st.title("✨ CraftMyJob")
st.caption("by Job Seekers Hub France 🇫🇷")

# ── Chargement référentiel métiers ─────────────────────────────────────────────
@st.cache_data
def load_metiers() -> pd.DataFrame:
    return pd.read_csv("referentiel_metiers_craftmyjob_final.csv", dtype=str)
df_metiers = load_metiers()

# ── Construction TF-IDF pour SIS ───────────────────────────────────────────────
@st.cache_data
def build_tfidf(df: pd.DataFrame):
    corpus = (
        df["Activites"].fillna("") + " "
        + df["Competences"].fillna("") + " "
        + df["Metier"].fillna("")
    ).tolist()
    vect = TfidfVectorizer(max_features=2000)
    X_ref = vect.fit_transform(corpus)
    return vect, X_ref

vect, X_ref = build_tfidf(df_metiers)

# ── UTILITAIRES ────────────────────────────────────────────────────────────────
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

def fetch_ft_token(cid: str, sec: str) -> str:
    auth_url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "grant_type":    "client_credentials",
        "client_id":     cid,
        "client_secret": sec,
        "scope":         "api_offresdemploiv2 o2dsoffre"
    }
    r = requests.post(auth_url, data=data, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]

def build_keywords(text: str, max_terms: int = 7) -> str:
    words = re.findall(r"\w{2,}", text.lower())
    stop = {"et","ou","la","le","les","de","des","du","un","une","à","en","pour","par","avec","sans","sur","dans","au"}
    seen, keys = set(), []
    for w in words:
        if w in seen or w in stop:
            continue
        seen.add(w); keys.append(w)
        if len(keys) >= max_terms:
            break
    return ",".join(keys)

def search_offres(token: str, mots: str, loc: str, limit: int = 5) -> list:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"motsCles": mots, "localisation": loc, "range": f"0-{limit-1}"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code == 204:
        return []
    if r.status_code not in (200, 206):
        st.error(f"FT API {r.status_code} : {r.text}")
        return []
    return r.json().get("resultats", [])

def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    user_doc = " ".join([inp["missions"], inp["skills"], inp["job_title"]])
    v_user = vect.transform([user_doc])
    cosines = cosine_similarity(v_user, X_ref).flatten()
    df2 = df.copy()
    df2["score"] = (cosines * 100).round(1)
    return df2.nlargest(top_k, "score")

def search_communes(query: str, limit: int = 10) -> list[str]:
    url = "https://geo.api.gouv.fr/communes"
    params = {"nom": query, "fields": "nom,codesPostaux", "boost": "population", "limit": limit}
    r = requests.get(url, params=params, timeout=5)
    r.raise_for_status()
    out = []
    for c in r.json():
        cp = c["codesPostaux"][0] if c["codesPostaux"] else "00000"
        out.append(f"{c['nom']} ({cp})")
    return out

# ── 1️⃣ Que souhaites-tu faire dans la vie ? ─────────────────────────────────────
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

job_title = st.text_input("🔤 Intitulé du poste souhaité")
missions  = st.text_area("📋 Missions principales")
values    = st.text_area("🏢 Valeurs (facultatif)")
skills    = st.text_area("🧠 Compétences clés")

# ── Autocomplete multi-villes via Geo API ───────────────────────────────────────
typed = st.text_input("📍 Commencez à taper une ville…")
raw = search_communes(typed) if typed else []
options = list(dict.fromkeys(st.session_state.locations + raw))
selections = st.multiselect("Sélectionnez une ou plusieurs villes", options, default=st.session_state.locations)
st.session_state.locations = selections

postal_codes = [re.search(r"\((\d{5})\)", loc).group(1) for loc in st.session_state.locations if re.search(r"\((\d{5})\)", loc)]

experience_level = st.radio("🎯 Niveau d'expérience", ["Débutant(e)","Expérimenté(e)","Senior"])
contract_type    = st.selectbox("📄 Type de contrat", ["CDI","Freelance","CDD","Stage"])
remote           = st.checkbox("🏠 Full remote")

# ── 2️⃣ Tes clés API ─────────────────────────────────────────────────────────────
st.header("2️⃣ Tes clés API")
openai_key   = st.text_input("🔑 OpenAI API Key", type="password")
ft_client_id = st.text_input("🔑 Pôle-Emploi Client ID", type="password")
ft_secret    = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

# ── 3️⃣ Générations IA ───────────────────────────────────────────────────────────
st.header("3️⃣ Générations IA")
templates = {
    "📄 Bio LinkedIn":        "Rédige une bio LinkedIn engageante et professionnelle.",
    "✉️ Mail de candidature": "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV":             "Génère un mini-CV (5-7 lignes), souligne deux mots-clés.",
    "🧩 CV optimisé IA":      "Rédige un CV optimisé, souligne deux mots-clés."
}
choices = st.multiselect("Choisis ce que tu veux générer", list(templates), default=list(templates)[:2])

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

# ── 4️⃣ Matching & Offres (deux boutons) ────────────────────────────────────────
inp = {
    "job_title":        job_title,
    "missions":         missions,
    "values":           values,
    "skills":           skills,
    "locations":        st.session_state.locations,
    "experience_level": experience_level,
    "contract_type":    contract_type,
    "remote":           remote
}

# — Bouton A: IA & Top 5 Offres
if st.button("🚀 Générer IA & Offres"):
    # IA
    for lbl in choices:
        try:
            out = get_gpt_response(generate_prompt(lbl, inp, cv_text), openai_key)
            st.subheader(lbl); st.markdown(out)
            if lbl == "🧩 CV optimisé IA":
                pdf = PDFGen.to_pdf(out)
                st.download_button("📥 Télécharger CV", data=pdf, file_name="CV.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"❌ Erreur IA ({lbl}): {e}")

    # Top 5 offres
    token = fetch_ft_token(ft_client_id, ft_secret)
    st.subheader(f"🔎 Top 5 offres pour « {job_title} »")
    mots = build_keywords(job_title + " " + skills)
    all_offres = sum((search_offres(token, mots, cp, 5) for cp in postal_codes), [])
    seen, uniq = set(), []
    for o in all_offres:
        url = o.get("contact",{}).get("urlOrigine","")
        if url and url not in seen:
            seen.add(url); uniq.append(o)
    if uniq:
        for o in uniq[:5]:
            st.markdown(f"**{o['intitule']}** – {o['lieuTravail']['libelle']}\n[Voir]({o['contact']['urlOrigine']})\n---")
    else:
        st.info("🔍 Aucune offre trouvée pour ce poste.")

st.markdown("---")

# — Bouton B: SIS seul
if st.button("🧠 Lancer SIS"):
    token = fetch_ft_token(ft_client_id, ft_secret)
    st.subheader("🧠 SIS – Métiers qui te correspondent")
    top6 = scorer_metier(inp, df_metiers)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        # Top 3 offres par métier
        sub_mots = build_keywords(r["Metier"])
        subs = sum((search_offres(token, sub_mots, cp, 3) for cp in postal_codes), [])
        seen2, uniq2 = set(), []
        for o in subs:
            lien = o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","")
            if lien and lien not in seen2:
                seen2.add(lien); uniq2.append(o)
        if uniq2:
            for o in uniq2[:3]:
                date = o.get("dateCreation","—")[:10]
                desc = (o.get("description","") or "").replace("\n"," ")[:150] + "…"
                st.markdown(
                    f"• **{o['intitule']}**\n"
                    f"  _Publié le {date}_\n"
                    f"  {desc}\n"
                    f"  [Voir / Postuler]({lien})"
                )
        else:
            st.info("• Aucune offre trouvée pour ce métier dans tes villes.")



