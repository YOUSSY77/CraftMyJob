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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Stop-words FR pour filtrer les mots-clés
FRENCH_STOP_WORDS = {
    "et","ou","la","le","les","de","des","du","un","une",
    "à","en","pour","par","avec","sans","sur","dans","au","aux"
}

# ── 0b) session_state pour multi-villes
if "locations" not in st.session_state:
    st.session_state.locations = []

# ── Config page
st.set_page_config(page_title="CraftMyJob", layout="centered")
st.title("✨ CraftMyJob")
st.caption("by Job Seekers Hub France 🇫🇷")

# ── 1) Chargement référentiel métiers
@st.cache_data
def load_metiers() -> pd.DataFrame:
    return pd.read_csv("referentiel_metiers_craftmyjob_final.csv", dtype=str)
df_metiers = load_metiers()

# ── 2) TF-IDF pour matching SIS
@st.cache_data
def build_tfidf(df):
    corpus = (df["Activites"].fillna("") + " " +
              df["Competences"].fillna("") + " " +
              df["Metier"].fillna("")).tolist()
    vect = TfidfVectorizer(max_features=2000)
    X_ref = vect.fit_transform(corpus)
    return vect, X_ref

vect, X_ref = build_tfidf(df_metiers)

# ── 3) Helpers IA / PDF
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

# ── 4) Auth Pôle-Emploi
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

# ── 5) Filtrer & limiter à 7 mots-clés
def build_keywords(text: str, max_terms: int = 7) -> str:
    words = re.findall(r"\w{2,}", text.lower(), flags=re.UNICODE)
    seen, keys = set(), []
    for w in words:
        if w in seen or w in ENGLISH_STOP_WORDS or w in FRENCH_STOP_WORDS:
            continue
        seen.add(w); keys.append(w)
        if len(keys) >= max_terms:
            break
    return ",".join(keys)

# ── 6) Recherche offres FT corrigée
def search_offres(token: str, mots: str, loc: str, limit: int = 5) -> list:
    url = "https://api.francetravail.io/api/offres/v2/offres/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "motsCles":     build_keywords(mots),
        "localisation": loc,
        "range":        f"0-{limit-1}"
    }
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code == 204:
        return []
    if r.status_code == 401:
        st.error("❌ FT API 401 – identifiants invalides : vérifie ton Client ID/Secret")
        return []
    if r.status_code not in (200, 206):
        st.error(f"❌ FT API {r.status_code} : {r.text}")
        return []
    return r.json().get("resultats", [])

# ── 7) Scoring métiers via cosinus
def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    user_doc = " ".join([inp["missions"], inp["skills"], inp["job_title"]])
    v_user   = vect.transform([user_doc])
    cosines  = cosine_similarity(v_user, X_ref).flatten()
    df2      = df.copy()
    df2["score"] = (cosines * 100).round(1)
    return df2.nlargest(top_k, "score")

# ── 8) generate_prompt manquant
def generate_prompt(label: str, inp: dict, cv: str) -> str:
    base = (
        f"Profil du candidat :\n"
        f"- Intitulé      : {inp['job_title']}\n"
        f"- Missions      : {inp['missions']}\n"
        f"- Compétences   : {inp['skills']}\n"
        f"- Valeurs       : {inp['values']}\n"
        f"- Localisation  : {', '.join(inp['locations'])}\n"
        f"- Niveau        : {inp['experience_level']}\n"
        f"- Contrat       : {inp['contract_type']}\n"
        f"- Télétravail   : {'Oui' if inp['remote'] else 'Non'}\n"
    )
    if cv:
        base += f"- Extrait CV     : {cv[:300]}...\n"
    return base + "\n" + templates[label]

# ── 9) UI : formulaire profil
st.header("1️⃣ Décris ton projet professionnel")
uploaded_cv = st.file_uploader("📂 CV (optionnel)", type=["pdf","docx","txt"])
cv_text = ""
if uploaded_cv:
    ext = uploaded_cv.name.split(".")[-1].lower()
    if ext == "pdf":
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(uploaded_cv).pages)
    elif ext == "docx":
        cv_text = " ".join(p.text for p in Document(uploaded_cv).paragraphs)
    else:
        cv_text = uploaded_cv.read().decode()

job_title = st.text_input("🔤 Poste souhaité")
missions  = st.text_area("📋 Missions principales")
values    = st.text_area("🏢 Valeurs (facultatif)")
skills    = st.text_area("🧠 Compétences clés")

# ── 10) UI : autocomplete multi-villes
def search_communes(query, limit=10):
    url = "https://geo.api.gouv.fr/communes"
    params = {"nom": query, "fields": "nom,codesPostaux", "boost": "population", "limit": limit}
    r = requests.get(url, params=params, timeout=5); r.raise_for_status()
    return [f"{c['nom']} ({c['codesPostaux'][0] if c['codesPostaux'] else '00000'})" for c in r.json()]

typed = st.text_input("📍 Tape une ville…")
raw   = search_communes(typed) if typed else []
opts  = list(dict.fromkeys(st.session_state.locations + raw))
sel   = st.multiselect("Sélectionne tes villes", opts, default=st.session_state.locations)
st.session_state.locations = sel

postal_codes = [re.search(r"\((\d{5})\)", c).group(1) for c in sel if re.search(r"\((\d{5})\)", c)]

experience_level = st.radio("🎯 Niveau", ["Débutant(e)","Expérimenté(e)","Senior"])
contract_type    = st.selectbox("📄 Contrat", ["CDI","Freelance","CDD","Stage"])
remote           = st.checkbox("🏠 Télétravail")

# ── 11) UI : clés API
st.header("2️⃣ Clés API")
openai_key   = st.text_input("🔑 OpenAI Key", type="password")
ft_client_id = st.text_input("🔑 FT Client ID", type="password")
ft_secret    = st.text_input("🔑 FT Secret",   type="password")

# ── 12) UI : choix IA
st.header("3️⃣ Générations IA")
templates = {
    "📄 Bio LinkedIn":         "Rédige une bio LinkedIn engageante et professionnelle.",
    "✉️ Mail de candidature":  "Écris un mail de candidature spontanée clair et convaincant.",
    "📃 Mini CV":              "Génère un mini-CV (5–7 lignes), souligne deux mots-clés.",
    "🧩 CV optimisé IA":       "Rédige un CV optimisé, souligne deux mots-clés."
}
choices = st.multiselect("Que veux-tu générer ?", list(templates), default=list(templates)[:2])

# ── 13) ACTION : Lancer tout
st.header("4️⃣ Matching & Offres")
if st.button("🚀 Lancer tout"):
    # validations
    if not openai_key:
        st.error("🔑 OpenAI Key manquante"); st.stop()
    if not (ft_client_id and ft_secret and postal_codes):
        st.warning("🔑 FT credentials + au moins 1 ville requis"); st.stop()

    inp = {
        "job_title":        job_title,
        "missions":         missions,
        "values":           values,
        "skills":           skills,
        "locations":        sel,
        "experience_level": experience_level,
        "contract_type":    contract_type,
        "remote":           remote
    }

    # — IA
    for lbl in choices:
        try:
            out = get_gpt_response(generate_prompt(lbl, inp, cv_text), openai_key)
            st.subheader(lbl); st.markdown(out)
            if lbl == "🧩 CV optimisé IA":
                pdf = PDFGen.to_pdf(out)
                st.download_button("📥 Télécharger CV", data=pdf,
                                   file_name="CV_optimise.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"❌ Erreur IA ({lbl}) : {e}")

    # — Top 5 offres pour le poste
    token     = None
    try:
        token = fetch_ft_token(ft_client_id, ft_secret)
    except requests.HTTPError:
        st.error("❌ Auth FT échouée : vérifie tes identifiants"); token = None

    if token:
        st.subheader(f"🔎 Top 5 offres pour « {job_title} »")
        offres_all = []
        for cp in postal_codes:
            offres_all += search_offres(token, f"{job_title} {skills}", cp, limit=5)
        seen, uniq = set(), []
        for o in offres_all:
            url = o.get("contact",{}).get("urlOrigine","")
            if url and url not in seen:
                seen.add(url); uniq.append(o)
        if uniq:
            for o in uniq[:5]:
                st.markdown(f"**{o['intitule']}** – {o['lieuTravail']['libelle']}  \n"
                            f"[Voir / Postuler]({o['contact']['urlOrigine']})\n---")
        else:
            st.info("🔍 Aucune offre trouvée.")

        # — SIS
        st.subheader("🧠 SIS – Les métiers qui te correspondent")
        top6 = scorer_metier(inp, df_metiers, top_k=6)
        for _, r in top6.iterrows():
            st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
            subs, seen2, uniq2 = [], set(), []
            for cp in postal_codes:
                subs += search_offres(token, r["Metier"], cp, limit=3)
            for o in subs:
                u2 = o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","")
                if u2 and u2 not in seen2:
                    seen2.add(u2); uniq2.append(o)
            if uniq2:
                for o in uniq2[:3]:
                    date = o.get("dateCreation","—")[:10]
                    lien = o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","#")
                    desc = (o.get("description","") or "").replace("\n"," ")[:150] + "…"
                    st.markdown(
                        f"• **{o['intitule']}**  \n"
                        f"  _Publié le {date}_  \n"
                        f"  {desc}  \n"
                        f"  [Voir / Postuler]({lien})"
                    )
            else:
                st.info("• Aucune offre pour ce métier.")


