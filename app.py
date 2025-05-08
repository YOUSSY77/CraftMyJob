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

# ── 1) STREAMLIT CONFIG & STYLING ──────────────────────────────────────────
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
except Exception:
    pass

st.title("✨ CraftMyJob – Votre assistant emploi intelligent")

# ── 2) DATA & MODEL PREP ─────────────────────────────────────────────────
@st.cache_data
def load_referentiel(path: str = "referentiel_metiers_craftmyjob_final.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")

@st.cache_data
def build_tfidf(df: pd.DataFrame, max_features: int = 2000):
    corpus = df["Activites"] + " " + df["Competences"] + " " + df["Metier"]
    vect = TfidfVectorizer(max_features=max_features)
    matrix = vect.fit_transform(corpus)
    return vect, matrix

referentiel = load_referentiel()
vecteur, tfidf_matrix = build_tfidf(referentiel)

# ── 3) UTILITIES ─────────────────────────────────────────────────────────
def build_keywords(texts: list[str], max_terms: int = 7) -> str:
    combined = " ".join(texts).lower()
    tokens = re.findall(r"\w{2,}", combined)
    stop = {"et","ou","la","le","les","de","des","du","un","une","à","en","pour","par","avec","sans","sur","dans","au","aux"}
    seen, kws = set(), []
    for t in tokens:
        if t in stop or t in seen:
            continue
        seen.add(t)
        kws.append(t)
        if len(kws) >= max_terms:
            break
    return ",".join(kws)


def get_gpt_response(prompt: str, key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Tu es un expert en recrutement et personal branding."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    r = requests.post(url, json=data, headers=headers, timeout=30)
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


def fetch_ftoken(cid: str, secret: str) -> str:
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {"grant_type":"client_credentials","client_id":cid,"client_secret":secret,"scope":"api_offresdemploiv2 o2dsoffre"}
    r = requests.post(url, data=data, timeout=10)
    r.raise_for_status()
    return r.json().get("access_token","")


def search_offres(token: str, mots: str, lieu: str, limit: int=5) -> list:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    hdr = {"Authorization": f"Bearer {token}"}
    prm = {"motsCles": mots, "localisation": lieu, "range": f"0-{limit-1}"}
    r = requests.get(url, headers=hdr, params=prm, timeout=10)
    if r.status_code == 204:
        return []
    if r.status_code not in (200,206):
        st.error(f"FT API {r.status_code}: {r.text}")
        return []
    return r.json().get("resultats", [])


def scorer_metiers(profile: dict, df: pd.DataFrame, top_k: int=6) -> pd.DataFrame:
    doc = " ".join([profile["missions"], profile["skills"], profile["job_title"], profile.get("desired_skills","")])
    v = vecteur.transform([doc])
    cos = cosine_similarity(v, tfidf_matrix).flatten()
    df2 = df.copy()
    df2["cosine"] = cos
    df2["fz_t"] = df2["Metier"].apply(lambda m: fuzz.token_set_ratio(m, profile["job_title"]) / 100)
    df2["fz_m"] = df2["Activites"].apply(lambda a: fuzz.token_set_ratio(a, profile["missions"]) / 100)
    df2["fz_c"] = df2["Competences"].apply(lambda c: fuzz.token_set_ratio(c, profile["skills"]) / 100)
    df2["score"] = (0.5*df2["cosine"] + 0.2*df2["fz_t"] + 0.15*df2["fz_m"] + 0.15*df2["fz_c"]) * 100
    return df2.nlargest(top_k, "score").reset_index(drop=True)


def search_territoires(query: str, limit: int=10) -> list[str]:
    res = []
    if re.fullmatch(r"\d{2}", query):
        r = requests.get(f"https://geo.api.gouv.fr/departements/{query}/communes", params={"fields":"nom,codesPostaux","limit":limit}, timeout=5)
        r.raise_for_status()
        for e in r.json():
            cp = e["codesPostaux"][0] if e["codesPostaux"] else "00000"
            res.append(f"{e['nom']} ({cp})")
        res.append(f"Departement {query}")
    else:
        r1 = requests.get("https://geo.api.gouv.fr/communes", params={"nom":query,"fields":"nom,codesPostaux","limit":limit}, timeout=5)
        if r1.status_code == 200:
            for e in r1.json():
                cp = e["codesPostaux"][0] if e["codesPostaux"] else "00000"
                res.append(f"{e['nom']} ({cp})")
        r2 = requests.get("https://geo.api.gouv.fr/regions", params={"nom":query,"fields":"nom,code"}, timeout=5)
        if r2.status_code == 200:
            for rg in r2.json():
                res.append(f"{rg['nom']} (region:{rg['code']})")
    return list(dict.fromkeys(res))

# ── 4) PROFILE FORM ───────────────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")
# CV upload
cv_text = ""
up = st.file_uploader("📂 CV (optionnel)", type=["pdf","docx","txt"])
if up:
    ext = up.name.split('.')[-1].lower()
    if ext == 'pdf': cv_text = " ".join(p.extract_text() or "" for p in PdfReader(up).pages)
    elif ext == 'docx': cv_text = " ".join(p.text for p in Document(up).paragraphs)
    else: cv_text = up.read().decode(errors='ignore')

job_title = st.text_input("🔤 Poste souhaité")
missions  = st.text_area("📋 Missions principales")
skills    = st.text_area("🧠 Compétences clés")
desired_skills = st.text_area("✨ Compétences ciblées (facultatif)")

# Territoires
st.markdown("""<div class='section-header'>🌍 Territoires</div>""", unsafe_allow_html=True)
typed = st.text_input("Tapez commune/département/région…")
opts = search_territoires(typed) if typed else []
default = st.session_state.get('locations', [])
sel = st.multiselect("Sélectionnez vos territoires", options=(default+opts), default=default)
st.session_state.locations = sel

# Expérience & contrat
exp_level = st.radio("🎯 Expérience", ["Débutant (0-2 ans)","Expérimenté (2-5 ans)","Senior (5+ ans)"])
contract  = st.selectbox("📄 Contrat", ["CDI","CDD","Freelance","Stage"])
remote    = st.checkbox("🏠 Full remote")

# ── 5) API & IA TEMPLATES ─────────────────────────────────────────────────
st.header("2️⃣ Clés API & Génération IA")
key_openai   = st.text_input("🔑 OpenAI API Key", type="password")
key_pe_id    = st.text_input("🔑 Pôle-Emploi ID",    type="password")
key_pe_secret= st.text_input("🔑 Pôle-Emploi Secret",type="password")

tpls = {
    "Bio LinkedIn": "Rédige une bio LinkedIn professionnelle.",
    "Mail candidature": "Écris un mail de candidature spontanée.",
    "Mini CV": "Génère un mini-CV (5-7 lignes).",
    "CV optimisé IA": "Optimise le CV en soulignant deux mots-clés."  
}
choices = st.multiselect("Générations IA", list(tpls.keys()), default=list(tpls.keys())[:2])

# ── 6) ACTION ────────────────────────────────────────────────────────────
if st.button("🚀 Lancer"):
    if not key_openai:
        st.error("Clé OpenAI requise")
        st.stop()
    if not (key_pe_id and key_pe_secret and sel):
        st.error("Identifiants Pôle-Emploi + territoires requis")
        st.stop()

    profile = {"job_title":job_title,"missions":missions,"skills":skills,
               "desired_skills":desired_skills}

    # IA
    st.header("🧠 Génération IA")
    for name in choices:
        prm = (f"Poste: {job_title}\nMissions: {missions}\nCompétences: {skills}\n"+
               (f"Compétences ciblées: {desired_skills}\n" if desired_skills else "")+
               f"Territoires: {', '.join(sel)}\nExpérience: {exp_level}\nContrat: {contract}\nTélétravail: {'Oui' if remote else 'Non'}\n\n{tpls[name]}")
        try:
            out = get_gpt_response(prm, key_openai)
            st.subheader(name)
            st.markdown(out)
            if name == "CV optimisé IA":
                pdf = PDFGen.to_pdf(out)
                st.download_button("📥 Télécharger CV optimisé", data=pdf, file_name="CV_optimisé.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Erreur IA {name}: {e}")

    # Pôle-Emploi token
    token = fetch_ftoken(key_pe_id, key_pe_secret)

    # Top offres Poste
    st.header(f"4️⃣ Top offres pour '{job_title}'")
    kw = build_keywords([job_title, skills])
    offers = []
    for lieu in sel:
        offers += search_offres(token, kw, lieu, limit=5)
    unique = {}
    for o in offers:
        url = o.get('contact',{}).get('urlOrigine','')
        if url and url not in unique:
            unique[url] = o
    if unique:
        for o in list(unique.values())[:5]:
            st.markdown(f"**{o['intitule']}** – {o['lieuTravail']['libelle']}  \n[Voir]({o['contact']['urlOrigine']})\n---")
    else:
        st.info("Aucune offre trouvée.")

    # SIS Métiers
    st.header("5️⃣ SIS – Métiers recommandés")
    top6 = scorer_metiers(profile, referentiel, top_k=6)
    for _,r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        keym = build_keywords([r['Metier']])
        subs=[]
        for lieu in sel:
            subs += search_offres(token, keym, lieu, limit=3)
        seen, u = set(), []
        for o in subs:
            link = o.get('contact',{}).get('urlPostulation') or o.get('contact',{}).get('urlOrigine','')
            if link and link not in seen:
                seen.add(link); u.append(o)
        if u:
            for o in u[:3]:
                dt=o.get('dateCreation','')[:10]
                desc=(o.get('description','') or '').replace('\n',' ')[:150]+'…'
                st.markdown(f"• **{o['intitule']}** (_Publié {dt}_)  \n{desc}  \n[Voir]({link})")
        else:
            st.info(f"Aucune offre pour {r['Metier']}.")
