# -*- coding: utf-8 -*-
"""
CraftMyJob Pro – Streamlit app for smart job suggestions
"""
import os, io, re, requests
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import streamlit as st
from PIL import Image
from fpdf import FPDF
from PyPDF2 import PdfReader
from docx import Document
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── 0) CLEAN ENVIRONMENT
for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(v, None)

# ── 1) STREAMLIT CONFIG & STYLING
st.set_page_config(page_title="CraftMyJob Pro", layout="centered", page_icon="💼")
st.markdown("""
<style>
  .stButton>button { background-color:#2563EB; color:white; border-radius:6px; transition:all .2s; }
  .stButton>button:hover { background-color:#1D4ED8; transform:translateY(-1px); }
  .pill { display:inline-block; background:#EFF6FF; color:#2563EB;
          padding:4px 10px; border-radius:9999px; margin:2px; font-size:0.85rem; }
  .offer-card { border-left:4px solid #2563EB; padding:8px; margin-bottom:8px; }
  .cv-summary { background:#E8F5E9; padding:12px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)
st.title("✨ CraftMyJob Pro – Votre assistant emploi intelligent")

# ── 2) DATA & TF-IDF CACHE
@st.cache_data
def load_referentiel(path: str = "referentiel_metiers_craftmyjob_final.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")

@st.cache_data
def build_tfidf(df: pd.DataFrame):
    corpus = (df['Activites'].str.lower() + ' ' + df['Competences'].str.lower() + ' ' + df['Metet'].str.lower())
    vect = TfidfVectorizer(max_features=2000)
    mat = vect.fit_transform(corpus)
    return vect, mat

referentiel = load_referentiel()
vecteur, tfidf_matrix = build_tfidf(referentiel)

# ── 3) UTILITIES

def normalize_location(loc: str) -> str:
    m = re.match(r"^(.+?) \((\d{5})\)", loc)
    if m: return m.group(1)
    m = re.match(r"Département (\d{2})", loc)
    if m: return m.group(1)
    m = re.match(r"^(.+) \(region:(\d+)\)", loc)
    if m: return m.group(1)
    return loc

def get_date_range(months: int = 2):
    end = datetime.now().date()
    start = end - timedelta(days=30*months)
    return start.isoformat(), end.isoformat()

def search_territoires(query: str, limit: int = 10) -> List[str]:
    out = []
    # département
    if re.fullmatch(r"\d{2}", query):
        r = requests.get(f"https://geo.api.gouv.fr/departements/{query}/communes",
                         params={"fields":"nom,codesPostaux","limit":limit}, timeout=5)
        r.raise_for_status()
        for c in r.json():
            cp = c['codesPostaux'][0] if c['codesPostaux'] else '00000'
            out.append(f"{c['nom']} ({cp})")
        out.append(f"Département {query}")
        return list(dict.fromkeys(out))
    # communes
    r1 = requests.get("https://geo.api.gouv.fr/communes",
                      params={"nom":query,"fields":"nom,codesPostaux","limit":limit}, timeout=5)
    if r1.status_code == 200:
        for c in r1.json():
            cp = c['codesPostaux'][0] if c['codesPostaux'] else '00000'
            out.append(f"{c['nom']} ({cp})")
    # régions
    r2 = requests.get("https://geo.api.gouv.fr/regions",
                      params={"nom":query,"fields":"nom,code"}, timeout=5)
    if r2.status_code == 200:
        for rg in r2.json():
            out.append(f"{rg['nom']} (region:{rg['code']})")
    return list(dict.fromkeys(out))

def build_keywords(texts: List[str], max_terms: int = 7) -> str:
    tok = re.findall(r"\w{2,}", " ".join(texts).lower())
    stop = {"et","ou","la","le","les","de","des","du","un","une",
            "à","en","pour","par","avec","sans","sur","dans","au","aux"}
    kws, seen = [], set()
    for t in tok:
        if t in stop or t in seen: continue
        seen.add(t); kws.append(t)
        if len(kws) >= max_terms: break
    return ",".join(kws)

def get_gpt(prompt: str, key: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization":f"Bearer {key}"},
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "Tu es un expert en recrutement et personal branding."},
                {"role": "user",   "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

class PDFGen:
    @staticmethod
    def to_pdf(text: str) -> io.BytesIO:
        buf = io.BytesIO(); pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
        for ln in text.split("\n"): pdf.multi_cell(0,8,ln)
        pdf.output(buf); buf.seek(0); return buf

def fetch_ftoken(cid: str, secret: str) -> str:
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {"grant_type":"client_credentials","client_id":cid,
            "client_secret":secret,"scope":"api_offresdemploiv2 o2dsoffre"}
    r = requests.post(url,data=data,timeout=10); r.raise_for_status()
    return r.json().get("access_token","")

def search_offres(token: str, mots: str, lieu: str, limit: int = 5) -> List[Dict]:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    d0, d1 = get_date_range(2)
    params = {"motsCles": mots, "localisation": lieu,
              "range": f"0-{limit-1}",
              "dateDebut": d0, "dateFin": d1,
              "tri": "dateCreation"}
    r = requests.get(url, headers={"Authorization":f"Bearer {token}"}, params=params, timeout=10)
    if r.status_code == 204: return []
    if r.status_code not in (200,206):
        st.error(f"FT API {r.status_code}: {r.text}")
        return []
    return r.json().get("resultats", [])

def filter_by_loc(ofs: List[Dict], loc_norm: str) -> List[Dict]:
    low = loc_norm.lower()
    out = []
    for o in ofs:
        lib = o.get('lieuTravail',{}).get('libelle','').lower()
        cp = str(o.get('lieuTravail',{}).get('codePostal',''))
        if low in lib or low == cp:
            out.append(o)
    return out

def scorer_metier(profile: Dict, df: pd.DataFrame, top_k: int=6) -> pd.DataFrame:
    doc = f"{profile['missions']} {profile['skills']} {profile['job_title']}"
    vec = vecteur.transform([doc]); cos = cosine_similarity(vec, tfidf_matrix).flatten()
    D = df.copy(); D['cosine'] = cos
    D['fz_t'] = D['Metet'] if False else df['Metier'].apply(lambda m: fuzz.token_set_ratio(m,profile['job_title'])/100)
    D['fz_m'] = df['Activites'].apply(lambda a: fuzz.token_set_ratio(a,profile['missions'])/100)
    D['fz_c'] = df['Competences'].apply(lambda c: fuzz.token_set_ratio(c,profile['skills'])/100)
    D['score'] = (0.5*D['cosine'] + 0.2*D['fz_t'] + 0.15*D['fz_m'] + 0.15*D['fz_c'])*100
    return D.nlargest(top_k, 'score')

# ── 4) UI & FORM ──────────────────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")
cv_text = ""
up = st.file_uploader("📂 CV (optionnel)", type=["pdf","docx","txt"])
if up:
    ext = up.name.lower().rsplit('.',1)[-1]
    if ext=='pdf': cv_text = " ".join(p.extract_text() for p in PdfReader(up).pages)
    elif ext=='docx': cv_text = " ".join(p.text for p in Document(up).paragraphs)
    else: cv_text = up.read().decode(errors='ignore')

job_title = st.text_input("🔤 Intitulé souhaité*")
missions = st.text_area("📋 Missions principales*")
skills   = st.text_area("🧠 Compétences clés*")
if skills:
    pills = [f"<span class='pill'>{s.strip()}</span>" for s in skills.split(',') if s.strip()]
    st.markdown(''.join(pills), unsafe_allow_html=True)

st.markdown("---")
st.subheader("🌍 Territoires")
typed = st.text_input("Commune/Département/Région…")
opts  = search_territoires(typed) if typed else []
sel   = st.multiselect("Sélectionnez vos territoires", options=opts, key='locs')

exp      = st.radio("🎯 Expérience", ["Débutant (0-2 ans)", "Expérimenté (2-5 ans)", "Senior (5+ ans)"])
contracts= st.multiselect("📄 Contrats", ["CDI","CDD","Freelance","Stage","Alternance"], default=["CDI","CDD"])
remote   = st.checkbox("🏠 Full Remote")

st.markdown("---")
st.subheader("🔑 Clés API & IA Générateurs")
key_openai   = st.text_input("🔑

