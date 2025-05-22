# -*- coding: utf-8 -*-
"""
CraftMyJob – Streamlit app for smart job suggestions
Optimizations: geo-filter, TFIDF+fuzzy mix, SIS fusion, contract fallback, pagination
"""
import os
import io
import re
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from fpdf import FPDF
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ── 0) CLEAN ENVIRONMENT
for var in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(var, None)

# ── 1) STREAMLIT CONFIG & STYLE
st.set_page_config(page_title="CraftMyJob – Job Seekers Hub France", layout="centered")
st.markdown('''
<style>
  .stButton>button { background-color:#2E86C1; color:white; border-radius:4px; }
  h1,h2,h3 { color:#2E86C1; }
</style>
''', unsafe_allow_html=True)

# Logo
try:
    st.image(Image.open("logo_jobseekers.PNG"), width=100)
except:
    pass

st.title("✨ CraftMyJob – Votre assistant emploi intelligent")

# ── 2) LOAD DATA & TFIDF
@st.cache_data
def load_referentiel(path: str="referentiel_metiers_craftmyjob_final.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")

referentiel = load_referentiel()
# build TF-IDF on metiers corpus
@st.cache_data
def build_tfidf(df: pd.DataFrame):
    corpus = (df['Activites'] + ' ' + df['Competences'] + ' ' + df['Metier']).tolist()
    vect = TfidfVectorizer(max_features=2000)
    mat = vect.fit_transform(corpus)
    return vect, mat

vecteur, tfidf_matrix = build_tfidf(referentiel)

# ── 3) UTILITIES

def normalize_location(loc: str) -> str:
    # extract plain name or dept code
    if m:= re.match(r"^(.+?) \((\d{5})\)", loc):
        return m.group(2)  # use postal code for filtering
    if m:= re.match(r"Département (\d{2})", loc):
        return m.group(1)
    if m:= re.match(r"^(.+) \(region:(\d+)\)", loc):
        return m.group(2)
    return loc

# FT token
@st.cache_data
def fetch_ftoken(cid: str, secret: str) -> str:
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {"grant_type":"client_credentials","client_id":cid,"client_secret":secret,
            "scope":"api_offresdemploiv2 o2dsoffre"}
    r = requests.post(url, data=data, timeout=10)
    r.raise_for_status()
    return r.json().get('access_token','')

# Search offres with pagination
def search_offres(token: str, mots: str, loc: str, range_max: int=29) -> pd.DataFrame:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    offerings = []
    step = 30
    for start in range(0, range_max+1, step):
        end = start + step -1
        params = {"motsCles": mots, "localisation": loc, "range": f"{start}-{end}"}
        r = requests.get(url, headers={"Authorization":f"Bearer {token}"}, params=params, timeout=10)
        if r.status_code==204: break
        if r.status_code not in (200,206):
            st.error(f"FT API {r.status_code}: {r.text}")
            break
        offerings += r.json().get('resultats', [])
        if len(offerings) >= range_max+1:
            break
    return pd.DataFrame(offerings)

# filter by geo: postal code startswith dept or fallback in text
def filter_by_location(df: pd.DataFrame, loc_norm: str) -> pd.DataFrame:
    if df.empty: return df
    mask = df['lieuTravail'].apply(lambda lt: str(lt.get('codePostal','')).startswith(loc_norm))
    return df[mask | df['lieuTravail'].apply(lambda lt: loc_norm.lower() in lt.get('libelle','').lower())]

# scoring offres: mix TFIDF+fuzzy
def score_offres(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if df.empty: return df
    corpus = df['intitule'] + ' ' + df.get('description','')
    X = vecteur.transform([query])
    Y = vecteur.transform(corpus)
    cos = cosine_similarity(X, Y).flatten()
    fuzzies = df['intitule'].apply(lambda x: fuzz.WRatio(x, query)/100)
    df2 = df.copy()
    df2['score_mix'] = 0.7*cos + 0.3*fuzzies
    return df2.sort_values('score_mix', ascending=False)

# build keywords from profile
def build_keywords(texts: list[str], max_terms: int=7) -> str:
    words = re.findall(r"\w{2,}", ' '.join(texts).lower())
    stop = set(["et","ou","le","la","de","des","du","un","une"])
    kws=[]
    for w in words:
        if w in stop: continue
        if w not in kws: kws.append(w)
        if len(kws)>=max_terms: break
    return ' '.join(kws)

# SIS scoring: fuse metier_score + freq of romeCode
@st.cache_data
def load_rome_offres():
    return pd.DataFrame()  # placeholder if needed

def score_sis(profile: dict, referentiel: pd.DataFrame, offres: pd.DataFrame) -> pd.DataFrame:
    # TFIDF+fuzzy against referentiel
    df_sis = scorer_metier(profile, referentiel, top_k=len(referentiel))
    # count romeCode freq in offres
    freq = offres['romeCode'].value_counts().to_dict()
    maxf = max(freq.values()) if freq else 1
    df_sis['freq_norm'] = df_sis['romeCode'].map(lambda r: freq.get(r,0)/maxf)
    df_sis['final_score'] = 0.7*(df_sis['score']/100) + 0.3*df_sis['freq_norm']
    return df_sis.sort_values('final_score', ascending=False).head(6)

# keep original scorer_metier

def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int=6) -> pd.DataFrame:
    doc = ' '.join([inp['job_title'], inp['missions'], inp['skills']])
    v = vecteur.transform([doc]); cos = cosine_similarity(v, tfidf_matrix).flatten()
    d2 = df.copy(); d2['cosine']=cos
    d2['fz'] = d2['Metier'].apply(lambda m: fuzz.WRatio(m, inp['job_title'])/100)
    d2['score'] = 0.7*d2['cosine'] + 0.3*d2['fz']
    return d2.sort_values('score', ascending=False).head(top_k)

# ── 4) UI: Profile Form
st.header("1️⃣ Profil & préférences")
job_title = st.text_input("🔤 Poste souhaité")
missions = st.text_area("📋 Missions principales")
skills = st.text_area("🧠 Compétences clés")

st.subheader("🌍 Territoires")
typed = st.text_input("Commune ou département (2 chiffres)…")
# simple dept support
territoires = [typed] if re.fullmatch(r"\d{2}", typed) else [typed]
sel = st.multiselect("Sélectionnez", territoires, default=territoires)

ft_id = st.text_input("🔑 Pôle-Emploi Client ID", type="password")
ft_sec= st.text_input("🔑 Pôle-Emploi Secret", type="password")

if st.button("🚀 Lancer tout"):
    # validations
    if not all([job_title, missions, skills, ft_id, ft_sec, sel]):
        st.error("Veuillez renseigner tous les champs requis."); st.stop()
    # fetch token
    try:
        token=fetch_ftoken(ft_id, ft_sec)
    except requests.HTTPError as e:
        st.error(f"Erreur FT: {e.response.status_code}"); st.stop()

    # build keywords
    mots=build_keywords([job_title, skills])
    # collect offres
    all_df=pd.DataFrame()
    for loc in sel:
        locn = normalize_location(loc)
        df_o = search_offres(token, mots, locn)
        df_o = filter_by_location(df_o, locn)
        all_df = pd.concat([all_df, df_o], ignore_index=True)
    # unique
    all_df.drop_duplicates(subset=['intitule','lieuTravail'], inplace=True)
    # score offres
    scored=score_offres(all_df, job_title)
    st.header("🎯 Top Offres")
    for _,row in scored.head(30).iterrows():
        title=row['intitule']; lib=row['lieuTravail']['libelle']; code=row['lieuTravail']['codePostal']
        link = row.get('contact',{}).get('urlPostulation') or row.get('contact',{}).get('urlOrigine','')
        st.markdown(f"**{title}** – {lib} [{code}]\n[Voir >>]({link})")

    # SIS métiers
    st.header("🧩 SIS – Métiers recommandés")
    sis_df = score_sis({'job_title':job_title,'missions':missions,'skills':skills}, referentiel, all_df)
    for _,r in sis_df.iterrows():
        st.markdown(f"**{r['Metier']}** – Score {r['final_score']:.2f}")
