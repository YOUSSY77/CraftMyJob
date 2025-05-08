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

# ── 1) STREAMLIT CONFIG & STYLING ─────────────────────────────────────────
st.set_page_config(page_title="CraftMyJob – Job Seekers Hub France", layout="centered")
st.markdown("""
<style>
  .stButton>button { background-color:#2E86C1; color:white; border-radius:4px; }
  .section-header { font-size:1.2rem; font-weight:600; margin-top:1.5em; color:#2E86C1; }
  h1,h2,h3 { color:#2E86C1; }
  .offer-link a { color:#2E86C1; text-decoration:none; }
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
def normalize_location(loc: str) -> str:
    """Extract plain name or department for API and filtering."""
    m = re.match(r"^(.+?) \(\d{5}\)", loc)
    if m:
        return m.group(1)
    m2 = re.match(r"Departement (\d{2})", loc)
    if m2:
        return m2.group(1)
    m3 = re.match(r"^(.+) \(region:(\d+)\)", loc)
    if m3:
        return m3.group(1)
    return loc


def search_territoires(query: str, limit: int = 10) -> list:
    """Autocomplete communes, départements, régions via geo.api.gouv.fr."""
    res = []
    if re.fullmatch(r"\d{2}", query):
        r = requests.get(f"https://geo.api.gouv.fr/departements/{query}/communes",
                         params={"fields":"nom,codesPostaux","limit":limit}, timeout=5)
        r.raise_for_status()
        for e in r.json():
            cp = e["codesPostaux"][0] if e["codesPostaux"] else "00000"
            res.append(f"{e['nom']} ({cp})")
        res.append(f"Departement {query}")
    else:
        r1 = requests.get("https://geo.api.gouv.fr/communes",
                          params={"nom":query,"fields":"nom,codesPostaux","limit":limit}, timeout=5)
        if r1.status_code == 200:
            for e in r1.json():
                cp = e["codesPostaux"][0] if e["codesPostaux"] else "00000"
                res.append(f"{e['nom']} ({cp})")
        r2 = requests.get("https://geo.api.gouv.fr/regions",
                          params={"nom":query,"fields":"nom,code"}, timeout=5)
        if r2.status_code == 200:
            for rg in r2.json():
                res.append(f"{rg['nom']} (region:{rg['code']})")
    return list(dict.fromkeys(res))


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


def filter_by_location(offers: list, loc_norm: str) -> list:
    """Keep offers whose lieuTravail.libelle contains loc_norm."""
    filtered = []
    for o in offers:
        lib = o.get('lieuTravail', {}).get('libelle', '')
        if loc_norm.lower() in lib.lower():
            filtered.append(o)
    return filtered


def get_gpt_response(prompt: str, key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {"model":"gpt-3.5-turbo","messages":[{"role":"system","content":"Tu es un expert en recrutement et personal branding."},{"role":"user","content":prompt}],"temperature":0.7,"max_tokens":800}
    r = requests.post(url, json=data, headers=headers, timeout=30); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

class PDFGen:
    @staticmethod
    def to_pdf(text: str) -> io.BytesIO:
        buf = io.BytesIO(); pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
        for line in text.split("\n"): pdf.multi_cell(0, 8, line)
        pdf.output(buf); buf.seek(0); return buf


def fetch_ftoken(cid: str, secret: str) -> str:
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {"grant_type":"client_credentials","client_id":cid,"client_secret":secret,"scope":"api_offresdemploiv2 o2dsoffre"}
    r = requests.post(url, data=data, timeout=10); r.raise_for_status(); return r.json().get("access_token", "")


def search_offres(token: str, mots: str, lieu: str, limit: int=5) -> list:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    r = requests.get(url, headers={"Authorization":f"Bearer {token}"}, params={"motsCles":mots,"localisation":lieu,"range":f"0-{limit-1}"}, timeout=10)
    if r.status_code == 204: return []
    if r.status_code not in (200, 206): st.error(f"FT API {r.status_code}: {r.text}"); return []
    return r.json().get("resultats", [])

# ── 4) PROFILE FORM ──────────────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")
cv_text = ""; up = st.file_uploader("📂 CV (optionnel)", type=["pdf","docx","txt"])
if up:
    ext = up.name.rsplit('.',1)[-1].lower()
    if ext=='pdf': cv_text=" ".join(p.extract_text() or "" for p in PdfReader(up).pages)
    elif ext=='docx': cv_text=" ".join(p.text for p in Document(up).paragraphs)
    else: cv_text=up.read().decode(errors='ignore')
job_title=st.text_input("🔤 Poste souhaité")
missions=st.text_area("📋 Missions principales")
skills=st.text_area("🧠 Compétences clés")
desired_skills=st.text_area("✨ Compétences ciblées (optionnel)")

st.markdown("""<div class='section-header'>🌍 Territoires</div>""",unsafe_allow_html=True)
typed=st.text_input("Tapez commune/département/région…")
opts=search_territoires(typed) if typed else []
default=st.session_state.get('locations',[])
sel=st.multiselect("Sélectionnez vos territoires",options=(default+opts),default=default)
st.session_state.locations=sel

exp_level=st.radio("🎯 Expérience",["Débutant (0-2 ans)","Expérimenté (2-5 ans)","Senior (5+ ans)"])
contract=st.selectbox("📄 Contrat",["CDI","CDD","Freelance","Stage"])
remote=st.checkbox("🏠 Full remote")

st.header("2️⃣ Clés API & IA")
key_openai=st.text_input("🔑 OpenAI API Key",type="password")
key_pe_id=st.text_input("🔑 Pôle-Emploi ID",type="password")
key_pe_secret=st.text_input("🔑 Pôle-Emploi Secret",type="password")

tpls={"Bio LinkedIn":"Rédige une bio LinkedIn professionnelle.","Mail candidature":"Écris un mail de candidature spontanée.","Mini CV":"Génère un mini-CV (5-7 lignes).","CV optimisé IA":"Optimise le CV en soulignant deux mots-clés."}
choices=st.multiselect("Générations IA",list(tpls.keys()),default=list(tpls.keys())[:2])

if st.button("🚀 Lancer"):
    if not key_openai: st.error("Clé OpenAI requise"); st.stop()
    if not(key_pe_id and key_pe_secret and sel): st.error("Identifiants Pôle-Emploi+territoires requis"); st.stop()
    profile={"job_title":job_title,"missions":missions,"skills":skills,"desired_skills":desired_skills}

    # IA
    st.header("🧠 Génération IA")
    for name in choices:
        prm=(f"Poste: {job_title}\nMissions: {missions}\nCompétences: {skills}\n"+ (f"Compétences ciblées: {desired_skills}\n" if desired_skills else "")+f"Territoires: {', '.join(sel)}\nExpérience: {exp_level}\nContrat: {contract}\nTélétravail: {'Oui' if remote else 'Non'}\n\n{tpls[name]}")
        try:
            out=get_gpt_response(prm,key_openai); st.subheader(name); st.markdown(out)
            if name=="CV optimisé IA": buf=PDFGen.to_pdf(out); st.download_button("📥 Télécharger CV optimisé",data=buf,file_name="CV_optimisé.pdf",mime="application/pdf")
        except Exception as e: st.error(f"Erreur IA {name}: {e}")

    # Token
    token=fetch_ftoken(key_pe_id,key_pe_secret)

        # Top offres
    st.header(f"4️⃣ Top offres pour '{job_title}'")
    kw = build_keywords([job_title, skills])
    all_of = []
    for loc in sel:
        loc_norm = normalize_location(loc)
        offres = search_offres(token, kw, loc_norm, limit=5)
        offres = filter_by_location(offres, loc_norm)
        all_of.extend(offres)
    uniq = {}
    for o in all_of:
        url = o.get('contact', {}).get('urlPostulation') or o.get('contact', {}).get('urlOrigine', '')
        if url and url not in uniq:
            uniq[url] = o
    if uniq:
        for url, o in list(uniq.items())[:5]:
            lib = o.get('lieuTravail', {}).get('libelle', '')
            title = o.get('intitule', '–')
            st.markdown(
                f"**{title}** – {lib}  
"
                f"<span class='offer-link'><a href='{url}' target='_blank'>Voir</a></span>
---",
                unsafe_allow_html=True
            )
    else:
        st.info("Aucune offre trouvée...")

    # SIS Métiers
    st.header("5️⃣ SIS – Métiers recommandés")
    top6=scorer_metiers(profile,referentiel,top_k=6)
    for _,r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%"); keym=build_keywords([r['Metier']]); subs=[]
        for loc in sel: loc_norm=normalize_location(loc); tmp=search_offres(token,keym,loc_norm,limit=3); subs.extend(filter_by_location(tmp,loc_norm))
        seen=[]; for o in subs:
            url=o.get('contact',{}).get('urlPostulation') or o.get('contact',{}).get('urlOrigine','');
            if url not in seen: seen.append(url); lib=o.get('lieuTravail',{}).get('libelle',''); dt=o.get('dateCreation','')[:10]; desc=(o.get('description','') or '').replace("\n"," ")[:150]+'…';
            st.markdown(f"• **{o['intitule']}** – {lib} (_Publié {dt}_)  \n{desc}  \n<span class='offer-link'><a href='{url}' target='_blank'>Voir</a></span>",unsafe_allow_html=True)
        if not subs: st.info(f"Aucune offre pour {r['Metier']}...")
