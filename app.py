# -*- coding: utf-8 -*-
import os
# ── 0) Supprimer les vars de proxy hérités
for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(v, None)

import streamlit as st
from PIL import Image
import requests
from PyPDF2 import PdfReader
from docx import Document
import io
from fpdf import FPDF
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ── 1) CONFIG PAGE & STYLE ──────────────────────────────────────────────
st.set_page_config(page_title="CraftMyJob – by Job Seekers Hub France", layout="centered")
# logo (mettre logo_jobseekers.png à la racine)
try:
    logo = Image.open("logo_jobseekers.png")
    st.image(logo, width=120)
except:
    pass

st.markdown("""
<style>
  .stButton>button { background-color:#2E86C1; color:white; border-radius:4px; }
  h2, h3 { color:#2E86C1; }
  .section-header { font-size:1.2rem; font-weight:600; margin-top:1.5em; }
</style>
""", unsafe_allow_html=True)

st.title("✨ CraftMyJob")
st.markdown("""
<h2 class="section-header">🧠 SIS – Smart Job Suggestion</h2>
<p>Notre module <strong>SIS</strong> analyse votre profil et vos villes pour vous proposer métiers & offres ciblés.</p>
""", unsafe_allow_html=True)

# ── 2) LOAD MÉTIERS & TF-IDF ─────────────────────────────────────────────
@st.cache_data
def load_metiers() -> pd.DataFrame:
    return pd.read_csv("referentiel_metiers_craftmyjob_final.csv", dtype=str)

df_metiers = load_metiers()

@st.cache_data
def build_tfidf(df: pd.DataFrame):
    corpus = (df["Activites"].fillna("") + " " +
              df["Competences"].fillna("") + " " +
              df["Metier"].fillna("")).tolist()
    vect = TfidfVectorizer(max_features=2000)
    X = vect.fit_transform(corpus)
    return vect, X

vect, X_ref = build_tfidf(df_metiers)

# ── 3) UTILITAIRES ───────────────────────────────────────────────────────
def build_keywords(texts: list[str], max_terms: int = 7) -> str:
    combined = " ".join(texts).lower()
    mots = re.findall(r"\w{2,}", combined, flags=re.UNICODE)
    stop = {"et","ou","la","le","les","de","des","du","un","une",
            "à","en","pour","par","avec","sans","sur","dans","au","aux"}
    vus, clés = set(), []
    for m in mots:
        if m in stop or m in vus:
            continue
        vus.add(m); clés.append(m)
        if len(clés) >= max_terms:
            break
    return ",".join(clés)

def get_gpt_response(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model":"gpt-3.5-turbo",
        "messages":[
            {"role":"system","content":"Tu es un expert en recrutement et en personal branding."},
            {"role":"user","content":prompt}
        ],
        "temperature":0.7,
        "max_tokens":800
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
    data = {"grant_type":"client_credentials","client_id":cid,"client_secret":sec,
            "scope":"api_offresdemploiv2 o2dsoffre"}
    r = requests.post(auth_url, data=data, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]

def search_offres(token: str, mots: str, loc: str, limit: int=5) -> list:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    headers = {"Authorization":f"Bearer {token}"}
    params = {"motsCles":mots,"localisation":loc,"range":f"0-{limit-1}"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code==204: return []
    if r.status_code not in (200,206):
        st.error(f"FT API {r.status_code} : {r.text}")
        return []
    return r.json().get("resultats", [])

def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int=6) -> pd.DataFrame:
    # TF-IDF
    doc = " ".join([inp["missions"], inp["skills"], inp["job_title"]])
    v_user = vect.transform([doc])
    cos = cosine_similarity(v_user, X_ref).flatten()
    df2 = df.copy()
    df2["cosine"] = cos
    # fuzzy
    df2["fz_t"] = df2["Metier"].apply(lambda m: fuzz.token_set_ratio(m,inp["job_title"])/100)
    df2["fz_m"] = df2["Activites"].apply(lambda a: fuzz.token_set_ratio(a,inp["missions"])/100)
    df2["fz_c"] = df2["Competences"].apply(lambda c: fuzz.token_set_ratio(c,inp["skills"])/100)
    df2["score"] = (0.5*df2["cosine"]+0.2*df2["fz_t"]+0.15*df2["fz_m"]+0.15*df2["fz_c"])*100
    return df2.nlargest(top_k,"score")

# ── 4) AUTOCOMPLÉTION VILLES & DÉPARTEMENTS ────────────────────────────────
def search_communes(query: str, limit: int=10) -> list[str]:
    # département : code INSEE à 2 chiffres
    if re.fullmatch(r"\d{2}", query):
        url = f"https://geo.api.gouv.fr/departements/{query}/communes"
        params = {"fields":"nom,codesPostaux","boost":"population","limit":limit}
    else:
        url = "https://geo.api.gouv.fr/communes"
        params = {"nom":query,"fields":"nom,codesPostaux","boost":"population","limit":limit}
    r = requests.get(url, params=params, timeout=5); r.raise_for_status()
    out=[]
    for c in r.json():
        cp = c["codesPostaux"][0] if c["codesPostaux"] else "00000"
        out.append(f"{c['nom']} ({cp})")
    return out

# session-state pour villes
if "locations" not in st.session_state:
    st.session_state.locations = []

# ── 5) FORM PROFILE ────────────────────────────────────────────────────────
st.header("1️⃣ Que souhaites-tu faire dans la vie ?")
uploaded_cv = st.file_uploader("📂 Optionnel : ton CV", type=["pdf","docx","txt"])
cv_text=""
if uploaded_cv:
    ext=uploaded_cv.name.rsplit(".",1)[-1].lower()
    if ext=="pdf":
        cv_text=" ".join(p.extract_text() or "" for p in PdfReader(uploaded_cv).pages)
    elif ext=="docx":
        cv_text=" ".join(p.text for p in Document(uploaded_cv).paragraphs)
    else:
        cv_text=uploaded_cv.read().decode()

job_title=st.text_input("🔤 Intitulé du poste souhaité")
missions=st.text_area("📋 Missions principales")
values=st.text_area("🏢 Valeurs (facultatif)")
skills=st.text_area("🧠 Compétences clés")

typed=st.text_input("📍 Commencez à taper une ville ou code département…")
raw=search_communes(typed) if typed else []
opts=list(dict.fromkeys(st.session_state.locations+raw))
sel=st.multiselect("Sélectionnez villes / départements", opts,
                   default=st.session_state.locations, key="locations")
postal_codes=[re.search(r"\((\d{5})\)",v).group(1)
              for v in st.session_state.locations if re.search(r"\((\d{5})\)",v)]

experience_level=st.radio("🎯 Niveau d'expérience",
                          ["Débutant(e)","Expérimenté(e)","Senior"])
contract_type=st.selectbox("📄 Type de contrat",
                           ["CDI","Freelance","CDD","Stage"])
remote=st.checkbox("🏠 Full remote")

# ── 6) CLÉS & TEMPLATES ────────────────────────────────────────────────────
st.header("2️⃣ Tes clés API")
openai_key  = st.text_input("🔑 OpenAI API Key", type="password")
ft_client_id= st.text_input("🔑 Pôle-Emploi Client ID", type="password")
ft_secret   = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

st.header("3️⃣ Générations IA")
templates={
  "📄 Bio LinkedIn":"Rédige une bio LinkedIn engageante et professionnelle.",
  "✉️ Mail de candidature":"Écris un mail de candidature spontanée clair et convaincant.",
  "📃 Mini CV":"Génère un mini-CV (5-7 lignes), souligne deux mots-clés.",
  "🧩 CV optimisé IA":"Rédige un CV optimisé, souligne deux mots-clés."
}
choices=st.multiselect("Que générer ?", list(templates),
                       default=list(templates)[:2])

def generate_prompt(lbl, inp, cv):
    base=(f"Poste: {inp['job_title']}\n"
          f"Missions: {inp['missions']}\n"
          f"Compétences: {inp['skills']}\n"
          f"Valeurs: {inp['values']}\n"
          f"Localisation: {', '.join(inp['locations'])}\n"
          f"Expérience: {inp['experience_level']}\n"
          f"Contrat: {inp['contract_type']}\n"
          f"Télétravail: {'Oui' if inp['remote'] else 'Non'}\n")
    if cv: base+=f"CV extrait: {cv[:300]}...\n"
    return base+"\n"+templates[lbl]

# ── 7) ACTION ──────────────────────────────────────────────────────────────
if st.button("🚀 Lancer tout"):
    # validations
    if not openai_key:
        st.error("🔑 Clé OpenAI requise"); st.stop()
    if not (ft_client_id and ft_secret and postal_codes):
        st.error("🔑 Identifiants FT + villes requis"); st.stop()

    inp={"job_title":job_title,"missions":missions,
         "values":values,"skills":skills,
         "locations":st.session_state.locations,
         "experience_level":experience_level,
         "contract_type":contract_type,"remote":remote}

    # — IA
    for lbl in choices:
        try:
            out=get_gpt_response(generate_prompt(lbl,inp,cv_text),openai_key)
            st.subheader(lbl); st.markdown(out)
            if lbl=="🧩 CV optimisé IA":
                pdf=PDFGen.to_pdf(out)
                st.download_button("📥 Télécharger PDF",data=pdf,
                                   file_name="CV_optimise.pdf",mime="application/pdf")
        except Exception as e:
            st.error(f"❌ Erreur IA ({lbl}) : {e}")

    # — Top 5 Offres pour le poste
    token   = fetch_ft_token(ft_client_id, ft_secret)
    mots    = build_keywords([job_title,skills])
    st.header(f"4️⃣ Top 5 offres pour « {job_title} »")
    all_of=[]
    for cp in postal_codes:
        all_of += search_offres(token,mots,cp,5)
    seen,uniq=set(),[]
    for o in all_of:
        url=o.get("contact",{}).get("urlOrigine","")
        if url and url not in seen:
            seen.add(url); uniq.append(o)
    if uniq:
        for o in uniq[:5]:
            st.markdown(f"**{o['intitule']}** – {o['lieuTravail']['libelle']}  \n[Voir]({o['contact']['urlOrigine']})\n---")
    else:
        st.info("🔍 Aucune offre trouvée pour ce poste.")

    # — SIS – métiers + offres par métier
    st.header("5️⃣ SIS – Les métiers qui te correspondent")
    top6=scorer_metier(inp,df_metiers,6)
    for _,r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        subs=[]
        cle=r["Metier"]
        mots_m=build_keywords([cle])
        for cp in postal_codes:
            subs+=search_offres(token,mots_m,cp,3)
        s2,u2=set(),[]
        for o in subs:
            link=o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","")
            if link and link not in s2:
                s2.add(link); u2.append(o)
        if u2:
            for o in u2[:3]:
                dt=o.get("dateCreation","—")[:10]
                lien=o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","#")
                desc=(o.get("description","")or"").replace("\n"," ")[:150]+"…"
                st.markdown(f"• **{o['intitule']}**  \n  _Publié le {dt}_  \n  {desc}  \n  [Voir / Postuler]({lien})")
        else:
            st.info("• Aucune offre trouvée pour ce métier dans tes villes.")
