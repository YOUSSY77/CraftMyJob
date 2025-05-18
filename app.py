# -*- coding: utf-8 -*-
"""
CraftMyJob – Streamlit app for smart job suggestions
"""
import os
import re
import io
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from datetime import datetime, timedelta

# ── 0) CLEAN ENVIRONMENT ─────────────────────────────────────────────────
for var in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(var, None)

# ── 1) STREAMLIT CONFIG & STYLING ─────────────────────────────────────────
st.set_page_config(page_title="CraftMyJob – Job Seekers Hub France", layout="centered")
st.markdown("""
<style>
  .stButton>button { background-color:#2E86C1; color:white; border-radius:4px; }
  .section-header { font-size:1.2rem; font-weight:600; margin-top:1.5em; color:#2E86C1; }
  h1,h2,h3 { color:#2E86C1; }
  .offer-link a { color:#2E86C1; text-decoration:none; }
  .cv-summary { color:#1F8A70; }
  .ats-tag { background:#D6EAF8; padding:4px 8px; border-radius:4px; margin-right:4px; display:inline-block; }
</style>
""", unsafe_allow_html=True)
try:
    logo = Image.open("logo_jobseekers.PNG")
    st.image(logo, width=120)
except:
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
@st.cache_data
def search_territoires(query: str, limit: int=10) -> list[str]:
    res=[]
    if re.fullmatch(r"\d{2}", query):
        r=requests.get(f"https://geo.api.gouv.fr/departements/{query}/communes", params={"fields":"nom,codesPostaux","limit":limit}, timeout=5)
        r.raise_for_status()
        for e in r.json(): cp=e.get("codesPostaux",["00000"])[0]; res.append(f"{e['nom']} ({cp})")
        res.append(f"Département {query}")
        return list(dict.fromkeys(res))
    for endpoint,field in [("communes","nom"),("regions","nom")]:
        url=f"https://geo.api.gouv.fr/{endpoint}"
        params={"nom":query, "fields": ("nom,codesPostaux" if endpoint=="communes" else "nom,code"), "limit":limit}
        r=requests.get(url, params=params, timeout=5)
        if r.ok:
            for e in r.json():
                if endpoint=="communes": cp=e.get("codesPostaux",["00000"])[0]; res.append(f"{e['nom']} ({cp})")
                else: res.append(f"{e['nom']} (region:{e['code']})")
    return list(dict.fromkeys(res))

def build_keywords(texts: list[str], max_terms: int=5) -> str:
    combined=" ".join(texts).lower()
    tokens=re.findall(r"\w{2,}",combined)
    stop={"et","ou","la","le","les","de","des","du","un","une","à","en","pour","par","avec","sans","sur","dans","au","aux"}
    seen,kws=set(),[]
    for t in tokens:
        if t in stop or t in seen: continue
        seen.add(t); kws.append(t)
        if len(kws)>=max_terms: break
    return ",".join(kws)

def normalize_location(loc: str) -> str:
    if m:=re.match(r"^(.+?) \((\d{5})\)",loc): return m.group(1)
    if m2:=re.match(r"Département (\d{2})",loc): return m2.group(1)
    if m3:=re.match(r"^(.+) \(region:(\d+)\)",loc): return m3.group(1)
    return loc

def get_date_range(m: int=2):
    end=datetime.now().date(); start=end-timedelta(days=30*m)
    return start.isoformat(), end.isoformat()

def get_gpt_response(prompt: str,key: str) -> str:
    url="https://api.openai.com/v1/chat/completions"
    headers={"Authorization":f"Bearer {key}"}
    data={"model":"gpt-3.5-turbo","messages":[{"role":"system","content":"Tu es un expert en recrutement."},{"role":"user","content":prompt}],"temperature":0.7,"max_tokens":800}
    r=requests.post(url,json=data,headers=headers,timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def fetch_ftoken(cid: str,secret: str) -> str:
    r=requests.post("https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire", data={"grant_type":"client_credentials","client_id":cid,"client_secret":secret,"scope":"api_offresdemploiv2 o2dsoffre"},timeout=10)
    r.raise_for_status(); return r.json().get("access_token","")

def fetch_all_offres(token: str, mots: str, lieu: str, batch_size: int=100, max_batches: int=5) -> list:
    url="https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    d0,d1=get_date_range(2)
    all_off=[]
    for i in range(max_batches):
        start=i*batch_size; end=start+batch_size-1
        params={"motsCles":mots,"localisation":lieu,"range":f"{start}-{end}","dateDebut":d0,"dateFin":d1,"tri":"pertinence"}
        r=requests.get(url,params=params,headers={"Authorization":f"Bearer {token}"},timeout=10)
        if r.status_code not in (200,206): break
        res=r.json().get("resultats",[]); all_off+=res
        if len(res)<batch_size: break
    return all_off

def filter_by_location(offers: list,loc_norm: str) -> list:
    ln=loc_norm.lower(); return [o for o in offers if ln in o.get("lieuTravail_libelle",""
).lower()]

def scorer_metier(inp: dict,df: pd.DataFrame,top_k: int=6) -> pd.DataFrame:
    doc=f"{inp['missions']} {inp['skills']} {inp['job_title']}"
    v=vecteur.transform([doc]); cos=cosine_similarity(v,tfidf_matrix).flatten()
    df2=df.copy(); df2['cosine']=cos
    df2['fz_t']=df2['Metier'].apply(lambda m:fuzz.WRatio(m,inp['job_title'])/100)
    df2['fz_m']=df2['Activites'].apply(lambda a:fuzz.partial_ratio(a,inp['missions'])/100)
    df2['fz_c']=df2['Competences'].apply(lambda c:fuzz.partial_ratio(c,inp['skills'])/100)
    df2['score']=(0.5*df2['cosine']+0.2*df2['fz_t']+0.15*df2['fz_m']+0.15*df2['fz_c'])*100
    return df2.nlargest(top_k,'score')

# ── 4) PROFILE FORM ──────────────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")
cv_text=""; up=st.file_uploader("📂 CV (optionnel)",type=["pdf","docx","txt"])
if up:
    ext=up.name.rsplit('.',1)[-1].lower()
    if ext=="pdf": cv_text=" ".join(p.extract_text() or "" for p in PdfReader(up).pages)
    elif ext=="docx": cv_text=" ".join(p.text for p in Document(up).paragraphs)
    else: cv_text=up.read().decode(errors="ignore")
job_title=st.text_input("🔤 Poste souhaité")
missions=st.text_area("📋 Missions principales")
skills=st.text_area("🧠 Compétences clés")
st.markdown("<div class='section-header'>🌍 Territoires</div>",unsafe_allow_html=True)
typed=st.text_input("Tapez commune/département/région…"); opts=search_territoires(typed) if typed else []
def_locs=st.session_state.get('locations',[])
sel=st.multiselect("Sélectionnez vos territoires",options=(def_locs+opts),default=def_locs)
st.session_state['locations']=sel
exp_level=st.radio("🎯 Expérience",["Débutant (0-2 ans)","Expérimenté (2-5 ans)","Senior (5+ ans)"])
contract=st.multiselect("📄 Types de contrat",["CDI","CDD","Freelance","Stage","Alternance"],default=["CDI","CDD","Freelance"])
remote=st.checkbox("🏠 Full remote")
# ── 5) API KEYS ─────────────────────────────────────────────────────────
st.header("2️⃣ Clés API & IA")
key_openai=st.text_input("🔑 OpenAI API Key",type="password")
key_pe_id=st.text_input("🔑 Pôle-Emploi Client ID",type="password")
key_pe_secret=st.text_input("🔑 Pôle-Emploi Client Secret",type="password")
# ── 6) ACTION ─────────────────────────────────────────────────────────
if st.button("🚀 Lancer tout"):
    if not key_openai: st.error("Clé OpenAI requise"); st.stop()
    if not (key_pe_id and key_pe_secret and sel): st.error("ID Pôle-Emploi & territoires requis"); st.stop()
    profile={'job_title':job_title,'missions':missions,'skills':skills}
    if cv_text:
        summary=get_gpt_response(f"Résumé en 5 points clés du CV:\n{cv_text[:1000]}",key_openai)
        st.markdown("**Résumé CV:**",unsafe_allow_html=True)
        for l in summary.split("\n"): st.markdown(f"- <span class='cv-summary'>{l}</span>",unsafe_allow_html=True)
    st.header("🧠 Génération IA")
    tpls={'📄 Bio LinkedIn':"Rédige une bio LinkedIn professionnelle.", '✉️ Mail de candidature':"Écris un mail de candidature spontanée.", '📃 Mini CV':"Génère un mini-CV (5-7 lignes)."}
    choices=st.multiselect("Générations IA",list(tpls.keys()),default=list(tpls.keys())[:2])
    for name in choices:
        inst=tpls[name] + (" Ne mentionne ni localisation ni ancienneté." if name.startswith('📄') else "")
        prompt_lines=[f"Poste: {job_title}",f"Missions: {missions}",f"Compétences: {skills}"]
        if cv_text: prompt_lines.append(f"Résumé CV: {summary[:300]}")
        prompt_lines += [f"Territoires: {', '.join(sel)}",f"Expérience: {exp_level}",f"Contrat(s): {', '.join(contract)}",f"Télétravail: {'Oui' if remote else 'Non'}","",inst]
        res=get_gpt_response("\n".join(prompt_lines),key_openai)
        st.subheader(name); st.markdown(res)
    token=fetch_ftoken(key_pe_id.strip(),key_pe_secret.strip())
    st.header(f"4️⃣ Top 30 offres pour '{job_title}'")
    # ATS keywords
    ats=build_keywords([missions,skills],max_terms=5)
    st.markdown("**Mots-clés recommandés :**")
    for tag in ats.split(','): st.markdown(f"<span class='ats-tag'>{tag}</span>",unsafe_allow_html=True)
    st.markdown("---")
    # fetch offers
    variants=[job_title] + generate_title_variants(job_title) + []
    kws=" ".join(variants + build_keywords([job_title]+[missions]+[skills],max_terms=3).split(','))
    all_off=[]
    for loc in sel:
        locn=normalize_location(loc)
        offs=fetch_all_offres(token,kws,locn)
        all_off+=filter_by_location(offs,locn)
    all_off=[o for o in all_off if o.get('typeContrat','') in contract]
    seen={};
    for o in all_off:
        url=o.get('url') or o.get('contact',{}).get('urlPostulation')
        if url and url not in seen: seen[url]=o
    cands=list(seen.values())[:30]
    for o in cands:
        wr=fuzz.WRatio(o.get('intitule',''),job_title)
        pr=fuzz.partial_ratio(o.get('intitule',''),job_title)
        dr=fuzz.partial_ratio(o.get('description_extrait','')[:200],missions)
        score=int(0.5*wr+0.3*pr+0.2*dr)
        with st.expander(f"{o.get('intitule','–')} — {score}%"):
            c1,c2=st.columns([3,1])
            with c1:
                st.markdown(f"- **Contrat** : {o.get('typeContrat','–')}")
                st.markdown(f"- **Lieu** : {o.get('lieuTravail_libelle','')}")
                st.markdown(f"- **Publié** : {o.get('dateCreation','')[:10]}")
                desc=(o.get('description_extrait','') or o.get('description',''))
                st.markdown(f"**Description :** {desc[:150]}…")
                st.markdown(f"<span class='offer-link'><a href='{url}' target='_blank'>Voir l'offre</a></span>",unsafe_allow_html=True)
            with c2:
                st.progress(score/100)
    st.header("5️⃣ SIS – Métiers recommandés")
    top6=scorer_metier(profile,referentiel)
    for _,r in top6.iterrows(): st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
    
