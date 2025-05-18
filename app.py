# -*- coding: utf-8 -*-
"""
CraftMyJob – Streamlit app for smart job suggestions
"""
import os, re, requests
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
  .stButton>button {background-color:#2E86C1;color:white;border-radius:4px;}
  .section-header {font-size:1.2rem;font-weight:600;margin-top:1.5em;color:#2E86C1;}
  h1,h2,h3 {color:#2E86C1;}
  .offer-link a {color:#2E86C1;text-decoration:none;}
  .cv-summary {color:#1F8A70;}
</style>""", unsafe_allow_html=True)
try:
    logo = Image.open("logo_jobseekers.PNG")
    st.image(logo, width=120)
except:
    pass
st.title("✨ CraftMyJob – Votre assistant emploi intelligent")

# ── 2) DATA & MODEL PREP ─────────────────────────────────────────────────
@st.cache_data
def load_referentiel(path="referentiel_metiers_craftmyjob_final.csv"):
    return pd.read_csv(path, dtype=str).fillna("")

@st.cache_data
def build_tfidf(df, max_features=2000):
    corpus = df["Activites"] + " " + df["Competences"] + " " + df["Metier"]
    vect = TfidfVectorizer(max_features=max_features)
    matrix = vect.fit_transform(corpus)
    return vect, matrix

referentiel = load_referentiel()
vecteur, tfidf_matrix = build_tfidf(referentiel)

# ── 3) UTILITIES ─────────────────────────────────────────────────────────
def normalize_location(loc):
    if m:=re.match(r"^(.+?) \((\d{5})\)$", loc): return m.group(2)
    if m2:=re.match(r"Département (\d{2})$", loc): return m2.group(1)
    if m3:=re.match(r"^(.+) \(region:(\d+)\)$", loc): return m3.group(1).lower()
    return loc.lower()

def get_date_range(months=2):
    end = datetime.now().date()
    start = end - timedelta(days=30*months)
    return start.isoformat(), end.isoformat()

def fetch_ftoken(cid, secret):
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "grant_type":"client_credentials",
        "client_id":cid,
        "client_secret":secret,
        "scope":"api_offresdemploiv2 o2dsoffre"
    }
    r = requests.post(url, data=data, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]

def fetch_all_offres(token, mots, lieu, batch_size=100, max_batches=5):
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    dateDebut, dateFin = get_date_range(2)
    toutes = []
    for i in range(max_batches):
        rng = f"{i*batch_size}-{i*batch_size+batch_size-1}"
        params = {
            "motsCles":mots, "localisation":lieu,
            "range":rng, "dateDebut":dateDebut,
            "dateFin":dateFin, "tri":"pertinence"
        }
        r = requests.get(url, headers={"Authorization":f"Bearer {token}"}, params=params, timeout=10)
        if r.status_code not in (200,206) or not r.json().get("resultats"):
            break
        toutes += r.json()["resultats"]
    return toutes

def filter_by_location(offers, loc_norm):
    ln=loc_norm.lower()
    return [o for o in offers if ln in o.get("lieuTravail_libelle","").lower()]

def get_gpt_response(prompt, key):
    url="https://api.openai.com/v1/chat/completions"
    headers={"Authorization":f"Bearer {key}"}
    data={"model":"gpt-3.5-turbo",
          "messages":[{"role":"system","content":"Tu es expert en recrutement."},
                      {"role":"user","content":prompt}],
          "temperature":0.7,"max_tokens":800}
    r=requests.post(url,json=data,headers=headers,timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def generate_title_variants(title):
    v={title.strip()}
    if not title.endswith("s"): v.add(title+"s")
    else: v.add(title.rstrip("s"))
    syn={
      "commercial":["sales","ingénieur commercial","chargé de clientèle"],
      "developer":["développeur"],"manager":["responsable","chef"]
    }
    for w in title.lower().split():
        if w in syn: v.update(syn[w])
    return list(v)

def select_discriminant_skills(text, vectorizer, top_n=5):
    toks=re.findall(r"\w{2,}",text.lower())
    feats=vectorizer.get_feature_names_out(); idfs=vectorizer.idf_
    m=dict(zip(feats,idfs))
    scored=sorted({(t,m.get(t,0)) for t in toks},key=lambda x:x[1],reverse=True)
    return [t for t,_ in scored[:top_n]]

def scorer_metier(inp,df,top_k=6):
    doc=f"{inp['missions']} {inp['skills']} {inp['job_title']}"
    v=vecteur.transform([doc]); cos=cosine_similarity(v,tfidf_matrix).flatten()
    df2=df.copy(); df2["cosine"]=cos
    df2["fz_t"]=df2["Metetier"].apply(lambda m:fuzz.WRatio(m,inp["job_title"])/100)
    df2["fz_m"]=df2["Activites"].apply(lambda a:fuzz.partial_ratio(a,inp["missions"])/100)
    df2["fz_c"]=df2["Competences"].apply(lambda c:fuzz.partial_ratio(c,inp["skills"])/100)
    df2["score"]=(0.5*df2["cosine"]+0.2*df2["fz_t"]+0.15*df2["fz_m"]+0.15*df2["fz_c"])*100
    return df2.nlargest(top_k,"score")

# ── 4) PROFILE FORM ──────────────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")
cv_text=""; up=st.file_uploader("📂 CV (optionnel)",type=["pdf","docx","txt"])
if up:
    ext=up.name.rsplit(".",1)[-1].lower()
    if ext=="pdf": cv_text=" ".join(p.extract_text() or "" for p in PdfReader(up).pages)
    elif ext=="docx": cv_text=" ".join(p.text for p in Document(up).paragraphs)
    else: cv_text=up.read().decode(errors="ignore")

job_title=st.text_input("🔤 Poste souhaité")
missions=st.text_area("📋 Missions principales")
skills=st.text_area("🧠 Compétences clés")

st.markdown("<div class='section-header'>🌍 Territoires</div>",unsafe_allow_html=True)
typed=st.text_input("Tapez commune/département/région…")
opts=search_territoires(typed) if typed else []
sel=st.multiselect("Sélectionnez vos territoires",options=opts,default=[])
exp_level=st.radio("🎯 Expérience",["Débutant (0-2 ans)","Expérimenté (2-5 ans)","Senior (5+ ans)"])
contract=st.multiselect("📄 Types de contrat",["CDI","CDD","Freelance","Stage","Alternance"],default=["CDI","CDD","Freelance"])
remote=st.checkbox("🏠 Full remote")

# ── 5) CLÉS API & IA ───────────────────────────────────────────────────
st.header("2️⃣ Clés API & IA")
key_openai   =st.text_input("🔑 OpenAI API Key",type="password")
key_pe_id    =st.text_input("🔑 Pôle-Emploi Client ID",type="password")
key_pe_secret=st.text_input("🔑 Pôle-Emploi Client Secret",type="password")

# ── 6) ACTION ─────────────────────────────────────────────────────────
if st.button("🚀 Lancer tout"):
    if not key_openai:    st.error("Clé OpenAI requise"); st.stop()
    if not (key_pe_id and key_pe_secret and sel):
        st.error("ID Pôle-Emploi & territoires requis"); st.stop()

    profile={"job_title":job_title,"missions":missions,"skills":skills}
    # 6.1) Résumé CV
    if cv_text:
        summary=get_gpt_response(f"Résumé en 5 points clés du CV:\n{cv_text[:1000]}",key_openai)
        st.markdown("**Résumé CV:**",unsafe_allow_html=True)
        for l in summary.split("\n"): st.markdown(f"- <span class='cv-summary'>{l}</span>",unsafe_allow_html=True)

    # 6.2) Générations IA
    st.header("🧠 Génération IA")
    tpls={'📄 Bio LinkedIn':"Rédige une bio LinkedIn professionnelle.",
          '✉️ Mail de candidature':"Écris un mail de candidature spontanée.",
          '📃 Mini CV':"Génère un mini-CV (5-7 lignes)."}
    choices=st.multiselect("Générations IA",list(tpls.keys()),default=list(tpls.keys())[:2])
    for name in choices:
        inst=tpls[name]
        if name=='📄 Bio LinkedIn': inst+=" Ne mentionne ni localisation ni ancienneté, limite à 4 lignes."
        lines=[f"Poste: {job_title}",f"Missions: {missions}",f"Compétences: {skills}"]
        if cv_text: lines.append(f"Résumé CV: {summary[:300]}")
        lines+= [f"Territoires: {', '.join(sel)}",f"Expérience: {exp_level}",
                 f"Contrat(s): {', '.join(contract)}",f"Télétravail: {'Oui' if remote else 'Non'}","",inst]
        try:
            res=get_gpt_response("\n".join(lines),key_openai)
            st.subheader(name); st.markdown(res)
        except requests.HTTPError as e:
            st.error(f"OpenAI ({e.response.status_code}):{e.response.text}"); st.stop()

    # 6.3) Token Pôle-Emploi
    cid,secret=key_pe_id.strip(),key_pe_secret.strip()
    st.write(f"🔍 Debug Client ID: '{cid}' (len={len(cid)})")
    try:
        token=fetch_ftoken(cid,secret); st.success("✅ Token OK")
    except requests.HTTPError as e:
        st.error(f"Pôle-Emploi ({e.response.status_code}):{e.response.text}"); st.stop()

    # 6.4) Top 30 Offres améliorées
    st.header(f"4️⃣ Top 30 offres pour '{job_title}'")
    vars=generate_title_variants(job_title)
    dskills=select_discriminant_skills(skills,vecteur,top_n=5)
    kws=" ".join([job_title]+vars+dskills)

    all_off=[]
    for loc in sel:
        locn=normalize_location(loc)
        offs=fetch_all_offres(token,kws,locn,batch_size=100,max_batches=5)
        all_off+=filter_by_location(offs,locn)
    all_off=[o for o in all_off if o.get("typeContrat","") in contract]

    seen={}
    for o in all_off:
        url=o.get("url","") or o.get("contact",{}).get("urlPostulation","") 
        if url and url not in seen: seen[url]=o
    cands=list(seen.values())
    for o in cands:
        wr=fuzz.WRatio(o.get("intitule",""),job_title)
        pr=fuzz.partial_ratio(o.get("intitule",""),job_title)
        dr=fuzz.partial_ratio(o.get("description_extrait","")[:200],missions)
        o["score_match"]=0.5*wr+0.3*pr+0.2*dr

    top30=sorted(cands,key=lambda x:x["score_match"],reverse=True)[:30]
    if top30:
        for o in top30:
            st.markdown(
                f"**{o['intitule']}** ({o['typeContrat']}) – {o['lieuTravail_libelle']} (_Publié {o['dateCreation'][:10]}_)  \n"
                f"Match: **{int(o['score_match'])}%**  \n"
                f"<span class='offer-link'><a href='{o['url']}' target='_blank'>Voir</a></span>\n---",
                unsafe_allow_html=True
            )
    else:
        st.info("Aucune offre pertinente trouvée.")

    # 6.5) SIS Métiers complet
    st.header("5️⃣ SIS – Métiers recommandés")
    top6=scorer_metier(profile,referentiel,top_k=6)
    for _,r in top6.iterrows():
        m,rsc=r["Metier"],int(r["score"])
        st.markdown(f"**{m}** – {rsc}%")
        subs=[]
        for loc in sel:
            subs+=requests.get(
                "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search",
                headers={"Authorization":f"Bearer {token}"},
                params={
                    "motsCles":m,"localisation":normalize_location(loc),
                    "range":"0-2","dateDebut":get_date_range()[0],
                    "dateFin":get_date_range()[1],"tri":"pertinence"
                }
            ).json().get("resultats",[])
        subs=[o for o in subs if o.get("typeContrat","") in contract]
        seen2=set()
        if subs:
            for o in subs:
                url2=o.get("url","") or o.get("contact",{}).get("urlPostulation","")
                if url2 not in seen2:
                    seen2.add(url2)
                    st.markdown(
                        f"• **{o['intitule']}** ({o['typeContrat']}) – {o['lieuTravail_libelle']} (_Publié {o['dateCreation'][:10]}_)  \n"
                        f"{(o['description_extrait'] or '')[:150]}…  \n"
                        f"<span class='offer-link'><a href='{url2}' target='_blank'>Voir / Postuler</a></span>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("Aucune offre pour ce métier.")
