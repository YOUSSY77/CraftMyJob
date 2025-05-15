# -*- coding: utf-8 -*-
"""
CraftMyJob – Streamlit app for smart job suggestions with extended date range
"""
import os, io, re, requests
import pandas as pd
import streamlit as st
from PIL import Image
from fpdf import FPDF
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from datetime import datetime, timedelta

# ── 0) CLEAN ENVIRONMENT ─────────────────────────────────────────────────────
for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(v, None)

# ── 1) STREAMLIT CONFIG & STYLING ─────────────────────────────────────────────
st.set_page_config(page_title="CraftMyJob – Job Seekers Hub France", layout="centered")
st.markdown("""
<style>
  .stButton>button { background-color:#2E86C1; color:white; border-radius:4px; }
  .section-header { font-size:1.2rem; font-weight:600; margin-top:1.5em; color:#2E86C1; }
  h1,h2,h3 { color:#2E86C1; }
  .offer-link a { color:#2E86C1; text-decoration:none; }
  .cv-summary { color: green; margin-bottom:1em; }
</style>
""", unsafe_allow_html=True)

# Logo
try:
    logo = Image.open("logo_jobseekers.PNG")
    st.image(logo, width=120)
except:
    pass

st.title(" CraftMyJob – Votre assistant emploi intelligent")

# ── 2) DATA & MODEL PREP ───────────────────────────────────────────────────────
@st.cache_data
def load_referentiel(path="referentiel_metiers_craftmyjob_final.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")

@st.cache_data
def build_tfidf(df: pd.DataFrame, max_features:int=2000):
    corpus = df["Activites"] + " " + df["Competences"] + " " + df["Metier"]
    vect   = TfidfVectorizer(max_features=max_features)
    mat    = vect.fit_transform(corpus)
    return vect, mat

referentiel     = load_referentiel()
vecteur, tfidf_matrix = build_tfidf(referentiel)

# ── 3) UTILITIES ──────────────────────────────────────────────────────────────
def normalize_location(loc:str) -> str:
    m = re.match(r"^(.+?) \(\d{5}\)", loc)
    if m: return m.group(1)
    m = re.match(r"Département (\d{2})", loc)
    if m: return m.group(1)
    m = re.match(r"^(.+) \(region:(\d+)\)", loc)
    if m: return m.group(1)
    return loc

def get_date_range(months:int=2):
    end   = datetime.now().date()
    start = end - timedelta(days=months*30)
    return start.isoformat(), end.isoformat()

def search_territoires(query:str, limit:int=10) -> list[str]:
    res=[]
    if re.fullmatch(r"\d{2}", query):
        r = requests.get(f"https://geo.api.gouv.fr/departements/{query}/communes",
                         params={"fields":"nom,codesPostaux","limit":limit}, timeout=5)
        r.raise_for_status()
        for e in r.json():
            cp = e.get("codesPostaux",["00000"])[0]
            res.append(f"{e['nom']} ({cp})")
        res.append(f"Département {query}")
        return list(dict.fromkeys(res))
    r1 = requests.get("https://geo.api.gouv.fr/communes",
                      params={"nom":query,"fields":"nom,codesPostaux","limit":limit}, timeout=5)
    if r1.status_code==200:
        for e in r1.json():
            cp=res.append(f"{e['nom']} ({cp})") if False else None
            cp = e.get("codesPostaux",["00000"])[0]; res.append(f"{e['nom']} ({cp})")
    r2 = requests.get("https://geo.api.gouv.fr/regions",
                      params={"nom":query,"fields":"nom,code"}, timeout=5)
    if r2.status_code==200:
        for rg in r2.json():
            res.append(f"{rg['nom']} (region:{rg['code']})")
    return list(dict.fromkeys(res))

def build_keywords(texts:list[str], max_terms:int=7) -> str:
    combined = " ".join(texts).lower()
    tokens   = re.findall(r"\w{2,}", combined)
    stop     = {"et","ou","la","le","les","de","des","du","un","une",
                "à","en","pour","par","avec","sans","sur","dans","au","aux"}
    seen,kws=set(),[]
    for t in tokens:
        if t in stop or t in seen: continue
        seen.add(t); kws.append(t)
        if len(kws)>=max_terms: break
    return ",".join(kws)

def get_gpt_response(prompt:str, key:str) -> str:
    url    = "https://api.openai.com/v1/chat/completions"
    hdr    = {"Authorization":f"Bearer {key}","Content-Type":"application/json"}
    payload= {
        "model":"gpt-3.5-turbo",
        "messages":[
            {"role":"system","content":"Tu es un expert en recrutement et personal branding."},
            {"role":"user","content":prompt}
        ],
        "temperature":0.7,"max_tokens":800
    }
    r = requests.post(url, json=payload, headers=hdr, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

class PDFGen:
    @staticmethod
    def to_pdf(text:str) -> io.BytesIO:
        buf=io.BytesIO(); pdf=FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
        for line in text.split("\n"): pdf.multi_cell(0,8,line)
        pdf.output(buf); buf.seek(0); return buf

def fetch_ftoken(cid:str, secret:str) -> str:
    url  = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {"grant_type":"client_credentials","client_id":cid,
            "client_secret":secret,"scope":"api_offresdemploiv2 o2dsoffre"}
    r = requests.post(url,data=data,timeout=10)
    r.raise_for_status()
    return r.json().get("access_token","")

def search_offres(token:str, mots:str, lieu:str, limit:int=5) -> list:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    dateDebut, dateFin = get_date_range(2)
    params = {
        "motsCles":mots, "localisation":lieu,
        "range":f"0-{limit-1}",
        "dateDebut":dateDebut,"dateFin":dateFin,
        "tri":"dateCreation"
    }
    r = requests.get(url, headers={"Authorization":f"Bearer {token}"}, params=params, timeout=10)
    if r.status_code==204: return []
    if r.status_code not in (200,206):
        st.error(f"FT API {r.status_code}: {r.text}")
        return []
    return r.json().get("resultats",[])

def filter_by_location(offers:list, loc_norm:str)->list:
    out=[]; needle=loc_norm.lower()
    for o in offers:
        lib = o.get("lieuTravail",{}).get("libelle","").lower()
        cp  = str(o.get("lieuTravail",{}).get("codePostal",""))
        if needle in lib or needle==cp:
            out.append(o)
    return out

def scorer_metier(inp:dict, df:pd.DataFrame, top_k:int=6)->pd.DataFrame:
    doc    = f"{inp['missions']} {inp['skills']} {inp['job_title']}"
    v_user = vecteur.transform([doc])
    cos    = cosine_similarity(v_user, tfidf_matrix).flatten()
    df2    = df.copy(); df2["cosine"]=cos
    df2["fz_t"] = df2["Metier"].apply(lambda m: fuzz.token_set_ratio(m, inp["job_title"])/100)
    df2["fz_m"] = df2["Activites"].apply(lambda a: fuzz.token_set_ratio(a, inp["missions"])/100)
    df2["fz_c"] = df2["Competences"].apply(lambda c: fuzz.token_set_ratio(c, inp["skills"])/100)
    df2["score"]= (0.5*df2["cosine"]+0.2*df2["fz_t"]+0.15*df2["fz_m"]+0.15*df2["fz_c"])*100
    return df2.nlargest(top_k,"score")

# ── 4) FORM PROFILE ─────────────────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")

cv_text = ""
up = st.file_uploader("📂 CV (optionnel)", type=["pdf","docx","txt"])

if up is not None:
    # on ne rentre ici que si un fichier a été uploadé
    try:
        ext = up.name.rsplit(".", 1)[-1].lower()
    except Exception:
        ext = ""
    if ext == "pdf":
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(up).pages)
    elif ext == "docx":
        cv_text = " ".join(p.text for p in Document(up).paragraphs)
    else:
        # txt ou autre
        cv_text = up.read().decode(errors="ignore")
# si up est None, cv_text reste ""


job_title=st.text_input("🔤 Poste souhaité")
missions = st.text_area("📋 Missions principales")
skills   = st.text_area("🧠 Compétences clés")

st.markdown("<div class='section-header'>🌍 Territoires</div>", unsafe_allow_html=True)
typed   = st.text_input("Tapez commune/département/région…")
opts    = search_territoires(typed) if typed else []
default = st.session_state.get("locations",[])
sel     = st.multiselect("Sélectionnez vos territoires", options=(default+opts), default=default)
st.session_state.locations = sel

exp_level = st.radio("🎯 Expérience", ["Débutant (0-2 ans)","Expérimenté (2-5 ans)","Senior (5+ ans)"])
contract  = st.multiselect("📄 Types de contrat",
                           options=["CDI","CDD","Freelance","Stage","Alternance"],
                           default=["CDI","CDD","Freelance"])
remote    = st.checkbox("🏠 Full remote")

# ── 5) CLÉS API & IA ─────────────────────────────────────────────────────────
st.header("2️⃣ Clés API & IA")
key_openai    = st.text_input("🔑 OpenAI API Key",    type="password")
key_pe_id     = st.text_input("🔑 Pôle-Emploi ID",     type="password")
key_pe_secret = st.text_input("🔑 Pôle-Emploi Secret", type="password")

tpls = {
  "📄 Bio LinkedIn":    "Rédige une bio LinkedIn professionnelle.",
  "✉️ Mail candidature": "Écris un mail de candidature spontanée.",
  "📃 Mini CV":         "Génère un mini-CV (5-7 lignes).",
  "🧩 CV optimisé IA":  "Optimise le CV en soulignant deux mots-clés."
}
choices = st.multiselect("Générations IA", list(tpls.keys()), default=list(tpls.keys())[:2])

# ── 6) ACTION ────────────────────────────────────────────────────────────────
if st.button("🚀 Lancer tout"):
    # validations
    if not key_openai:
        st.error("🔑 Clé OpenAI requise"); st.stop()
    if not (key_pe_id and key_pe_secret and sel):
        st.error("🔑 Pôle-Emploi + territoires requis"); st.stop()

    # ① Résumé CV en 5 points
    if cv_text:
        prompt_cv = f"Résume ce CV en 5 points clés pour un poste de « {job_title} »."
        summary   = get_gpt_response(prompt_cv, key_openai)
        st.markdown(f"<div class='cv-summary'>{summary}</div>", unsafe_allow_html=True)
    else:
        summary = ""

    # ② Générations IA
    st.header("🧠 Génération IA")
    for name in choices:
        instr = tpls[name]
        if name=="📄 Bio LinkedIn":
            instr += " Ne mentionne aucune localisation et limite la bio à 4 lignes."
        prompt = "\n".join([
            f"Résumé CV: {summary}",
            f"Poste: {job_title}",
            f"Missions: {missions}",
            f"Compétences: {skills}",
            f"Territoires: {', '.join(sel)}",
            f"Expérience: {exp_level}",
            f"Contrat(s): {', '.join(contract)}",
            f"Télétravail: {'Oui' if remote else 'Non'}",
            "",
            instr
        ])
        try:
            res = get_gpt_response(prompt, key_openai)
            st.subheader(name); st.markdown(res)
            if name=="🧩 CV optimisé IA":
                buf=PDFGen.to_pdf(res)
                st.download_button("📥 Télécharger CV optimisé",
                                   data=buf,file_name="CV_optimisé.pdf",mime="application/pdf")
        except requests.HTTPError as e:
            if e.response.status_code==401:
                st.error("Clé OpenAI invalide ou expirée.")
            else:
                st.error(f"Erreur OpenAI ({e.response.status_code})")
            st.stop()

    # ③ Token Pôle-Emploi
    try:
        token = fetch_ftoken(key_pe_id, key_pe_secret)
    except requests.HTTPError as e:
        if e.response.status_code==401:
            st.error("🔑 Identifiants Pôle-Emploi invalides ou expirés.")
        else:
            st.error(f"⚠️ Erreur Pôle-Emploi ({e.response.status_code})")
        st.stop()

    # ④ Top Offres pour le poste (fuzzy + géo + contrat)
    st.header(f"4️⃣ Top offres pour « {job_title} »")
    raw = []
    for loc in sel:
        loc_norm = normalize_location(loc)
        raw += search_offres(token, job_title, loc_norm, limit=20)
    # géo
    raw = [o for o in raw 
           if normalize_location(o["lieuTravail"]["libelle"]).lower().find(loc_norm.lower())>=0
           or str(o["lieuTravail"]["codePostal"])==loc_norm]
    # fuzzy score & contrat
    for o in raw:
        o["_fuzz"] = fuzz.token_set_ratio(job_title, o.get("intitule",""))
    filtered = [o for o in raw if o["_fuzz"]>=60 and o.get("typeContrat","") in contract]
    # dédup + tri
    uniq = {}
    for o in sorted(filtered, key=lambda x: x["_fuzz"], reverse=True):
        url = o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","")
        if url and url not in uniq:
            uniq[url] = o
    if uniq:
        for o in list(uniq.values())[:5]:
            url   = o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","")
            title = o.get("intitule","–")
            lib   = o["lieuTravail"]["libelle"]
            typ   = o.get("typeContrat","–")
            score = o["_fuzz"]
            st.markdown(
                f"**{title}** ({typ}) – {lib} – *{score}% similarité*  \n"
                f"<span class='offer-link'><a href='{url}' target='_blank'>Voir l'offre</a></span>\n---",
                unsafe_allow_html=True
            )
    else:
        st.info("Aucune offre suffisamment proche trouvée.")

    # ⑤ SIS – Métiers recommandés
    st.header("5️⃣ SIS – Métiers recommandés")
    top6 = scorer_metier(profile, referentiel, top_k=6)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        subs=[]
        for loc in sel:
            loc_norm = normalize_location(loc)
            tmp = search_offres(token, r["Metet"] if False else r["Metier"], loc_norm, limit=20)
            # géo & contrat
            tmp = filter_by_location(tmp, loc_norm)
            tmp = [o for o in tmp if o.get("typeContrat","") in contract]
            subs.extend(tmp)
        # dédup + date
        uniq2={}
        for o in sorted(subs, key=lambda x: x.get("dateCreation",""), reverse=True):
            url2 = o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","")
            if url2 and url2 not in uniq2:
                uniq2[url2]=o
        if uniq2:
            for o in list(uniq2.values())[:3]:
                url2  = o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","")
                title = o.get("intitule","–")
                lib   = o["lieuTravail"]["libelle"]
                typ   = o.get("typeContrat","–")
                dt    = o.get("dateCreation","")[:10]
                desc  = (o.get("description","") or "").replace("\n"," ")[:150]+"…"
                st.markdown(
                    f"• **{title}** ({typ}) – {lib} (_Publié {dt}_)  \n"
                    f"{desc}  \n"
                    f"<span class='offer-link'><a href='{url2}' target='_blank'>Voir / Postuler</a></span>",
                    unsafe_allow_html=True
                )
        else:
            st.info(f"Aucune offre SIS pour « {r['Metet'] if False else r['Metier']} ».")
