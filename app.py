# -*- coding: utf-8 -*-
"""
CraftMyJob – Streamlit app with CV résumé & refined IA prompts
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

# ── 0) CLEAN ENVIRONMENT ────────────────────────────────────────
for v in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(v, None)

# ── 1) STREAMLIT CONFIG & STYLE ────────────────────────────────
st.set_page_config(page_title="CraftMyJob – Job Seekers Hub France", layout="centered")
st.markdown("""
<style>
  .stButton>button { background-color:#2E86C1; color:white; border-radius:4px; }
  .section-header { font-size:1.2rem; font-weight:600; margin-top:1.5em; color:#2E86C1; }
  .success { color:green; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# Logo
try:
    logo = Image.open("logo_jobseekers.PNG")
    st.image(logo, width=120)
except:
    pass

st.title("CraftMyJob – Votre assistant emploi intelligent")

# ── 2) DATA & TF-IDF PREP ───────────────────────────────────────
@st.cache_data
def load_referentiel(path="referentiel_metiers_craftmyjob_final.csv"):
    return pd.read_csv(path, dtype=str).fillna("")

@st.cache_data
def build_tfidf(df, max_features=2000):
    corpus = df["Activites"] + " " + df["Competences"] + " " + df["Metier"]
    vect = TfidfVectorizer(max_features=max_features)
    mat  = vect.fit_transform(corpus)
    return vect, mat

referentiel      = load_referentiel()
vecteur, tfidf_matrix = build_tfidf(referentiel)

# ── 3) UTILITIES ────────────────────────────────────────────────
def normalize_location(loc: str) -> str:
    if m := re.match(r"^(.+?) \(\d{5}\)", loc):
        return m.group(1)
    if m := re.match(r"Département (\d{2})", loc):
        return m.group(1)
    if m := re.match(r"^(.+) \(region:(\d+)\)", loc):
        return m.group(1)
    return loc

def get_date_range(months:int=2):
    end   = datetime.now().date()
    start = end - timedelta(days=months*30)
    return start.isoformat(), end.isoformat()

def build_keywords(texts:list[str], max_terms:int=7)->str:
    tokens = re.findall(r"\w{2,}", " ".join(texts).lower())
    stop   = {"et","ou","la","le","les","de","des","du","un","une",
              "à","en","pour","par","avec","sans","sur","dans","au","aux"}
    kws, seen = [], set()
    for t in tokens:
        if t not in stop and t not in seen:
            seen.add(t); kws.append(t)
            if len(kws)==max_terms: break
    return ",".join(kws)

def get_gpt_response(prompt:str, key:str)->str:
    url = "https://api.openai.com/v1/chat/completions"
    hdr = {"Authorization":f"Bearer {key}", "Content-Type":"application/json"}
    data= {
        "model":"gpt-3.5-turbo",
        "messages":[
            {"role":"system","content":"Tu es un expert en recrutement et personal branding."},
            {"role":"user","content":prompt}
        ],
        "temperature":0.7,"max_tokens":800
    }
    r = requests.post(url, json=data, headers=hdr, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

class PDFGen:
    @staticmethod
    def to_pdf(text:str)->io.BytesIO:
        buf = io.BytesIO()
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0,8,line)
        pdf.output(buf); buf.seek(0)
        return buf

def fetch_ftoken(cid:str, secret:str)->str:
    url = ("https://entreprise.pole-emploi.fr"
           "/connexion/oauth2/access_token?realm=/partenaire")
    data = {
        "grant_type":"client_credentials",
        "client_id":cid,
        "client_secret":secret,
        "scope":"api_offresdemploiv2 o2dsoffre"
    }
    r = requests.post(url, data=data, timeout=10)
    r.raise_for_status()
    return r.json().get("access_token","")

def search_offres(token:str, mots:str, lieu:str, limit:int=5)->list:
    url = ("https://api.francetravail.io"
           "/partenaire/offresdemploi/v2/offres/search")
    dateDebut, dateFin = get_date_range(2)
    params = {
      "motsCles":mots,
      "localisation":lieu,
      "range":f"0-{limit-1}",
      "dateDebut":dateDebut,
      "dateFin":dateFin,
      "tri":"dateCreation"
    }
    r = requests.get(url, headers={"Authorization":f"Bearer {token}"}, params=params, timeout=10)
    if r.status_code==204: return []
    if r.status_code not in (200,206):
        st.error(f"FT API {r.status_code}: {r.text}")
        return []
    return r.json().get("resultats", [])

def filter_by_location(offers:list, loc_norm:str)->list:
    out=[]; norm=loc_norm.lower()
    for o in offers:
        lib = o.get("lieuTravail",{}).get("libelle","").lower()
        cp  = str(o.get("lieuTravail",{}).get("codePostal",""))
        if norm in lib or norm==cp: out.append(o)
    return out

def scorer_metier(inp:dict, df:pd.DataFrame, top_k:int=6)->pd.DataFrame:
    doc    = f"{inp['missions']} {inp['skills']} {inp['job_title']}"
    v_user = vecteur.transform([doc])
    cos    = cosine_similarity(v_user, tfidf_matrix).flatten()
    df2    = df.copy(); df2["cosine"]=cos
    df2["fz_t"]=df2["Metier"].map(lambda m: fuzz.token_set_ratio(m, inp["job_title"])/100)
    df2["fz_m"]=df2["Activites"].map(lambda a: fuzz.token_set_ratio(a, inp["missions"])/100)
    df2["fz_c"]=df2["Competences"].map(lambda c: fuzz.token_set_ratio(c, inp["skills"])/100)
    df2["score"]=(0.5*df2["cosine"]+0.2*df2["fz_t"]+0.15*df2["fz_m"]+0.15*df2["fz_c"])*100
    return df2.nlargest(top_k, "score")

# ── 4) FORM PROFILE ─────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")

# CV uploader & extraction
cv_text=""
up = st.file_uploader("📂 CV (optionnel)", type=["pdf","docx","txt"])
if up:
    ext = up.name.rsplit(".",1)[-1].lower()
    if ext=="pdf":
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(up).pages)
    elif ext=="docx":
        cv_text = " ".join(p.text for p in Document(up).paragraphs)
    else:
        cv_text = up.read().decode(errors="ignore")

job_title = st.text_input("🔤 Poste souhaité", placeholder="Ex : Responsable CRM")
missions  = st.text_area("📋 Missions principales", placeholder="Décrivez vos futures missions…")
skills    = st.text_area("🧠 Compétences clés", placeholder="3-5 compétences majeures…")

st.markdown("<div class='section-header'>🌍 Territoires</div>", unsafe_allow_html=True)
typed   = st.text_input("Tapez commune/département/région…", placeholder="Ex : Hauts-de-Seine ou 92")
opts    = search_territoires(typed) if typed else []
default = st.session_state.get("locations", [])
sel     = st.multiselect("Sélectionnez vos territoires", options=(default+opts), default=default)
st.session_state.locations = sel

exp_level = st.radio("🎯 Expérience",
    ["Débutant (0-2 ans)","Expérimenté (2-5 ans)","Senior (5+ ans)"])
contract  = st.multiselect("📄 Types de contrat",
    ["CDI","CDD","Freelance","Stage","Alternance"],
    default=["CDI","CDD","Freelance"])
remote    = st.checkbox("🏠 Full remote")

# ── 5) CLÉS API & IA ────────────────────────────────────────────
st.header("2️⃣ Clés API & IA")
key_openai    = st.text_input("🔑 OpenAI API Key",    type="password")
key_pe_id     = st.text_input("🔑 Pôle-Emploi Client ID",     type="password")
key_pe_secret = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

tpls = {
  "📄 Bio LinkedIn":    "Rédige une bio LinkedIn professionnelle.",
  "✉️ Mail candidature": "Écris un mail de candidature spontanée.",
  "📃 Mini CV":          "Génère un mini-CV (5-7 lignes).",
  "🧩 CV optimisé IA":   "Optimise le CV en soulignant deux mots-clés."
}
choices = st.multiselect("Générations IA", list(tpls.keys()), default=list(tpls.keys())[:2])

# ── 6) ACTION ────────────────────────────────────────────────────
if st.button("🚀 Lancer tout"):
    # validations
    if not key_openai:
        st.error("🔑 Clé OpenAI requise"); st.stop()
    if not (key_pe_id and key_pe_secret and sel):
        st.error("🔑 Clé Pôle-Emploi + territoires requis"); st.stop()

    # ① Résumé CV en 5 points
    summary_cv = ""
    if cv_text:
        try:
            prompt_cv = "Peux-tu me faire un résumé en 5 points clairs de ce CV ?\n\n" + cv_text
            summary_cv = get_gpt_response(prompt_cv, key_openai)
            st.markdown("<div class='success'>✅ Résumé CV généré !</div>", unsafe_allow_html=True)
            st.markdown(summary_cv)
        except Exception as e:
            st.warning("⚠️ Impossible de générer le résumé CV : " + str(e))

    # ② Token Pôle-Emploi
    try:
        token = fetch_ftoken(key_pe_id, key_pe_secret)
        st.markdown("<div class='success'>✅ Token Pôle-Emploi OK !</div>", unsafe_allow_html=True)
    except requests.HTTPError as e:
        code = e.response.status_code; text = e.response.text
        if code==401:
            st.error("🔑 Identifiants Pôle-Emploi invalides.")
        else:
            st.error(f"⚠️ Erreur Pôle-Emploi ({code}) : {text}")
        st.stop()

    # ③ Générations IA
    st.header("🧠 Génération IA")
    for name in choices:
        instr = tpls[name]
        if name=="📄 Bio LinkedIn":
            instr += " Ne mentionne aucune localisation et limite la bio à 4 lignes."
        prompt = "\n".join(filter(None,[
            f"Poste: {job_title}",
            f"Missions: {missions}",
            f"Compétences: {skills}",
            f"Résumé CV:\n{summary_cv}" if summary_cv else "",
            f"Territoires: {', '.join(sel)}",
            f"Expérience: {exp_level}",
            f"Contrat(s): {', '.join(contract)}",
            f"Télétravail: {'Oui' if remote else 'Non'}",
            "",
            instr
        ]))
        try:
            out = get_gpt_response(prompt, key_openai)
            st.subheader(name); st.markdown(out)
            if name=="🧩 CV optimisé IA":
                pdf = PDFGen.to_pdf(out)
                st.download_button("📥 Télécharger CV optimisé", data=pdf,
                                   file_name="CV_optimisé.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Erreur IA ({name}) : {e}")

    # ④ Top Offres pour le poste
    st.header(f"4️⃣ Top offres pour « {job_title} »")
    all_offres = []
    for loc in sel:
        loc_norm = normalize_location(loc)
        offs = search_offres(token, job_title, loc_norm, limit=5)
        offs = filter_by_location(offs, loc_norm)
        offs = [o for o in offs if o.get("typeContrat","") in contract]
        all_offres.extend(offs)

    seen = {}
    for o in all_offres:
        url = o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","")
        if url and url not in seen:
            seen[url] = o

    if seen:
        for url,o in list(seen.items())[:5]:
            t = o.get("intitule","–"); lib = o["lieuTravail"]["libelle"]
            cp = o["lieuTravail"]["codePostal"]; tp = o.get("typeContrat","–")
            st.markdown(f"**{t}** ({tp}) – {lib} [{cp}]  \n[Voir l'offre]({url})\n---")
    else:
        st.info("🔍 Aucune offre trouvée.")

    # ⑤ SIS Métiers recommandés
    st.header("5️⃣ SIS – Métiers recommandés")
    profile = {"job_title":job_title,"missions":missions,"skills":skills}
    top6 = scorer_metier(profile, referentiel, top_k=6)
    for _,r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        kws = r["Metet" if False else "Metier"]  # => 'Metier'
        subs=[]
        for loc in sel:
            loc_norm = normalize_location(loc)
            tmp = search_offres(token, kws, loc_norm, limit=3)
            tmp = filter_by_location(tmp, loc_norm)
            tmp = [o for o in tmp if o.get("typeContrat","") in contract]
            subs.extend(tmp)
        if subs:
            seen2=set()
            for o in subs[:3]:
                url2 = o.get("contact",{}).get("urlPostulation") or o.get("contact",{}).get("urlOrigine","")
                if url2 not in seen2:
                    seen2.add(url2)
                    dt   = o.get("dateCreation","")[:10]
                    lib2 = o["lieuTravail"]["libelle"]; tp2=o.get("typeContrat","–")
                    desc = (o.get("description","") or "").replace("\n"," ")[:150]+"…"
                    st.markdown(f"• **{o['intitule']}** ({tp2}) – {lib2} (_Publié {dt}_)  \n{desc}  \n[Voir / Postuler]({url2})")
        else:
            st.info("Aucune offre SIS trouvée pour ces métiers.")

