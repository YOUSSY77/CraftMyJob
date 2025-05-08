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
    if m := re.match(r"^(.+?) \(\d{5}\)", loc):
        return m.group(1)
    if m2 := re.match(r"^Départment (\d{2})", loc):
        return m2.group(1)
    if m3 := re.match(r"^(.+) \(region:(\d+)\)", loc):
        return m3.group(1)
    return loc

def search_territoires(query: str, limit: int = 10) -> list[str]:
    res = []
    # 1) code INSEE département (2 chiffres)
    if re.fullmatch(r"\d{2}", query):
        r = requests.get(
            f"https://geo.api.gouv.fr/departements/{query}/communes",
            params={"fields":"nom,codesPostaux","limit":limit}, timeout=5
        )
        r.raise_for_status()
        for c in r.json():
            cp = c["codesPostaux"][0] if c["codesPostaux"] else "00000"
            res.append(f"{c['nom']} ({cp})")
        res.append(f"Départment {query}")
        return list(dict.fromkeys(res))

    # 2) nom de département
    r_dep = requests.get(
        "https://geo.api.gouv.fr/departements",
        params={"nom": query, "fields": "code,nom"}, timeout=5
    )
    if r_dep.status_code == 200 and (deps := r_dep.json()):
        code = deps[0]["code"]
        # communes du département
        r_com = requests.get(
            f"https://geo.api.gouv.fr/departements/{code}/communes",
            params={"fields":"nom,codesPostaux","limit":limit}, timeout=5
        )
        r_com.raise_for_status()
        for c in r_com.json():
            cp = c["codesPostaux"][0] if c["codesPostaux"] else "00000"
            res.append(f"{c['nom']} ({cp})")
        res.append(f"Départment {code}")
        return list(dict.fromkeys(res))

    # 3) communes
    r1 = requests.get(
        "https://geo.api.gouv.fr/communes",
        params={"nom":query,"fields":"nom,codesPostaux","limit":limit}, timeout=5
    )
    if r1.status_code == 200:
        for c in r1.json():
            cp = c["codesPostaux"][0] if c["codesPostaux"] else "00000"
            res.append(f"{c['nom']} ({cp})")
    # régions
    r2 = requests.get(
        "https://geo.api.gouv.fr/regions",
        params={"nom":query,"fields":"nom,code"}, timeout=5
    )
    if r2.status_code == 200:
        for rg in r2.json():
            res.append(f"{rg['nom']} (region:{rg['code']})")

    return list(dict.fromkeys(res))

def build_keywords(texts: list[str], max_terms: int = 7) -> str:
    combined = " ".join(texts).lower()
    tokens = re.findall(r"\w{2,}", combined)
    stop = {"et","ou","la","le","les","de","des","du","un","une",
            "à","en","pour","par","avec","sans","sur","dans","au","aux"}
    seen, kws = set(), []
    for t in tokens:
        if t in stop or t in seen:
            continue
        seen.add(t); kws.append(t)
        if len(kws) >= max_terms:
            break
    return ",".join(kws)

def filter_by_location(offers: list, loc_norm: str) -> list:
    filtered = []
    for o in offers:
        lib = o.get('lieuTravail', {}).get('libelle', '')
        if loc_norm.lower() in lib.lower():
            filtered.append(o)
    return filtered

def get_gpt_response(prompt: str, key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}
    data = {
        "model":"gpt-3.5-turbo",
        "messages":[
            {"role":"system","content":"Tu es un expert en recrutement et personal branding."},
            {"role":"user","content":prompt}
        ],
        "temperature":0.7,
        "max_tokens":800
    }
    r = requests.post(url, json=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

class PDFGen:
    @staticmethod
    def to_pdf(text: str) -> io.BytesIO:
        buf = io.BytesIO()
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 8, line)
        pdf.output(buf); buf.seek(0)
        return buf

def fetch_ftoken(cid: str, secret: str) -> str:
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "grant_type":"client_credentials",
        "client_id":cid,
        "client_secret":secret,
        "scope":"api_offresdemploiv2 o2dsoffre"
    }
    r = requests.post(url, data=data, timeout=10)
    r.raise_for_status()
    return r.json().get("access_token","")

def search_offres(token: str, mots: str, lieu: str, limit: int=5) -> list:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    r = requests.get(
        url,
        headers={"Authorization":f"Bearer {token}"},
        params={"motsCles":mots,"localisation":lieu,"range":f"0-{limit-1}"},
        timeout=10
    )
    if r.status_code == 204:
        return []
    if r.status_code not in (200, 206):
        st.error(f"FT API {r.status_code}: {r.text}")
        return []
    return r.json().get("resultats", [])

def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int=6) -> pd.DataFrame:
    doc = " ".join([inp["missions"], inp["skills"], inp["job_title"]])
    v_user = vecteur.transform([doc])
    cos = cosine_similarity(v_user, tfidf_matrix).flatten()
    df2 = df.copy()
    df2["cosine"] = cos
    df2["fz_t"] = df2["Metier"].apply(lambda m: fuzz.token_set_ratio(m, inp["job_title"])/100)
    df2["fz_m"] = df2["Activites"].apply(lambda a: fuzz.token_set_ratio(a, inp["missions"])/100)
    df2["fz_c"] = df2["Competences"].apply(lambda c: fuzz.token_set_ratio(c, inp["skills"])/100)
    df2["score"] = (0.5*df2["cosine"] + 0.2*df2["fz_t"] + 0.15*df2["fz_m"] + 0.15*df2["fz_c"]) * 100
    return df2.nlargest(top_k, "score")

# ── 4) PROFILE FORM ──────────────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")
cv_text = ""
up = st.file_uploader("📂 CV (optionnel)", type=["pdf","docx","txt"])
if up:
    ext = up.name.rsplit('.',1)[-1].lower()
    if ext=='pdf':
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(up).pages)
    elif ext=='docx':
        cv_text = " ".join(p.text for p in Document(up).paragraphs)
    else:
        cv_text = up.read().decode(errors='ignore')

job_title     = st.text_input("🔤 Poste souhaité")
missions      = st.text_area("📋 Missions principales")
skills        = st.text_area("🧠 Compétences clés")
desired_skills = st.text_area("✨ Compétences ciblées (optionnel)")

st.markdown("<div class='section-header'>🌍 Territoires</div>", unsafe_allow_html=True)
typed  = st.text_input("Tapez commune/département/région…")
opts   = search_territoires(typed) if typed else []
default = st.session_state.get('locations', [])
sel    = st.multiselect("Sélectionnez vos territoires", options=(default+opts), default=default)
st.session_state.locations = sel

exp_level = st.radio("🎯 Expérience", ["Débutant (0-2 ans)","Expérimenté (2-5 ans)","Senior (5+ ans)"])
contract  = st.selectbox("📄 Contrat", ["CDI","CDD","Freelance","Stage"])
remote    = st.checkbox("🏠 Full remote")

st.header("2️⃣ Clés API & IA")
key_openai    = st.text_input("🔑 OpenAI API Key", type="password")
key_pe_id     = st.text_input("🔑 Pôle-Emploi ID", type="password")
key_pe_secret = st.text_input("🔑 Pôle-Emploi Secret", type="password")

tpls = {
    "📄 Bio LinkedIn":    "Rédige une bio LinkedIn professionnelle.",
    "✉️ Mail candidature":"Écris un mail de candidature spontanée.",
    "📃 Mini CV":         "Génère un mini-CV (5-7 lignes).",
    "🧩 CV optimisé IA":  "Optimise le CV en soulignant deux mots-clés."
}
choices = st.multiselect("Générations IA", list(tpls.keys()), default=list(tpls.keys())[:2])

# ── 5) ACTION ─────────────────────────────────────────────────────────────
if st.button("🚀 Lancer"):
    # Validations
    if not key_openai:
        st.error("🔑 Clé OpenAI requise"); st.stop()
    if not (key_pe_id and key_pe_secret and sel):
        st.error("🔑 Identifiants Pôle-Emploi et territoires requis"); st.stop()

    # Profile dict
    profile = {
        "job_title": job_title,
        "missions": missions,
        "skills": skills
    }

    # — IA Generations
    st.header("🧠 Génération IA")
    for name in choices:
        prompt_lines = [
            f"Poste: {job_title}",
            f"Missions: {missions}",
            f"Compétences: {skills}"
        ]
        if desired_skills:
            prompt_lines.append(f"Compétences ciblées: {desired_skills}")
        prompt_lines += [
            f"Territoires: {', '.join(sel)}",
            f"Expérience: {exp_level}",
            f"Contrat: {contract}",
            f"Télétravail: {'Oui' if remote else 'Non'}",
            "",
            tpls[name]
        ]
        prompt = "\n".join(prompt_lines)
        try:
            result = get_gpt_response(prompt, key_openai)
            st.subheader(name)
            st.markdown(result)
            if name == "🧩 CV optimisé IA":
                pdf_buf = PDFGen.to_pdf(result)
                st.download_button("📥 Télécharger CV optimisé", data=pdf_buf,
                                   file_name="CV_optimisé.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Erreur IA {name}: {e}")

    # — Top 5 Offres pour le poste
    token    = fetch_ftoken(key_pe_id, key_pe_secret)
    keywords = build_keywords([job_title, skills])
    st.header(f"4️⃣ Top offres pour « {job_title} »")
    all_offres = []
    for loc in sel:
        loc_norm = normalize_location(loc)
        off = search_offres(token, keywords, loc_norm, limit=5)
        off = filter_by_location(off, loc_norm)
        all_offres.extend(off)

    unique_offres = {}
    for o in all_offres:
        link = o.get('contact',{}).get('urlPostulation') or o.get('contact',{}).get('urlOrigine','')
        if link and link not in unique_offres:
            unique_offres[link] = o

    if unique_offres:
        for link, offer in list(unique_offres.items())[:5]:
            lib   = offer['lieuTravail']['libelle']
            title = offer['intitule']
            st.markdown(
                f"**{title}** – {lib}  \n"
                f"<span class='offer-link'><a href='{link}' target='_blank'>Voir</a></span>\n---",
                unsafe_allow_html=True
            )
    else:
        st.info("Aucune offre trouvée pour ce poste dans vos territoires.")

    # — 5️⃣ SIS : Métiers recommandés
    st.header("5️⃣ SIS – Métiers recommandés")
    top6 = scorer_metier(profile, referentiel, top_k=6)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        kws  = build_keywords([r['Metier']])
        subs = []
        for loc in sel:
            loc_norm = normalize_location(loc)
            tmp = search_offres(token, kws, loc_norm, limit=3)
            subs.extend(filter_by_location(tmp, loc_norm))

        seen = set()
        if subs:
            for o in subs:
                link = o.get('contact',{}).get('urlPostulation') or o.get('contact',{}).get('urlOrigine','')
                if link not in seen:
                    seen.add(link)
                    dt   = o.get("dateCreation","")[:10]
                    lib  = o['lieuTravail']['libelle']
                    desc = (o.get("description","") or "").replace("\n"," ")[:150] + "…"
                    st.markdown(
                        f"• **{o['intitule']}** – {lib} (_Publié {dt}_)  \n"
                        f"{desc}  \n"
                        f"<span class='offer-link'><a href='{link}' target='_blank'>Voir</a></span>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("Aucune offre trouvée pour ces métiers dans vos territoires.")

