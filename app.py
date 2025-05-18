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
from datetime import datetime, timedelta

# ── 0) CLEAN ENVIRONMENT ─────────────────────────────────────────────────
for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(var, None)

# ── 1) STREAMLIT CONFIG & STYLING ─────────────────────────────────────────
st.set_page_config(page_title="CraftMyJob – JOB SEEKERS HUB France", layout="centered")
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
st.title("CraftMyJob – Votre assistant emploi intelligent")

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
def search_territoires(query: str, limit: int = 10) -> list[str]:
    res = []
    if re.fullmatch(r"\d{2}", query):
        r = requests.get(
            f"https://geo.api.gouv.fr/departements/{query}/communes",
            params={"fields": "nom,codesPostaux", "limit": limit}, timeout=5
        )
        r.raise_for_status()
        for e in r.json():
            cp = e.get("codesPostaux", ["00000"])[0]
            res.append(f"{e['nom']} ({cp})")
        res.append(f"Département {query}")
        return list(dict.fromkeys(res))
    r1 = requests.get(
        "https://geo.api.gouv.fr/communes",
        params={"nom": query, "fields": "nom,codesPostaux", "limit": limit}, timeout=5
    )
    if r1.status_code == 200:
        for e in r1.json():
            cp = e.get("codesPostaux", ["00000"])[0]
            res.append(f"{e['nom']} ({cp})")
    r2 = requests.get(
        "https://geo.api.gouv.fr/regions",
        params={"nom": query, "fields": "nom,code"}, timeout=5
    )
    if r2.status_code == 200:
        for rg in r2.json():
            res.append(f"{rg['nom']} (region:{rg['code']})")
    return list(dict.fromkeys(res))


def normalize_location(loc: str) -> str:
    if m := re.match(r"^(.+?) \((\d{5})\)", loc):
        return m.group(1)
    if m2 := re.match(r"Département (\d{2})", loc):
        return m2.group(1)
    if m3 := re.match(r"^(.+) \(region:(\d+)\)", loc):
        return m3.group(1)
    return loc


def get_date_range(months: int = 2):
    end = datetime.now().date()
    start = end - timedelta(days=months * 30)
    return start.isoformat(), end.isoformat()


def get_gpt_response(prompt: str, key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Tu es un expert en recrutement et en personal branding."},
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
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 8, safe_line)
        pdf.output(buf)
        buf.seek(0)
        return buf


def fetch_ftoken(cid: str, secret: str) -> str:
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "grant_type": "client_credentials",
        "client_id": cid,
        "client_secret": secret,
        "scope": "api_offresdemploiv2 o2dsoffre"
    }
    r = requests.post(url, data=data, timeout=10)
    r.raise_for_status()
    return r.json().get("access_token", "")


def search_offres(token: str, mots: str, lieu: str, limit: int = 5) -> list:
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    dateDebut, dateFin = get_date_range(2)
    params = {
        "motsCles": mots,
        "localisation": lieu,
        "range": f"0-{limit-1}",
        "dateDebut": dateDebut,
        "dateFin": dateFin,
        "tri": "dateCreation"
    }
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=10)
    if r.status_code == 204:
        return []
    if r.status_code not in (200, 206):
        st.error(f"FT API {r.status_code}: {r.text}")
        return []
    return r.json().get("resultats", [])


def filter_by_location(offers: list, loc_norm: str) -> list:
    out = []
    cp_norm = loc_norm.lower()
    for o in offers:
        lib = o.get('lieuTravail', {}).get('libelle', "").lower()
        cp  = str(o.get('lieuTravail', {}).get('codePostal', ""))
        if cp_norm in lib or cp_norm == cp:
            out.append(o)
    return out


def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    doc = f"{inp['missions']} {inp['skills']} {inp['job_title']}"
    v_user = vecteur.transform([doc])
    cos = cosine_similarity(v_user, tfidf_matrix).flatten()
    df2 = df.copy()
    df2['cosine'] = cos
    df2['fz_t'] = df2['Metier'].apply(lambda m: fuzz.token_set_ratio(m, inp['job_title']) / 100)
    df2['fz_m'] = df2['Activites'].apply(lambda a: fuzz.token_set_ratio(a, inp['missions']) / 100)
    df2['fz_c'] = df2['Competences'].apply(lambda c: fuzz.token_set_ratio(c, inp['skills']) / 100)
    df2['score'] = (0.5*df2['cosine'] + 0.2*df2['fz_t'] + 0.15*df2['fz_m'] + 0.15*df2['fz_c']) * 100
    return df2.nlargest(top_k, 'score')

# ── 4) PROFILE FORM ──────────────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")
cv_text = ""
up = st.file_uploader("📂 CV (optionnel)", type=["pdf","docx","txt"])
if up:
    ext = up.name.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        cv_text = ' '.join(p.extract_text() or '' for p in PdfReader(up).pages)
    elif ext == 'docx':
        cv_text = ' '.join(p.text for p in Document(up).paragraphs)
    else:
        cv_text = up.read().decode(errors='ignore')

job_title = st.text_input("🔤 Poste souhaité")
missions  = st.text_area("📋 Missions principales")
skills    = st.text_area("🧠 Compétences clés")

st.markdown("<div class='section-header'>🌍 Territoires</div>", unsafe_allow_html=True)
typed = st.text_input("Tapez commune/département/région…")
opts  = search_territoires(typed) if typed else []
default_locs = st.session_state.get('locations', [])
sel   = st.multiselect("Sélectionnez vos territoires", options=(default_locs+opts), default=default_locs)
st.session_state['locations'] = sel

exp_level = st.radio("🎯 Expérience", ["Débutant (0-2 ans)", "Expérimenté (2-5 ans)", "Senior (5+ ans)"])
contract = st.multiselect("📄 Types de contrat", ["CDI","CDD","Freelance","Stage","Alternance"], default=["CDI","CDD","Freelance"])
remote   = st.checkbox("🏠 Full remote")

# ── 5) CLÉS API & IA ───────────────────────────────────────────────────
st.header("2️⃣ Clés API & IA")
key_openai    = st.text_input("🔑 OpenAI API Key", type="password")
key_pe_id     = st.text_input("🔑 Pôle-Emploi Client ID", type="password")
key_pe_secret = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

tpls = {
    '📄 Bio LinkedIn':    'Rédige une bio LinkedIn professionnelle.',
    '✉️ Mail de candidature': 'Écris un mail de candidature spontanée.',
    '📃 Mini CV':         'Génère un mini-CV (5-7 lignes).',
    '🧩 CV optimisé IA':  'Optimise le CV en soulignant deux mots-clés.'
}
choices = st.multiselect("Générations IA", list(tpls.keys()), default=list(tpls.keys())[:2])

# ── 6) ACTION ─────────────────────────────────────────────────────────
if st.button("🚀 Lancer tout"):
    if not key_openai:
        st.error("🔑 Clé OpenAI requise")
        st.stop()
    if not (key_pe_id and key_pe_secret and sel):
        st.error("🔑 Identifiants Pôle-Emploi et territoires requis")
        st.stop()

    profile = { 'job_title': job_title, 'missions': missions, 'skills': skills }

    # — 6.1) Résumé CV
    if cv_text:
        prompt_summary = f"Résumé en 5 points clés du CV suivant:\n{cv_text[:1000]}"
        summary = get_gpt_response(prompt_summary, key_openai)
        st.markdown("**Résumé CV:**", unsafe_allow_html=True)
        for line in summary.split('\n'):
            st.markdown(f"- <span class='cv-summary'>{line.strip()}</span>", unsafe_allow_html=True)

    # — 6.2) Générations IA
    st.header("🧠 Génération IA")
    for name in choices:
        instruction = tpls[name]
        if name == '📄 Bio LinkedIn':
            instruction += ' Ne mentionne aucune localisation ni ancienneté, limite à 4 lignes.'
        prompt_lines = [
            f"Poste: {job_title}",
            f"Missions: {missions}",
            f"Compétences: {skills}",
        ]
        if cv_text:
            prompt_lines.append(f"Résumé CV: {summary[:300]}")
        prompt_lines += [
            f"Territoires: {', '.join(sel)}",
            f"Expérience: {exp_level}",
            f"Contrat(s): {', '.join(contract)}",
            f"Télétravail: {'Oui' if remote else 'Non'}",
            '', instruction
        ]
        prompt = "\n".join(prompt_lines)
        try:
            res = get_gpt_response(prompt, key_openai)
            st.subheader(name)
            st.markdown(res)
            if name == '🧩 CV optimisé IA':
                buf = PDFGen.to_pdf(res)
                st.download_button("📥 Télécharger CV optimisé", data=buf, file_name="CV_optimise.pdf", mime="application/pdf")
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                st.error("Clé OpenAI invalide ou expirée.")
            else:
                st.error(f"Erreur OpenAI ({e.response.status_code}) : {e.response.text}")
            st.stop()

    # — 6.3) Token Pôle-Emploi
    try:
        token = fetch_ftoken(key_pe_id, key_pe_secret)
    except requests.HTTPError as e:
        status = e.response.status_code
        if status == 401:
            st.error("🔑 Identifiants Pôle-Emploi invalides ou expirés.")
        else:
            st.error(f"Erreur Pôle-Emploi (code {status}) : {e.response.text}")
        st.stop()

    # — 6.4) Top offres────────────────────────────────────────────────
    st.header(f"4️⃣ Top offres pour '{job_title}'")

    # Mots-clés ATS
    ats = build_keywords([missions, skills], max_terms=5)
    st.markdown("**Mots-clés recommandés :**")
    for tag in ats.split(","):
        st.markdown(f"<span class='ats-tag'>{tag}</span>", unsafe_allow_html=True)
    st.markdown("---")

    all_offres = []
    for loc in sel:
        loc_norm = normalize_location(loc)
        offs = search_offres(token, job_title, loc_norm, limit=5)
        offs = filter_by_location(offs, loc_norm)
        all_offres.extend(offs)
    all_offres = [o for o in all_offres if o.get('typeContrat','') in contract]

    seen = {}
    for o in all_offres:
        url = o.get('contact',{}).get('urlPostulation') or o.get('contact',{}).get('urlOrigine','')
        if url and url not in seen:
            seen[url] = o
    top_offres = list(seen.values())[:30]

    for o in top_offres:
        title = o.get('intitule','–')
        wr = fuzz.WRatio(title, job_title)
        pr = fuzz.partial_ratio(title, job_title)
        dr = fuzz.partial_ratio(o.get('description_extrait','')[:200], missions)
        score = int(0.5*wr + 0.3*pr + 0.2*dr)
        header = f"{title} — {score}%"
        with st.expander(header):
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown(f"- **Contrat** : {o.get('typeContrat','–')}")
                st.markdown(f"- **Lieu** : {o.get('lieuTravail',{}).get('libelle','–')}")
                st.markdown(f"- **Publié** : {o.get('dateCreation','')[:10]}")
                desc = (o.get('description_extrait','') or o.get('description',''))
                st.markdown(f"**Description :** {desc[:150]}…")
                st.markdown(f"<span class='offer-link'><a href='{url}' target='_blank'>Voir l'offre</a></span>", unsafe_allow_html=True)
            with col2:
                st.progress(score/100)

    # — 6.5) SIS – Métiers recommandés────────────────────────────────────
    st.header("5️⃣ SIS – Métiers recommandés")
    top6 = scorer_metier(profile, referentiel, top_k=6)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        subs = []
        for loc in sel:
            loc_norm = normalize_location(loc)
            subs.extend(search_offres(token, r['Metier'], loc_norm, limit=3))
        subs = [o for o in subs if o.get('typeContrat','') in contract]
        seen2 = set()
        if subs:
            for o in subs:
                url2 = o.get('contact',{}).get('urlPostulation') or o.get('contact',{}).get('urlOrigine','')
                if not url2 or url2 in seen2: continue
                seen2.add(url2)
                title2 = o.get('intitule','–')
                typ2   = o.get('typeContrat','–')
                lieu2  = o.get('lieuTravail',{}).get('libelle','–')
                date2  = o.get('dateCreation','')[:10]
                desc2  = (o.get('description_extrait','') or o.get('description',''))[:150] + '…'
                st.markdown(
                    f"• **{title2}** ({typ2}) – {lieu2} (_Publié {date2}_)  \n"
                    f"{desc2}  \n"
                    f"<span class='offer-link'><a href='{url2}' target='_blank'>Voir / Postuler</a></span>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Aucune offre trouvée pour ce métier dans vos territoires et contrats.")
