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
st.set_page_config(page_title="CraftMyJob – Job Seekers Hub France", layout="centered")
st.markdown("""
<style>
  .stButton>button { background-color:#2E86C1; color:white; border-radius:4px; }
  .section-header { font-size:1.2rem; font-weight:600; margin-top:1.5em; color:#2E86C1; }
  h1,h2,h3 { color:#2E86C1; }
  .offer-link a { color:#2E86C1; text-decoration:none; }
  .cv-summary { color:#1F8A70; }
</style>
""", unsafe_allow_html=True)
# Logo
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
def normalize_location(loc: str) -> str:
    """Extract plain name or department for API and filtering."""
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
    # communes
    r1 = requests.get(
        "https://geo.api.gouv.fr/communes",
        params={"nom": query, "fields": "nom,codesPostaux", "limit": limit}, timeout=5
    )
    if r1.status_code == 200:
        for e in r1.json():
            cp = e.get("codesPostaux", ["00000"])[0]
            res.append(f"{e['nom']} ({cp})")
    # régions
    r2 = requests.get(
        "https://geo.api.gouv.fr/regions",
        params={"nom": query, "fields": "nom,code"}, timeout=5
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
            # On remplace les caractères hors Latin-1 pour éviter les erreurs de police
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
# CV upload & extraction
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

# 6.4) Top 30 Offres optimisé
st.header(f"4️⃣ Top 30 offres pour '{job_title}'")

# Requête clef simple (titre seul) ou ajoutez vos variantes manuellement
keywords = job_title

# Récupération paginée des offres
all_offres = []
for loc in sel:
    loc_norm = normalize_location(loc)  # ex. '75' ou 'Paris'
    # fetch_all_offres doit être défini plus haut
    offs = fetch_all_offres(token, keywords, loc_norm)
    all_offres.extend(offs)

# Déplie le dict "lieuTravail" pour filtrage
import pandas as _pd
locs = _pd.json_normalize(all_offres)
locs = locs.rename(columns={
    'lieuTravail.codePostal': 'codePostal',
    'lieuTravail.libelle': 'lieuLibelle'
})
# Concatène avec le DataFrame
_df = _pd.DataFrame(all_offres)
_df['codePostal'] = locs['codePostal']
_df['lieuLibelle'] = locs['lieuLibelle']

# Filtre contrat
_df = _df[_df['typeContrat'].isin(contract)]

# Filtre géographique strict (codePostal si numérique)
def _geo_filter(r, loc):
    cp = str(r['codePostal'])
    if loc.isdigit():
        return cp.startswith(loc)
    return loc.lower() in r['lieuLibelle'].lower()

_df = _pd.concat([
    _df[_df.apply(lambda r: _geo_filter(r, normalize_location(loc)), axis=1)]
    for loc in sel
])
_df = _df.drop_duplicates(subset=['id'])

# Calcul des similarités (TF-IDF + fuzzy)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

vect = TfidfVectorizer(max_features=2000, stop_words="french")
X = vect.fit_transform(_df['intitule'] + ' ' + _df.get('description_extrait', _df.get('description', '')))
q = vect.transform([job_title])
_df['sim_cosine']      = cosine_similarity(q, X).flatten()
_df['sim_title_fuzzy'] = _df['intitule'].apply(lambda t: fuzz.WRatio(t, job_title) / 100)

# Score mixte 70% TF-IDF + 30% fuzzy
df_scores = _df.copy()
df_scores['score_mix'] = 0.7 * df_scores['sim_cosine'] + 0.3 * df_scores['sim_title_fuzzy']
# Appliquer un seuil minimal
df_scores = df_scores[df_scores['score_mix'] >= 0.5]

# Sélection Top 30
top30 = df_scores.nlargest(30, 'score_mix')
for _, o in top30.iterrows():
    title = o.get('intitule','–')
    typ   = o.get('typeContrat','–')
    lib   = o.get('lieuLibelle','–')
    dt    = o.get('dateCreation','')[:10]
    url   = o.get('url') or o.get('contact',{}).get('urlPostulation','')
    pct   = int(o['score_mix'] * 100)
    st.markdown(
        f"**{title}** ({typ}) – {lib} (_Publié {dt}_)  \n"
        f"Score: **{pct}%**  \n"
        f"<span class='offer-link'><a href='{url}' target='_blank'>Voir l'offre</a></span>\n---",
        unsafe_allow_html=True
    )

# 6.5) SIS – Métiers recommandés optimisé
st.header("5️⃣ SIS – Métiers recommandés")

# Comptage des romeCode dans le DataFrame initial
import pandas as _pd
rome_counts = _pd.Series(df['romeCode']).value_counts().rename('freq').reset_index().rename(columns={'index':'romeCode'})

# Scoring SIS original
sis_df = scorer_metier(profile, referentiel, top_k=len(referentiel))
sis_df = sis_df.rename(columns={'Metier':'metier','score':'sis_score'})
# Associer le romeCode
sis_df['romeCode'] = sis_df.apply(lambda r: referentiel[referentiel['Metier']==r['metier']]['romeCode'].values[0], axis=1)
# Fusion fréquence et fillna
sis_df = sis_df.merge(rome_counts, on='romeCode', how='left').fillna({'freq':0})
# Score final (70% sis_score + 30% fréquence normalisée)
max_freq = sis_df['freq'].max() or 1
sis_df['final_score'] = 0.7*sis_df['sis_score'] + 0.3*(sis_df['freq']/max_freq*100)

# Top 6 métiers
top6 = sis_df.nlargest(6, 'final_score')
for _, r in top6.iterrows():
    st.markdown(f"**{r['metier']}** – {int(r['final_score'])}% ({int(r['freq'])} offres)")
