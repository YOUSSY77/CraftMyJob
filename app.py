# -*- coding: utf-8 -*-
"""
CraftMyJob – Streamlit app for smart job suggestions with extended date range.
"""

import os
import io
import re
from datetime import datetime, timedelta

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

# 0. CLEAN ENVIRONMENT
ENV_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")
for var in ENV_VARS:
    os.environ.pop(var, None)

# 1. STREAMLIT CONFIGURATION & STYLING
st.set_page_config(
    page_title="CraftMyJob – Job Seekers Hub France",
    layout="centered"
)
st.markdown(
    """
    <style>
      .stButton>button { background-color:#2E86C1; color:white; border-radius:4px; }
      .section-header { font-size:1.2rem; font-weight:600; margin-top:1.5em; color:#2E86C1; }
      h1,h2,h3 { color:#2E86C1; }
      .offer-link a { color:#2E86C1; text-decoration:none; }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    logo = Image.open("logo_jobseekers.PNG")
    st.image(logo, width=120)
except FileNotFoundError:
    pass

st.title("✨ CraftMyJob – Votre assistant emploi intelligent")

# 2. DATA & MODEL PREPARATION
@st.cache_data
def load_referentiel(path: str = "referentiel_metiers_craftmyjob_final.csv") -> pd.DataFrame:
    """Load métiers referentiel from CSV."""
    df = pd.read_csv(path, dtype=str).fillna("")
    return df


@st.cache_data
def build_tfidf(df: pd.DataFrame, max_features: int = 2000):
    """Build TF-IDF vectorizer and matrix from referentiel."""
    corpus = df["Activites"] + " " + df["Competences"] + " " + df["Metier"]
    vect = TfidfVectorizer(max_features=max_features)
    matrix = vect.fit_transform(corpus)
    return vect, matrix


referentiel = load_referentiel()
vectorizer, tfidf_matrix = build_tfidf(referentiel)

# 3. UTILITIES
def normalize_location(loc: str) -> str:
    """Normalize location string to name or department code."""
    match = re.match(r"^(.+?) \(\d{5}\)", loc)
    if match:
        return match.group(1)
    match = re.match(r"Département (\d{2})", loc)
    if match:
        return match.group(1)
    match = re.match(r"^(.+) \(region:(\d+)\)", loc)
    if match:
        return match.group(1)
    return loc


def get_date_range(months: int = 2) -> tuple[str, str]:
    """Get ISO date range from now - `months` to today."""
    end = datetime.now().date()
    start = end - timedelta(days=months * 30)
    return start.isoformat(), end.isoformat()


def search_territoires(query: str, limit: int = 10) -> list[str]:
    """Search communes, départements, and regions via geo.api.gouv.fr."""
    results: list[str] = []
    if re.fullmatch(r"\d{2}", query):
        resp = requests.get(
            f"https://geo.api.gouv.fr/departements/{query}/communes",
            params={"fields": "nom,codesPostaux", "limit": limit},
            timeout=5
        )
        resp.raise_for_status()
        for entry in resp.json():
            cp = entry.get("codesPostaux", ["00000"])[0]
            results.append(f"{entry['nom']} ({cp})")
        results.append(f"Département {query}")
        return list(dict.fromkeys(results))

    resp_comm = requests.get(
        "https://geo.api.gouv.fr/communes",
        params={"nom": query, "fields": "nom,codesPostaux", "limit": limit},
        timeout=5
    )
    if resp_comm.ok:
        for entry in resp_comm.json():
            cp = entry.get("codesPostaux", ["00000"])[0]
            results.append(f"{entry['nom']} ({cp})")

    resp_reg = requests.get(
        "https://geo.api.gouv.fr/regions",
        params={"nom": query, "fields": "nom,code"},
        timeout=5
    )
    if resp_reg.ok:
        for entry in resp_reg.json():
            results.append(f"{entry['nom']} (region:{entry['code']})")

    return list(dict.fromkeys(results))


def build_keywords(texts: list[str], max_terms: int = 7) -> str:
    """Extract up to `max_terms` keywords from list of texts."""
    combined = " ".join(texts).lower()
    tokens = re.findall(r"\w{2,}", combined)
    stopwords = {
        "et", "ou", "la", "le", "les", "de", "des", "du", "un", "une",
        "à", "en", "pour", "par", "avec", "sans", "sur", "dans", "au", "aux"
    }
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_terms:
            break
    return ",".join(keywords)


def get_gpt_response(prompt: str, api_key: str) -> str:
    """Get response text from OpenAI ChatCompletion."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": (
                "Tu es un expert en recrutement et personal branding."
            )},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    resp = requests.post(url, json=data, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


class PDFGenerator:
    """Utility class for generating PDF from text."""
    @staticmethod
    def to_pdf(text: str) -> io.BytesIO:
        buffer = io.BytesIO()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 8, line)
        pdf.output(buffer)
        buffer.seek(0)
        return buffer


def fetch_ftoken(client_id: str, client_secret: str) -> str:
    """Fetch OAuth2 token from Pôle-Emploi."""
    url = (
        "https://entreprise.pole-emploi.fr"
        "/connexion/oauth2/access_token?realm=/partenaire"
    )
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "api_offresdemploiv2 o2dsoffre"
    }
    resp = requests.post(url, data=data, timeout=10)
    resp.raise_for_status()
    return resp.json().get("access_token", "")


def search_offres(
    token: str,
    mots_cles: str,
    localisation: str,
    limit: int = 5
) -> list[dict]:
    """Search job offers via Francetravail API."""
    start_date, end_date = get_date_range(2)
    params = {
        "motsCles": mots_cles,
        "localisation": localisation,
        "range": f"0-{limit-1}",
        "dateDebut": start_date,
        "dateFin": end_date,
        "tri": "dateCreation"
    }
    url = (
        "https://api.francetravail.io/partenaire/"
        "offresdemploi/v2/offres/search"
    )
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=10
    )
    if resp.status_code == 204:
        return []
    if resp.status_code not in (200, 206):
        st.error(f"FT API {resp.status_code}: {resp.text}")
        return []
    return resp.json().get("resultats", [])


def filter_offers_by_location(
    offers: list[dict],
    loc_norm: str
) -> list[dict]:
    """Filter offers by normalized location in label or postal code."""
    filtered: list[dict] = []
    loc_low = loc_norm.lower()
    for offer in offers:
        libelle = offer.get("lieuTravail", {}).get("libelle", "").lower()
        cp = str(offer.get("lieuTravail", {}).get("codePostal", ""))
        if loc_low in libelle or loc_low == cp:
            filtered.append(offer)
    return filtered


def scorer_metier(
    profile: dict,
    df: pd.DataFrame,
    top_k: int = 6
) -> pd.DataFrame:
    """Score and return top_k métiers based on user profile."""
    document = (
        f"{profile['missions']} {profile['skills']} "
        f"{profile['job_title']}"
    )
    user_vector = vectorizer.transform([document])
    cosine_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    df_copy = df.copy()
    df_copy['cosine'] = cosine_scores
    df_copy['fuzz_title'] = df_copy['Metier'].apply(
        lambda m: fuzz.token_set_ratio(m, profile['job_title']) / 100
    )
    df_copy['fuzz_missions'] = df_copy['Activites'].apply(
        lambda a: fuzz.token_set_ratio(a, profile['missions']) / 100
    )
    df_copy['fuzz_skills'] = df_copy['Competences'].apply(
        lambda c: fuzz.token_set_ratio(c, profile['skills']) / 100
    )
    df_copy['score'] = (
        0.5 * df_copy['cosine'] +
        0.2 * df_copy['fuzz_title'] +
        0.15 * df_copy['fuzz_missions'] +
        0.15 * df_copy['fuzz_skills']
    ) * 100
    return df_copy.nlargest(top_k, 'score')

# 4. PROFILE FORM
st.header("1️⃣ Profil & préférences")
cv_text = """
"""
uploaded_file = st.file_uploader(
    "📂 CV (optionnel)",
    type=["pdf", "docx", "txt"]
)
if uploaded_file:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        cv_text = "".join(
            page.extract_text() or "" for page in PdfReader(uploaded_file).pages
        )
    elif ext == "docx":
        cv_text = "".join(
            p.text for p in Document(uploaded_file).paragraphs
        )
    else:
        cv_text = uploaded_file.read().decode(errors="ignore")

job_title = st.text_input("🔤 Poste souhaité")
missions = st.text_area("📋 Missions principales")
skills = st.text_area("🧠 Compétences clés")

st.markdown(
    "<div class='section-header'>🌍 Territoires</div>",
    unsafe_allow_html=True
)
typed = st.text_input("Tapez commune/département/région…")
territoire_opts = search_territoires(typed) if typed else []
default_locs = st.session_state.get("locations", [])
locations = st.multiselect(
    "Sélectionnez vos territoires",
    options=default_locs + territoire_opts,
    default=default_locs
)
st.session_state.locations = locations

exp_level = st.radio(
    "🎯 Expérience",
    ["Débutant (0-2 ans)", "Expérimenté (2-5 ans)", "Senior (5+ ans)"]
)

contract_types = ["CDI", "CDD", "Freelance", "Stage", "Alternance"]
contracts = st.multiselect(
    "📄 Types de contrat",
    options=contract_types,
    default=["CDI", "CDD", "Freelance"]
)
remote = st.checkbox("🏠 Full remote")

# 5. API KEYS & IA CHOICES
st.header("2️⃣ Clés API & IA")
openai_key = st.text_input("🔑 OpenAI API Key", type="password")
pe_id = st.text_input("🔑 Pôle-Emploi ID", type="password")
pe_secret = st.text_input("🔑 Pôle-Emploi Secret", type="password")

tpl_ia = {
    "📄 Bio LinkedIn": "Rédige une bio LinkedIn professionnelle.",
    "✉️ Mail candidature": "Écris un mail de candidature spontanée.",
    "📃 Mini CV": "Génère un mini-CV (5-7 lignes).",
    "🧩 CV optimisé IA": "Optimise le CV en soulignant deux mots-clés."
}
ia_choices = st.multiselect(
    "Générations IA",
    options=list(tpl_ia.keys()),
    default=list(tpl_ia.keys())[:2]
)

# 6. ACTION BUTTON
if st.button("🚀 Lancer tout"):
    if not openai_key:
        st.error("Clé OpenAI requise")
        st.stop()
    if not (pe_id and pe_secret and locations):
        st.error("Pôle-Emploi + territoires requis")
        st.stop()

    profile = {"job_title": job_title, "missions": missions, "skills": skills}

    # IA Generations
    st.header("🧠 Génération IA")
    for name in ia_choices:
        instr = tpl_ia[name]
        if name == "📄 Bio LinkedIn":
            instr += (
                " Ne mentionne aucune localisation "
                "(pas de 'basé à ...') et fais une bio de max 4 lignes."
            )
        prompt = "\n".join([
            f"Poste: {job_title}",
            f"Missions: {missions}",
            f"Compétences: {skills}",
            f"Territoires: {', '.join(locations)}",
            f"Expérience: {exp_level}",
            f"Contrat(s): {', '.join(contracts)}",
            f"Télétravail: {'Oui' if remote else 'Non'}",
            "",
            instr
        ])
        try:
            res = get_gpt_response(prompt, openai_key)
            st.subheader(name)
            st.markdown(res)
            if name == "🧩 CV optimisé IA":
                buf = PDFGenerator.to_pdf(res)
                st.download_button(
                    "📥 Télécharger CV optimisé",
                    data=buf,
                    file_name="CV_optimise.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"Erreur IA {name}: {e}")

    # Pôle-Emploi token
    token = fetch_ftoken(pe_id, pe_secret)

    # 4️⃣ Top offres
    st.header(f"4️⃣ Top offres pour « {job_title} »")
    user_title = job_title.strip()
    keywords = build_keywords([user_title])
    all_offres: list[dict] = []
    for loc in locations:
        loc_norm = normalize_location(loc)
        offres = search_offres(token, keywords, loc_norm, limit=10)
        if not offres:
            offres = search_offres(token, user_title, loc_norm, limit=10)
        for o in offres:
            if fuzz.token_set_ratio(
                o.get("intitule", "").lower(), user_title.lower()
            ) >= 60:
                all_offres.append(o)
    # Filter by contract
    all_offres = [o for o in all_offres if o.get("typeContrat") in contracts]
    # Deduplicate & display
    seen: dict[str, dict] = {}
    for o in all_offres:
        url = o.get("contact", {}).get("urlPostulation") or o.get("contact", {}).get("urlOrigine", "")
        if url and url not in seen:
            seen[url] = o
    if seen:
        for url, o in list(seen.items())[:5]:
            st.markdown(
    f"**{o.get('intitule', '–')}** ({o.get('typeContrat', '–')}) – {o['lieuTravail']['libelle']} [{o['lieuTravail'].get('codePostal', '')}]  
"
    f"<span class='offer-link'><a href='{url}' target='_blank'>Voir l'offre</a></span>
---",
    unsafe_allow_html=True
            )
    else:
        st.info("Aucune offre trouvée pour ce poste dans vos territoires et contrats.")

    # 5️⃣ SIS – Métiers recommandés
    st.header("5️⃣ SIS – Métiers recommandés")
    top6 = scorer_metier(profile, referentiel, top_k=6)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
        suggestions: list[dict] = []
        for loc in locations:
            loc_norm = normalize_location(loc)
            subs = search_offres(token, r['Metier'], loc_norm, limit=3)
            subs = filter_offers_by_location(subs, loc_norm)
            suggestions.extend(subs)
        # filter by contract
        suggestions = [o for o in suggestions if o.get('typeContrat') in contracts]
        seen2: set[str] = set()
        if suggestions:
            for o in suggestions:
                url2 = o.get('contact', {}).get('urlPostulation') or o.get('contact', {}).get('urlOrigine', '')
                if url2 and url2 not in seen2:
                    seen2.add(url2)
                    dt = o.get('dateCreation', '')[:10]
                    lib = o['lieuTravail']['libelle']
                    typ = o.get('typeContrat', '–')
                    desc = (o.get('description', '') or "").replace("\n", " ")[:150] + "…"
                    st.markdown(
                        f"• **{o['intitule']}** ({typ}) – {lib} (_Publié {dt}_)  \n"
                        f"{desc}  \n"
                        f"<span class='offer-link'><a href='{url2}' target='_blank'>Voir / Postuler</a></span>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("Aucune offre trouvée pour ce métier dans vos territoires et contrats.")

