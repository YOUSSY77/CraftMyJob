# -*- coding: utf-8 -*-
"""
CraftMyJob – Streamlit app for smart job suggestions with pagination & pertinence
"""
import os
import re
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
    """Extract code postal or department or leave name."""
    if m := re.match(r"^(.+?) \((\d{5})\)$", loc):
        return m.group(2)
    if m2 := re.match(r"Département (\d{2})$", loc):
        return m2.group(1)
    if m3 := re.match(r"^(.+) \(region:(\d+)\)$", loc):
        return m3.group(1).lower()
    return loc.lower()

def get_date_range(months: int = 2):
    end = datetime.now().date()
    start = end - timedelta(days=months * 30)
    return start.isoformat(), end.isoformat()

def search_territoires(query: str, limit: int = 10) -> list[str]:
    res = []
    if re.fullmatch(r"\d{2}", query):
        r = requests.get(f"https://geo.api.gouv.fr/departements/{query}/communes",
                         params={"fields": "nom,codesPostaux", "limit": limit}, timeout=5)
        r.raise_for_status()
        for e in r.json():
            cp = e.get("codesPostaux", ["00000"])[0]
            res.append(f"{e['nom']} ({cp})")
        res.append(f"Département {query}")
        return list(dict.fromkeys(res))
    for endpoint, field in [("communes","nom"), ("regions","nom")]:
        r = requests.get(
            f"https://geo.api.gouv.fr/{endpoint}",
            params={"nom": query, "fields": f"{field},codesPostaux" if endpoint=="communes" else f"{field},code", "limit": limit},
            timeout=5
        )
        if r.ok:
            for e in r.json():
                if endpoint=="communes":
                    cp = e.get("codesPostaux", ["00000"])[0]
                    res.append(f"{e['nom']} ({cp})")
                else:
                    res.append(f"{e['nom']} (region:{e['code']})")
    return list(dict.fromkeys(res))

def get_gpt_response(prompt: str, key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Tu es un expert en recrutement et en personal branding."},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    r = requests.post(url, json=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def fetch_ftoken(cid: str, secret: str) -> str:
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "grant_type":    "client_credentials",
        "client_id":     cid,
        "client_secret": secret,
        "scope":         "api_offresdemploiv2 o2dsoffre"
    }
    r = requests.post(url, data=data, timeout=10)
    r.raise_for_status()
    return r.json().get("access_token", "")

def fetch_all_offres(token: str, mots: str, lieu: str, batch_size: int = 100, max_batches: int = 5) -> list:
    """
    Paginate through the API, fetching up to batch_size * max_batches offers per territory.
    """
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    dateDebut, dateFin = get_date_range(2)
    all_offres = []
    for batch in range(max_batches):
        start = batch * batch_size
        end = start + batch_size - 1
        params = {
            "motsCles": mots,
            "localisation": lieu,
            "range": f"{start}-{end}",
            "dateDebut": dateDebut,
            "dateFin": dateFin,
            "tri": "pertinence"
        }
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=10)
        if r.status_code == 204:
            break
        if r.status_code not in (200, 206):
            st.error(f"FT API {r.status_code}: {r.text}")
            break
        results = r.json().get("resultats", [])
        if not results:
            break
        all_offres.extend(results)
        if len(results) < batch_size:
            break
    return all_offres

def filter_by_location(offers: list, loc_norm: str) -> list:
    """Keep only offers whose lieuTravail_libelle contains loc_norm."""
    out = []
    ln = loc_norm.lower()
    for o in offers:
        lib = o.get("lieuTravail_libelle", "").lower()
        if ln in lib:
            out.append(o)
    return out

def generate_title_variants(title: str) -> list[str]:
    variants = {title.strip()}
    if not title.endswith("s"):
        variants.add(f"{title}s")
    else:
        variants.add(title.rstrip("s"))
    synmap = {
        "commercial": ["sales", "ingénieur commercial", "chargé de clientèle"],
        "developer": ["développeur"],
        "manager": ["responsable", "chef"],
    }
    for w in title.lower().split():
        if w in synmap:
            variants.update(synmap[w])
    return list(variants)

def select_discriminant_skills(text: str, vectorizer: TfidfVectorizer, top_n: int = 5) -> list[str]:
    tokens = re.findall(r"\w{2,}", text.lower())
    features = vectorizer.get_feature_names_out()
    idf_vals = vectorizer.idf_
    idf_map = dict(zip(features, idf_vals))
    scored = sorted({(t, idf_map.get(t,0)) for t in tokens}, key=lambda x: x[1], reverse=True)
    return [t for t,_ in scored[:top_n]]

def scorer_metier(inp: dict, df: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    doc = f"{inp['missions']} {inp['skills']} {inp['job_title']}"
    v = vecteur.transform([doc])
    cos = cosine_similarity(v, tfidf_matrix).flatten()
    df2 = df.copy()
    df2["cosine"] = cos
    df2["fz_t"] = df2["Metier"].apply(lambda m: fuzz.token_set_ratio(m, inp["job_title"])/100)
    df2["fz_m"] = df2["Activites"].apply(lambda a: fuzz.token_set_ratio(a, inp["missions"])/100)
    df2["fz_c"] = df2["Competences"].apply(lambda c: fuzz.token_set_ratio(c, inp["skills"])/100)
    df2["score"] = (0.5*df2["cosine"] + 0.2*df2["fz_t"] + 0.15*df2["fz_m"] + 0.15*df2["fz_c"])*100
    return df2.nlargest(top_k, "score")

# ── 4) PROFILE FORM ──────────────────────────────────────────────────────
st.header("1️⃣ Profil & préférences")
cv_text = ""
up = st.file_uploader("📂 CV (optionnel)", type=["pdf","docx","txt"])
if up:
    ext = up.name.rsplit(".",1)[-1].lower()
    if ext=="pdf":
        cv_text = " ".join(p.extract_text() or "" for p in PdfReader(up).pages)
    elif ext=="docx":
        cv_text = " ".join(p.text for p in Document(up).paragraphs)
    else:
        cv_text = up.read().decode(errors="ignore")

job_title = st.text_input("🔤 Poste souhaité")
missions  = st.text_area("📋 Missions principales")
skills    = st.text_area("🧠 Compétences clés")

st.markdown("<div class='section-header'>🌍 Territoires</div>", unsafe_allow_html=True)
typed = st.text_input("Tapez commune/département/région…")
opts  = search_territoires(typed) if typed else []
sel   = st.multiselect("Sélectionnez vos territoires", options=opts, default=[])
exp_level = st.radio("🎯 Expérience", ["Débutant (0-2 ans)","Expérimenté (2-5 ans)","Senior (5+ ans)"])
contract   = st.multiselect("📄 Types de contrat", ["CDI","CDD","Freelance","Stage","Alternance"], default=["CDI","CDD","Freelance"])
remote     = st.checkbox("🏠 Full remote")

# ── 5) CLÉS API & IA ───────────────────────────────────────────────────
st.header("2️⃣ Clés API & IA")
key_openai    = st.text_input("🔑 OpenAI API Key", type="password")
key_pe_id     = st.text_input("🔑 Pôle-Emploi Client ID", type="password")
key_pe_secret = st.text_input("🔑 Pôle-Emploi Client Secret", type="password")

tpls = {
    "📄 Bio LinkedIn": "Rédige une bio LinkedIn professionnelle.",
    "✉️ Mail de candidature": "Écris un mail de candidature spontanée.",
    "📃 Mini CV": "Génère un mini-CV (5-7 lignes)."
}
choices = st.multiselect("Générations IA", list(tpls.keys()), default=list(tpls.keys())[:2])

# ── 6) ACTION ─────────────────────────────────────────────────────────
if st.button("🚀 Lancer tout"):
    if not key_openai:
        st.error("🔑 Clé OpenAI requise"); st.stop()
    if not (key_pe_id and key_pe_secret and sel):
        st.error("🔑 Identifiants Pôle-Emploi et territoires requis"); st.stop()

    profile = {"job_title": job_title, "missions": missions, "skills": skills}
    # 6.1) Résumé CV
    if cv_text:
        summary = get_gpt_response(f"Résumé en 5 points clés du CV suivant:\n{cv_text[:1000]}", key_openai)
        st.markdown("**Résumé CV:**", unsafe_allow_html=True)
        for line in summary.split("\n"):
            st.markdown(f"- <span class='cv-summary'>{line}</span>", unsafe_allow_html=True)

    # 6.2) Générations IA
    st.header("🧠 Génération IA")
    for name in choices:
        inst = tpls[name]
        if name == "📄 Bio LinkedIn":
            inst += " Ne mentionne aucune localisation ni ancienneté, limite à 4 lignes."
        prompt = "\n".join([
            f"Poste: {job_title}",
            f"Missions: {missions}",
            f"Compétences: {skills}",
            *( [f"Résumé CV: {summary[:300]}"] if cv_text else [] ),
            f"Territoires: {', '.join(sel)}",
            f"Expérience: {exp_level}",
            f"Contrat(s): {', '.join(contract)}",
            f"Télétravail: {'Oui' if remote else 'Non'}",
            "", inst
        ])
        try:
            res = get_gpt_response(prompt, key_openai)
            st.subheader(name)
            st.markdown(res)
        except requests.HTTPError as e:
            st.error(f"Erreur OpenAI ({e.response.status_code}): {e.response.text}"); st.stop()

    # 6.3) Token Pôle-Emploi (avec debug & strip)
    cid    = key_pe_id.strip()
    secret = key_pe_secret.strip()
    st.write(f"🔍 Debug Client ID: '{cid}' (longueur {len(cid)})")
    try:
        token = fetch_ftoken(cid, secret)
        st.success("✅ Token Pôle-Emploi récupéré avec succès")
    except requests.HTTPError as e:
        st.error(f"⛔️ Erreur Pôle-Emploi ({e.response.status_code}) : {e.response.text}")
        st.stop()

    # 6.4) Top 30 Offres
    st.header(f"4️⃣ Top 30 offres pour '{job_title}'")
    variants    = generate_title_variants(job_title)
    disc_skills = select_discriminant_skills(skills, vecteur, top_n=5)
    keywords    = " ".join([job_title] + variants + disc_skills)

    all_offres = []
    for loc in sel:
        loc_norm = normalize_location(loc)
        offs = fetch_all_offres(token, keywords, loc_norm, batch_size=100, max_batches=5)
        filtered = filter_by_location(offs, loc_norm)
        all_offres.extend(filtered)

    all_offres = [o for o in all_offres if o.get("typeContrat","") in contract]

    # Dedup + scoring
    seen = {}
    for o in all_offres:
        url = o.get("url","") or o.get("contact",{}).get("urlPostulation","") or o.get("contact",{}).get("urlOrigine","")
        if url and url not in seen:
            seen[url] = o
    candidates = list(seen.values())
    for o in candidates:
        t_sim = fuzz.token_set_ratio(o.get("intitule",""), job_title)
        d_sim = fuzz.token_set_ratio(o.get("description_extrait","")[:200], missions)
        o["score_match"] = 0.7 * t_sim + 0.3 * d_sim

    top30 = sorted(candidates, key=lambda x: x["score_match"], reverse=True)[:30]
    if top30:
        for o in top30:
            st.markdown(
                f"**{o['intitule']}** ({o['typeContrat']}) – {o['lieuTravail_libelle']} "
                f"(_Publié {o['dateCreation'][:10]}_)  \n"
                f"Match: **{int(o['score_match'])}%**  \n"
                f"<span class='offer-link'><a href='{o['url']}' target='_blank'>Voir l'offre</a></span>\n---",
                unsafe_allow_html=True
            )
    else:
        st.info("Aucune offre pertinente trouvée pour ce poste.")

    # 6.5) SIS Métiers
    st.header("5️⃣ SIS – Métiers recommandés")
    top6 = scorer_metier(profile, referentiel, top_k=6)
    for _, r in top6.iterrows():
        st.markdown(f"**{r['Metier']}** – {int(r['score'])}%")
