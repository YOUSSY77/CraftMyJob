def filter_by_location(offers: list, loc_norm: str) -> list:
    """
    Ne conserve que les annonces dont le 'lieuTravail_libelle'
    contient la chaîne loc_norm (ex : '75' ou 'Paris').
    """
    out = []
    loc_norm = loc_norm.lower()
    for o in offers:
        lib = o.get('lieuTravail_libelle', '').lower()
        if loc_norm in lib:
            out.append(o)
    return out


# … plus bas, dans votre bloc “Top 30” …

st.header(f"4️⃣ Top 30 offres pour '{job_title}'")

# 1) Collecte brute
keywords   = job_title
all_offres = []
for loc in sel:
    loc_norm = normalize_location(loc)       # ex. "75" ou "Paris"
    offs = search_offres(token, keywords, loc_norm, limit=30)
    all_offres.extend(offs)

# 2) Filtre uniquement par type de contrat
all_offres = [o for o in all_offres if o.get('typeContrat','') in contract]

# 3) Filtre strict par nom de lieu
all_offres = filter_by_location(all_offres, loc_norm)

# 4) Déduplication + scoring
seen = {}
for o in all_offres:
    url = o.get('url','') or o.get('contact',{}).get('urlPostulation','') or o.get('contact',{}).get('urlOrigine','')
    if url and url not in seen:
        seen[url] = o
candidates = list(seen.values())

for o in candidates:
    title_score = fuzz.token_set_ratio(o.get('intitule',''), job_title)
    desc_score  = fuzz.token_set_ratio(o.get('description_extrait','')[:200], missions)
    o['score_match'] = 0.7 * title_score + 0.3 * desc_score

# 5) Affichage du Top 30
top30 = sorted(candidates, key=lambda x: x['score_match'], reverse=True)[:30]

if top30:
    for o in top30:
        st.markdown(
            f"**{o['intitule']}** ({o['typeContrat']}) – "
            f"{o['lieuTravail_libelle']} (_Publié {o['dateCreation'][:10]}_)  \n"
            f"Score matching : **{int(o['score_match'])}%**  \n"
            f"<span class='offer-link'><a href='{o['url']}' target='_blank'>Voir l'offre</a></span>\n---",
            unsafe_allow_html=True
        )
else:
    st.info("Aucune offre pertinente trouvée pour ce poste.")
