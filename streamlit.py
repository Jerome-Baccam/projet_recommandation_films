import streamlit as st
import pandas as pd
import streamlit_authenticator as stauth
import streamlit_option_menu as som
from pathlib import Path
import ast

# On intègre un CSS personnalisé
css_path = Path(__file__).parent / "streamlit.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS file not found at: {css_path}")

# On intègre ici le fichier csv et on définit la liste des genres
film_csv = pd.read_csv("films_final.csv")
bdd = pd.DataFrame(film_csv)
genres = sorted(bdd['genre_1'].unique().tolist())

# Fait avec IA : on parse les acteurs et réalisateurs pour avoir des listes uniques (mais pas parfait, il faudrait nettoyer un peu plus les données)
all_unique_actors = set()
for actors_str in bdd['acteurs'].dropna():
    actor_list_parsed = ast.literal_eval(actors_str)
    if isinstance(actor_list_parsed, list):
        for actor in actor_list_parsed:
            if isinstance(actor, str):
                all_unique_actors.add(actor.strip())
    elif isinstance(actor_list_parsed, str):
        all_unique_actors.add(actor_list_parsed.strip())
acteur_list = ['Tout'] + sorted(list(all_unique_actors))
# Même chose pour les réalisateurs
all_unique_directors = set()
for directors_str in bdd['directeurs'].dropna():
    director_list_parsed = ast.literal_eval(directors_str)
    if isinstance(director_list_parsed, list):
        for director in director_list_parsed:
            if isinstance(director, str):
                all_unique_directors.add(director.strip())
    elif isinstance(director_list_parsed, str):
        all_unique_directors.add(director_list_parsed.strip())
realisateur_list = ['Tout'] + sorted(list(all_unique_directors))


# Notre home page (recherche films) 
def page1():
    # on initie un filtre nul au premier passage
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = bdd.copy()
    # on initie la page à 0 au premier passage
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0
    # on initie le trigger de reset des filtres (base à False)
    if 'reset_triggered' not in st.session_state:
        st.session_state.reset_triggered = False

    # On vérifie si le reset des filtres a été déclenché et on réinitialise tout si c'est le cas
    if st.session_state.reset_triggered:
        st.session_state.filtered_data = bdd.copy()
        st.session_state.page_number = 0
        st.session_state["filter_keyword"] = ""
        st.session_state["filter_note"] = "Any"
        st.session_state["filter_actor"] = "Any"
        st.session_state["filter_popularity"] = "Any"
        st.session_state["filter_director"] = "Any"
        st.session_state["filter_year_checkbox"] = False
        st.session_state["filter_year"] = 2025
        for i in range(1, 20):
            genre_key = f"genre_{i}"
            st.session_state[genre_key] = False
        st.session_state.reset_triggered = False # On réinitialise le bouton reset
        st.rerun() # On fait un rerun pour appliquer les changements

    # Je créé trois colonnes pour centrer le contenu
    lay_gauche, lay_centre, lay_droit = st.columns([1, 20, 1])
    # titre
    with lay_centre:
        # box stylisée pour le titre et les filtres
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-title">Recherche de films A&E</h1>', unsafe_allow_html=True)

        # Filtres container (dans la box stylisée)
        with st.container():
            st.subheader("Filtres")

            # colonnes des filtres principaux
            # A revoir car je verrai bien un order by pour la popularité ou les notes
            but_gauche, but_centre, but_droit = st.columns(3)
            with but_gauche:
                mot_clef = st.text_input("Mot-clef", key="filtre_mot_clef")
                note = st.selectbox("Note", options=["Tout", ">= 5", ">= 7", ">= 9"], key="filtre_note")
            with but_centre:
                actor = st.selectbox("Acteur", options=acteur_list, key="filtre_acteur")
                popularite = st.selectbox("Popularité", options=["Tout", "Basse", "Moyenne", "Haute"], key="filtre_popularite")
            with but_droit:
                director = st.selectbox("Réalisateur", options=realisateur_list, key="filtre_real")
                annee_checkbox = st.checkbox("Filtrer par Année", value=False, key="filtre_annee_checkbox")
                année = st.number_input("Année", min_value=1900, max_value=2025, value=2025, step=1, key="filtre_annee")

            st.write("Genre")
            # Genres comme toggle button dans des petites colonnes
            # J'aurais pu faire une boucle mais j'ai préféré faire à la main car l'IA me générait automatiquement le contenu après en avoir rentré quelques uns
            # On peut faire une boucle plus tard si on veut optimiser le code
            but_a, but_b, but_c, but_d, but_e = st.columns(5)
            with but_a:
                st.checkbox(f"{genres[1]}", key="genre_1")
                st.checkbox(f"{genres[6]}", key="genre_6")
                st.checkbox(f"{genres[11]}", key="genre_11")
                st.checkbox(f"{genres[16]}", key="genre_16")
            with but_b:
                st.checkbox(f"{genres[2]}", key="genre_2")
                st.checkbox(f"{genres[7]}", key="genre_7")
                st.checkbox(f"{genres[12]}", key="genre_12")
                st.checkbox(f"{genres[17]}", key="genre_17")
            with but_c:
                st.checkbox(f"{genres[3]}", key="genre_3")
                st.checkbox(f"{genres[8]}", key="genre_8")
                st.checkbox(f"{genres[13]}", key="genre_13")
                st.checkbox(f"{genres[18]}", key="genre_18")
            with but_d:
                st.checkbox(f"{genres[4]}", key="genre_4")
                st.checkbox(f"{genres[9]}", key="genre_9")
                st.checkbox(f"{genres[14]}", key="genre_14")
                st.checkbox(f"{genres[19]}", key="genre_19")
            with but_e:
                st.checkbox(f"{genres[5]}", key="genre_5")
                st.checkbox(f"{genres[10]}", key="genre_10")
                st.checkbox(f"{genres[15]}", key="genre_15")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1: # Bouton de filtrage
            if st.button("Filtrer"):
                # On créé un DF temporaire pour appliquer les filtres
                temp_bdd_filtre = bdd.copy()

                # Application des filtres un par un
                if mot_clef: # On met du lowercase pour éviter les soucis de casse et on cherche dans le titre et le résumé
                    mot_clef_lower = mot_clef.lower()
                    condition_titre = temp_bdd_filtre["titre"].astype(str).str.lower().str.contains(mot_clef_lower, na=False, regex=False)
                    condition_resume = temp_bdd_filtre["résumé"].astype(str).str.lower().str.contains(mot_clef_lower, na=False, regex=False)
                    temp_bdd_filtre = temp_bdd_filtre[condition_titre | condition_resume]
                if actor != "Tout":
                    temp_bdd_filtre = temp_bdd_filtre[temp_bdd_filtre["acteurs"].astype(str).str.contains(actor, case=False, na=False, regex=False)]
                if director != "Tout":
                    temp_bdd_filtre = temp_bdd_filtre[temp_bdd_filtre["directeurs"].astype(str).str.contains(director, case=False, na=False, regex=False)]
                if annee_checkbox:
                    temp_bdd_filtre = temp_bdd_filtre[temp_bdd_filtre["année"].astype(str).str.slice(-4).astype(int) == année]
                if note != "Tout":
                    seuil_note = int(note.split(">= ")[1]) # on prend juste le nombre après >=
                    temp_bdd_filtre = temp_bdd_filtre[temp_bdd_filtre["votes"] >= seuil_note]
                if popularite != "Tout": # Popularité est catégorisé en Basse (<5), Moyenne (5-10), Haute (>=10) (on peut changer ces seuils si vous voulez)
                    if popularite == "Basse":
                        temp_bdd_filtre = temp_bdd_filtre[temp_bdd_filtre["popularité"] < 5]
                    elif popularite == "Moyenne":
                        temp_bdd_filtre = temp_bdd_filtre[(temp_bdd_filtre["popularité"] >= 5) & (temp_bdd_filtre["popularité"] < 10)]
                    elif popularite == "Haute":
                        temp_bdd_filtre = temp_bdd_filtre[temp_bdd_filtre["popularité"] >= 10]
                # Genres
                for i in range(1, 20):
                    genre_key = f"genre_{i}"
                    if st.session_state.get(genre_key):
                        genre_value = genres[i]
                        temp_bdd_filtre = temp_bdd_filtre[temp_bdd_filtre["genres"].astype(str).str.contains(genre_value, case=False, na=False, regex=False)]
                # On stocke le DF filtré dans l'état de session
                st.session_state.filtered_data = temp_bdd_filtre.copy()
                st.session_state.page_number = 0 # On revient à la première page
                st.rerun() # On recharge la page pour afficher les résultats filtrés
        # Bouton de réinitialisation des filtres
        with filter_col2:
            if st.button("Réinitialiser les filtres"):
                st.session_state.reset_triggered = True
                st.rerun() # On recharge la page pour appliquer le reset

        st.subheader("Résultats de la recherche")

        # On fait la pagination des résultats avec 20 films par page et 5 par ligne
        films_par_page = 20
        total_films = len(st.session_state.filtered_data)
        total_pages = total_films // films_par_page
        if total_films % films_par_page != 0:
            total_pages += 1 # On ajoute une page supplémentaire pour les restants
        if total_films == 0: # Si aucun film ne correspond aux critères
            st.write("Aucun film ne correspond à vos critères de recherche.")
        else:
            start_idx = st.session_state.page_number * films_par_page # calcul des indices de début de la page courante (0*20, 1*20, etc)
            end_idx = min((st.session_state.page_number + 1) * films_par_page, total_films) # pour ne pas dépasser le total
            display_films = st.session_state.filtered_data.iloc[start_idx:end_idx] # Films à afficher sur la page courante

            st.write(f"Nombre de résultats : {total_films}")

            # On affiche le numéro de la page actuelle et le total des pages si résultat > 0
            if total_pages > 0:
                st.write(f"Page {st.session_state.page_number + 1} sur {total_pages}")

            # Boutons de navigation (prcédente sur col1/suivante sur col3)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col1:
                if st.button("Page Précédente", disabled=(st.session_state.page_number == 0)): # Désactivé si on est à la première page
                    st.session_state.page_number -= 1
                    st.rerun() # On recharge la page pour mettre à jour le contenu
            with col3:
                if st.button("Page Suivante", disabled=(st.session_state.page_number >= total_pages - 1)): # Désactivé si on est à la dernière page
                    st.session_state.page_number += 1
                    st.rerun() # On recharge la page pour mettre à jour le contenu

            # Pagination des films (5 par ligne)
            films_par_ligne = 5
            for i in range(0, len(display_films), films_par_ligne):
                ligne_films = display_films.iloc[i : i + films_par_ligne]
                cols = st.columns(films_par_ligne)
                # iterrow nous permet d'itérer sur les lignes d'un DF
                for col_idx, (idx, film) in enumerate(ligne_films.iterrows()): # On itère sur les films de la ligne dans notre filtered_data[indexes]
                    with cols[col_idx]: # pour chacune des 5 colonnes de la ligne on affiche un film
                        poster_url = film['poster_url']
                        # On affiche un placeholder si l'URL est invalide
                        if pd.isna(poster_url) or poster_url == "Inconnu" :
                            st.image("http://via.placeholder.com/150", width=150)
                        else:  # On affiche l'affiche du film
                            st.image(poster_url, width=150)
                        st.markdown(f"**{film['titre']}**") # Titre en gras

    # On ferme enfin la box principale
    st.markdown('</div>', unsafe_allow_html=True)


def page2():
    st.write("Un petit test pour du multipage app")
    st.image("https://media.makeameme.org/created/impressive-2y23ct.jpg", width=600)

def page3():
    st.write("Un autre petit test pour du multipage app")
    st.image("https://i.pinimg.com/564x/56/b9/a9/56b9a962f481a4212bce3f82b151433d.jpg", width=600)

pages = [ # On dirait que la taille des box est fixe, et en fonction du zoom du navigateur ça peut couper le texte
        st.Page(page1, icon="📽️", title="Recherche A&E"),
        st.Page(page2, icon="🎭", title="Le ciné en délire"),
        st.Page(page3, icon="🤡", title="A&E tracker by WCS"),
    ]
    # Setup de la navigation
st.set_page_config(layout="wide")
current_page = st.navigation(pages=pages, position="hidden")

    # Setup du menu
num_cols_menu = max(len(pages) + 1, 8) # Au moins 8 colonnes pour que le menu soit espacé, ca permettra aussi d'ajouter des boutons plus tard si besoin
columns_menu = st.columns(num_cols_menu, vertical_alignment="bottom")
columns_menu[0].write("**Menu**")
for col, page in zip(columns_menu[1:-1], pages): # On évite la première et la dernière colonne pour le titre "Menu" et le bouton logout
    col.page_link(page, icon=page.icon) # On crée un lien pour chaque page dans une colonne avec son icône
current_page.run()
