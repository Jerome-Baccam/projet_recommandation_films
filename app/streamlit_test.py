import streamlit as st
import pandas as pd
import streamlit_authenticator as stauth
import streamlit_option_menu as som
from pathlib import Path
import ast
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

# On intègre un CSS personnalisé
css_path = Path(__file__).parent / "streamlit.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS file not found at: {css_path}")

# Logo WCS & Ciné en Délire
logo_WCS = Path(__file__).parent / "wcs.jpg"
logo_cine_en_delire = Path(__file__).parent / "cine_en_delire.png"
placeholder = Path(__file__).parent / "placeholder_wcs.png"
# On intègre  le fichier csv et on définit la liste des genres
film_csv = pd.read_csv("films_final_2.csv")
bdd = pd.DataFrame(film_csv)

# Conversion des colonnes de listes pour le système de recommandation
for col in ['genres', 'acteurs', 'directeurs']:
    if col in bdd.columns:
        bdd[col] = bdd[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])

genres = sorted(bdd['genre_1'].unique().tolist())

# Fait avec IA : on parse les acteurs et réalisateurs pour avoir des listes uniques (mais pas parfait, il faudrait nettoyer un peu plus les données)
all_unique_actors = set()
for actors_str in film_csv['acteurs'].dropna():
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
for directors_str in film_csv['directeurs'].dropna():
    director_list_parsed = ast.literal_eval(directors_str)
    if isinstance(director_list_parsed, list):
        for director in director_list_parsed:
            if isinstance(director, str):
                all_unique_directors.add(director.strip())
    elif isinstance(director_list_parsed, str):
        all_unique_directors.add(director_list_parsed.strip())
realisateur_list = ['Tout'] + sorted(list(all_unique_directors))



# SYSTÈME DE RECOMMANDATION KNN 

def create_recommendation_model(df):
    """Créer un modèle KNN basé sur les genres, acteurs et réalisateurs"""
    try:
        # Pour collecter les features
        genres_list = []
        actors_list = []
        directors_list = []
        
        for idx, row in df.iterrows():
            # Genres
            if isinstance(row.get('genres'), list):
                genres_list.append(row['genres'])
            else:
                genres_list.append([])
            
            # Acteurs (top 5)
            if isinstance(row.get('acteurs'), list):
                actors_list.append(row['acteurs'][:5])
            else:
                actors_list.append([])
            
            # Réalisateurs
            if isinstance(row.get('directeurs'), list):
                directors_list.append(row['directeurs'])
            else:
                directors_list.append([])
        
        # Créer des matrices binaires pour chaque feature
        mlb_genres = MultiLabelBinarizer()
        mlb_actors = MultiLabelBinarizer()
        mlb_directors = MultiLabelBinarizer()
        
        genres_matrix = mlb_genres.fit_transform(genres_list)
        actors_matrix = mlb_actors.fit_transform(actors_list)
        directors_matrix = mlb_directors.fit_transform(directors_list)
        
        # Pondération des features pour donner plus d'importance aux réalisateurs
        # Genres x3, Acteurs x2, Réalisateurs x5
        genres_weighted = genres_matrix * 1.75
        actors_weighted = actors_matrix * 1.5
        directors_weighted = directors_matrix * 2
        
        # fusionner les matrices pondérées 
        feature_matrix = np.hstack([genres_weighted, actors_weighted, directors_weighted])
        
        # modèle KNN
        knn_model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
        knn_model.fit(feature_matrix)
        
        return knn_model, feature_matrix
    
    except Exception as e:
        st.error(f"Erreur lors de la création du modèle KNN: {e}")
        return None, None

def get_recommendations(titre, df, model_data, n=5):
    """Obtenir des recommandations avec KNN"""
    if model_data is None or model_data[0] is None:
        # recommandations par genre simple
        try:
            film = df[df['titre'] == titre].iloc[0]
            film_genres = set(film['genres']) if isinstance(film['genres'], list) else set()
            
            # Trouver des films avec des genres similaires
            similar = df[df['genres'].apply(
                lambda x: len(set(x).intersection(film_genres)) > 0 if isinstance(x, list) else False
            )]
            similar = similar[similar['titre'] != titre]
            return similar.head(n)
        except:
            return pd.DataFrame()
    
    try:
        knn_model, feature_matrix = model_data
        
        # Trouver l'index du film
        idx = df[df['titre'] == titre].index[0]
        
        # Obtenir les voisins les plus proches
        distances, indices = knn_model.kneighbors([feature_matrix[idx]], n_neighbors=n+1)
        
        # Exclure le film lui-même (premier résultat)
        movie_indices = indices[0][1:]
        
        return df.iloc[movie_indices]
    
    except Exception as e:
        st.error(f"Erreur dans get_recommendations: {e}")
        return pd.DataFrame()

# Créer le modèle KNN
model_data = create_recommendation_model(bdd)

# Initialiser session_state pour la sélection de film
if 'selected_film' not in st.session_state:
    st.session_state.selected_film = None



# FONCTION POUR AFFICHER UN FILM EN DÉTAIL + RECOMMANDATIONS 


def display_film_detail(film_data):
    """Affiche les détails d'un film avec ses recommandations"""
    
    # Bouton retour
    if st.button("← Retour à la recherche"):
        st.session_state.selected_film = None
        st.rerun()
    
    # Affichage du film
    col1, col2 = st.columns([2, 5])
    
    with col1:
        
        if pd.notna(film_data['poster_url']) and film_data['poster_url'] != 'Inconnu':
            try:
                st.image(film_data['poster_url'], use_container_width=True)
            except:
                st.image(placeholder, use_container_width=True)
                st.info("Affiche non disponible")
        else:
            st.image(placeholder, use_container_width=True)
            st.info("Affiche non disponible")
    
    with col2:
        st.title(film_data['titre'])
        
        # Réalisateurs
        if isinstance(film_data['directeurs'], list) and len(film_data['directeurs']) > 0:
            st.markdown(f"**Réalisé par :** {', '.join(film_data['directeurs'])}")
        
        # Acteurs
        if isinstance(film_data['acteurs'], list) and len(film_data['acteurs']) > 0:
            acteurs_str = ", ".join(film_data['acteurs'][:3])
            st.markdown(f"**En vedette :** {acteurs_str}")
        
        # Genres
        if isinstance(film_data['genres'], list) and len(film_data['genres']) > 0:
            st.markdown(f"**Genres :** {' | '.join(film_data['genres'])}")
        
        st.markdown("---")
        
        # Métriques
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if pd.notna(film_data['année']):
                try:
                    annee_str = str(film_data['année'])
                    if '-' in annee_str:
                        annee = annee_str.split('-')[-1]
                    else:
                        annee = int(float(annee_str))
                    st.metric("Année", annee)
                except:
                    st.metric("Année", film_data['année'])
        with col_b:
            if pd.notna(film_data['votes']):
                st.metric("Note", f"{film_data['votes']:.1f}/10")
        with col_c:
            if pd.notna(film_data['nombre de votes']):
                st.metric("Votes", f"{int(film_data['nombre de votes']):,}")
        
        st.markdown("---")
        
        # Synopsis
        st.markdown("### Synopsis")
        if pd.notna(film_data['résumé']) and film_data['résumé'] != 'Non disponible':
            st.write(film_data['résumé'])
        else:
            st.info("Synopsis non disponible")
    
    st.markdown("---")
    
    # RECOMMANDATIONS
    st.markdown("## Si vous avez aimé ce film, vous aimerez probablement :")
    
    recommendations = get_recommendations(film_data['titre'], bdd, model_data, 5)
    
    if not recommendations.empty:
        cols = st.columns(5)
        for idx, (i, row) in enumerate(recommendations.iterrows()):
            with cols[idx]:
                # Affiche
                if pd.notna(row['poster_url']) and row['poster_url'] != 'Inconnu':
                    try:
                        st.image(row['poster_url'], use_container_width=True)
                    except:
                        st.image(placeholder, use_container_width=True)
                        st.info("Affiche")
                else:
                    st.image(placeholder, use_container_width=True)
                    st.info("Affiche")
                
                # Titre
                st.markdown(f"**{row['titre']}**")
                
                # Note
                if pd.notna(row['votes']):
                    st.caption(f"{row['votes']:.1f}/10")
                
                # Bouton Infos pour voir ce film recommandé
                if st.button("Infos", key=f"rec_{idx}", use_container_width=True):
                    st.session_state.selected_film = row['titre']
                    st.rerun()
    else:
        st.warning("Aucune recommandation disponible pour ce film.")


def page1():
    # VÉRIFIER SI UN FILM EST SÉLECTIONNÉ 
    if st.session_state.selected_film is not None:
        film_data = bdd[bdd['titre'] == st.session_state.selected_film].iloc[0]
        display_film_detail(film_data)
        return  
    
        
    # initie un filtre nul au premier passage
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = bdd.copy()
    # initie la page à 0 au premier passage
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0
    # initie le trigger de reset des filtres (base à False)
    if 'reset_triggered' not in st.session_state:
        st.session_state.reset_triggered = False

    # vérifie si le reset des filtres a été déclenché et on réinitialise tout si c'est le cas
    if st.session_state.reset_triggered:
        st.session_state.filtered_data = bdd.copy()
        st.session_state.page_number = 0
        st.session_state["filtre_mot_clef"] = ""
        st.session_state["filtre_note"] = "Tout"
        st.session_state["filtre_acteur"] = "Tout"
        st.session_state["filtre_popularite"] = "Tout"
        st.session_state["filtre_real"] = "Tout"
        st.session_state["filtre_annee_checkbox"] = False
        st.session_state["filtre_annee"] = 2025
        for i in range(1, 20):
            genre_key = f"genre_{i}"
            st.session_state[genre_key] = False
        st.session_state.reset_triggered = False 
        st.rerun() 

    # Je créé trois colonnes pour centrer le contenu
    lay_gauche, lay_centre, lay_droit = st.columns([1, 20, 1])
    # titre
    with lay_centre:
        # box stylisée pour le titre et les filtres
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-title">Recherche de films A&E</h1>', unsafe_allow_html=True)

        # Filtres container (dans la box stylisée)
        with st.container(border=True):
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
                annee_checkbox = st.checkbox("Filtrer par Année",  key="filtre_annee_checkbox")
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
        # Boutons de navigation (prcédente sur col1/suivante sur col3) --> Je l'ai déplacé ici car casse l'UX sinon
        def boutons_navigation(key): # Attention il faudra à chaque fois rentrer un nouveau numéro pour recréer les boutons
            if total_films > 0 :
                col1, col2, col3 = st.columns([1, 8, 1])
                with col1:
                    if st.button("Page Précédente", key=key, disabled=(st.session_state.page_number == 0)): # Désactivé si on est à la première page
                        st.session_state.page_number -= 1
                        st.rerun() # On recharge la page pour mettre à jour le contenu
                with col3: # J'ai pas voulu me faire chier pour la génération de la deuxième clef
                    if st.button("Page Suivante",key=key*100, disabled=(st.session_state.page_number >= total_pages - 1)): # Désactivé si on est à la dernière page
                        st.session_state.page_number += 1
                        st.rerun() # On recharge la page pour mettre à jour le contenu
                        
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

            boutons_navigation(1)
            
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
                            st.image(placeholder, width=250)
                        else:  # On affiche l'affiche du film
                            poster = st.image(poster_url, width=250)
                        film_title = f"""<div class='film-title'><span>{film['titre']}</span></div>"""
                        st.html(film_title) # Titre en gras
                        
                        # Ajout d'un bouton "Infos" pour chaque film
                        if st.button("DETAILS", key=f"film_{idx}", width=250):
                            st.session_state.selected_film = film['titre']
                            st.rerun()
        
        # On remet les boutons de navigation en bas de la page pour l'user experience
        boutons_navigation(2)
                    
    # On ferme enfin la box principale
    st.markdown('</div>', unsafe_allow_html=True)


def page2():
    st.title("Le Ciné en Délire")
    st.image(logo_cine_en_delire, width=600)

def page3():
    st.title("A&E Tracker par la Wild Comedy Show")
    st.image(logo_WCS, width=600)

pages = [
        st.Page(page1, icon="📽️", title="Recherche A&E", default=True),
        st.Page(page2, icon="🎭", title="Le ciné en délire"),
        st.Page(page3, icon="🤡", title="A&E tracker by WCS"),
    ]
    # Setup de la navigation
st.set_page_config(layout="wide")
current_page = st.navigation(pages=pages, position="hidden")

    # Setup du menu
def menu ():
    Menu_font = """<div class='Menu_test'><span>Menu</span></div>"""
    with st.container(height=90):
        num_cols_menu = max(len(pages) + 1, 5)
        columns_menu = st.columns(num_cols_menu, vertical_alignment="bottom")
        columns_menu[0].html(Menu_font)
        for col, page in zip(columns_menu[1:-1], pages):
            col.page_link(page, icon=page.icon, width="stretch")
            
# On lance le menu puis la page
menu()
current_page.run()
# footer fixe en bas de page


st.markdown('<div class="app-footer">', unsafe_allow_html=True)

footer_col1, footer_col2, footer_col3, footer_col4, footer_col5 = st.columns([1, 1, 3, 1, 1])

with footer_col1:
    if logo_cine_en_delire.exists():
        st.image(logo_cine_en_delire, width=220)
    else:
        st.markdown("<p style='text-align: right; margin: 0; font-size: 20px; color: #c62828; font-weight: bold;'>WCS</p>", unsafe_allow_html=True)

with footer_col2:
    st.write("")

with footer_col3:
    st.markdown("<p style='text-align: center; margin: 0; font-size: 17px; color: #555;'>Application créée par la  Wild Comedy Show  pour Le ciné en délire. Données issus de IMDB, TMDB et AFCAE.<br><br>L'abus de film d'A&E provoque des poussées d'intelligence et un gonflement des chevilles. A consommer avec modération.</p>", unsafe_allow_html=True)

with footer_col4:
    st.write("")
with footer_col5:
    if logo_WCS.exists():
        st.image(logo_WCS, width=220)
    else:
        st.markdown("<p style='text-align: right; margin: 0; font-size: 20px; color: #c62828; font-weight: bold;'>WCS</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)