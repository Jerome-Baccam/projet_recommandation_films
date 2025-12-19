import streamlit as st
import pandas as pd
from pathlib import Path
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
import requests
from io import BytesIO

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
banner = Path(__file__).parent / "banniere.png"
crew = Path(__file__).parent / "la_team.png"
cliente = Path(__file__).parent / "cliente.png"
# On intègre  le fichier csv et on définit la liste des genres


def load_data():
    data_path = Path(__file__).parent / "filmsfinal.csv"
    return pd.read_csv(data_path)
film_csv = load_data()


def transfo_bdd():
    bdd = pd.DataFrame(film_csv)
    bdd['année'] = pd.to_datetime(bdd['année'], format='%d-%m-%Y').dt.year
    bdd = bdd.sort_values(by='titre', ascending=True)
    return bdd
bdd = transfo_bdd()

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

#resizing des affiches de films pour qu'ils aient tous la même taille
@st.cache_data
def poster_sizing(poster_link):
    r = requests.get(poster_link)
    poster_resized = Image.open(BytesIO(r.content))
    poster_resized = poster_resized.resize((750, 1125)) # il faut que ce soit un multiple de 250/375 pour que ça s'affiche bien dans les colonnes streamlit
    return poster_resized

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
        genres_weighted = genres_matrix * 2
        actors_weighted = actors_matrix * 1.5
        directors_weighted = directors_matrix * 3
        
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
    st.markdown("""<p style='font-size:30px'><br><br>""", unsafe_allow_html=True)
    
    with st.container(border=False, width='stretch', horizontal_alignment="center", vertical_alignment="center"):
        with st.container(border=False, width=1485, horizontal_alignment="center", vertical_alignment="center"):
            # Affichage du film
            col1, col2 = st.columns([2, 5])
            
            with col1:
                
                if pd.notna(film_data['poster_url']) and film_data['poster_url'] != 'Inconnu':
                    try:
                        st.image(poster_sizing(film_data['poster_url']), width='stretch')
                    except:
                        st.image(placeholder, width='stretch')
                        st.info("Affiche non disponible")
                else:
                    st.image(placeholder, width='stretch')
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
                    st.write(f"""<div class='synopsis'><span>{film_data['résumé']}</span></div>""", unsafe_allow_html=True)
                else:
                    st.info("Synopsis non disponible")
            
            st.markdown("---")
            
            # RECOMMANDATIONS
            st.markdown("## Si vous avez aimé ce film, vous aimerez probablement :")
            
            recommendations = get_recommendations(film_data['titre'], bdd, model_data, 6)
            
            if not recommendations.empty:
                cols = st.columns(6)
                for idx, (i, row) in enumerate(recommendations.iterrows()):
                    with cols[idx]:
                        # Affiche
                        if pd.notna(row['poster_url']) and row['poster_url'] != 'Inconnu':
                            try:
                                st.image(poster_sizing(row['poster_url']), width='stretch')
                            except:
                                st.image(placeholder, width='stretch')
                                st.info("Affiche")
                        else:
                            st.image(placeholder, width='stretch')
                        st.markdown(f"""<div class='film-title'><span>{row['titre']}</span</div>""", unsafe_allow_html=True)
                        
                        # Note
                        if pd.notna(row['votes']):
                            st.caption(f"{row['votes']:.1f}/10")
                        
                        # Bouton Infos pour voir ce film recommandé
                        if st.button("Infos", key=f"rec_{idx}", width='stretch'):
                            st.session_state.selected_film = row['titre']
                            st.rerun()
            else:
                st.warning("Aucune recommandation disponible pour ce film.")
        # Bouton retour
    if st.button("← Retour à la recherche", key="back_to_search"):
        st.session_state.selected_film = None
        st.rerun()


def page1():
    st.markdown("""<p style='font-size:30px'><br><br>""", unsafe_allow_html=True)
    # VÉRIFIER SI UN FILM EST SÉLECTIONNÉ 
    if st.session_state.selected_film is not None:
        film_data = bdd[bdd['titre'] == st.session_state.selected_film].iloc[0]
        display_film_detail(film_data)
        return  

    st.session_state.setdefault('filtered_data', bdd.copy())
    st.session_state.setdefault('page_number', 0)
    st.session_state.setdefault('reset_triggered', False)
    
    # vérifie si le reset des filtres a été déclenché et on réinitialise tout si c'est le cas
    if st.session_state.reset_triggered == True:
        st.session_state.filtered_data = bdd.copy()
        st.session_state.page_number = 0
        st.session_state["filtre_mot_clef"] = ""
        st.session_state["filtre_acteur"] = "Tout"
        st.session_state["filtre_real"] = "Tout"
        st.session_state["filtre_periode"] = (1897, 2025)
        for i in range(1, 20):
            genre_key = f"genre_{i}"
            st.session_state[genre_key] = False
        st.session_state['sort_by'] = 'Alphabétique'
        st.session_state['order_by'] = 'Croissant'
        st.session_state.reset_triggered = False 
        st.rerun() 

                # CONTENU DE LA PAGE PRINCIPALE
    # Bannière en haut
    with st.container(border=False, width='stretch', horizontal_alignment="center", vertical_alignment="center"):
        with st.container(border=False, width=1485, horizontal_alignment="center", vertical_alignment="center"):
            with st.container(vertical_alignment="center", height="stretch", border=False):
                st.image(banner, width='stretch')  

            # Je créé trois colonnes pour centrer le contenu
            lay_gauche, lay_centre, lay_droit = st.columns([1, 20, 1])
            # titre
            with lay_centre:
                st.markdown("<h1 class='main-title'>Recherche de films d'Art & Essai</h1>", unsafe_allow_html=True)
                # Filtres container (dans la box stylisée)
                with st.container(border=True):
                    st.subheader("Filtres")
                    # colonnes des filtres principaux
                    but_gauche, but_centre, but_droit = st.columns(3)
                    with but_gauche:
                        mot_clef = st.text_input("Mot-clef", key="filtre_mot_clef")
                        st.write("<br>", unsafe_allow_html=True)
                    with but_centre:
                        actor = st.selectbox("Acteur", options=acteur_list, key="filtre_acteur")
                        st.write("<br> ", unsafe_allow_html=True)
                    with but_droit:
                        director = st.selectbox("Réalisateur", options=realisateur_list, key="filtre_real")
                        st.write("<br> ", unsafe_allow_html=True)
                    
                    # Filtre période avec un slider
                    col1, col2, col3 = st.columns([1, 5, 1])
                    with col2:
                        date_sld = st.slider("**Sélectionnez une période**", 1897, 2025, (1897, 2025), key="filtre_periode")
                        st.write("Période choisie :",date_sld)
                    st.write("<br> ", unsafe_allow_html=True)
                    st.write("**Genres**")
                    
                    # Genres comme toggle button dans des petites colonnes
                    but_0, but_a, but_b, but_c, but_d, but_e = st.columns([2.5,5,5,5,5,5])
                    # On fait une liste des colonnes pour itérer dessus en ommettant la première colonne qui nous sert uniquement pour la pagination (but_0)
                    colonnes_genres = [but_a, but_b, but_c, but_d, but_e]
                    # On itère sur les genres à partir du deuxième (le premier est "Tout")
                    for i, genre_film in enumerate(genres[1:]):
                        # On répartit les genres dans les 5 colonnes de gauche à droite plutôt que de haut en bas
                        with colonnes_genres[i % len(colonnes_genres)]:
                            st.checkbox(f"{genre_film}", key=f"genre_{i+1}")
                    st.write("<br> ", unsafe_allow_html=True)
                    
                    # Option de tri
                    with st.container(horizontal=True):
                        tri1, tri2, tri3 = st.columns([5, 5, 10])
                        with tri1:
                            tri = st.selectbox('Trier par :', options=['Alphabétique', 'Année', 'Note', 'Popularité'], key='sort_by')
                        with tri2:
                            order = st.selectbox('Ordre :', options=['Croissant', 'Décroissant'], key='order_by')
                    st.write("<br><br>", unsafe_allow_html=True)
                    
                    # Boutons de filtrage et réinitialisation
                    filter_col1, filer, filter_col2 = st.columns([1,4,1])
                    with filter_col1: # Bouton de filtrage
                        if st.button("Filtrer", width='stretch', key="filter_but"):
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
                            if date_sld:
                                temp_bdd_filtre = temp_bdd_filtre[
                                    (temp_bdd_filtre["année"] >= date_sld[0]) &
                                    (temp_bdd_filtre["année"] <= date_sld[1])
                                ]
                            # Genres
                            for i in range(1, 20):
                                genre_key = f"genre_{i}"
                                if st.session_state.get(genre_key):
                                    genre_value = genres[i]
                                    temp_bdd_filtre = temp_bdd_filtre[temp_bdd_filtre["genres"].astype(str).str.contains(genre_value, case=False, na=False, regex=False)]
                            if tri == 'Alphabétique':
                                sort_column = 'titre'
                            elif tri == 'Année':
                                sort_column = 'année'
                            elif tri == 'Note':
                                sort_column = 'votes'
                            elif tri == 'Popularité':
                                sort_column = 'nombre de votes'
                            temp_bdd_filtre = temp_bdd_filtre.sort_values(by=sort_column, ascending=True)
                            if order =='Décroissant':
                                temp_bdd_filtre = temp_bdd_filtre.sort_values(by=sort_column, ascending=False)
                                    
                            # On stocke le DF filtré dans l'état de session
                            st.session_state.filtered_data = temp_bdd_filtre.copy()
                            st.session_state.page_number = 0 # On revient à la première page
                            st.rerun() # On recharge la page pour afficher les résultats filtrés
                    # Bouton de réinitialisation des filtres
                    with filter_col2:
                        if st.button("Réinitialiser les filtres", width='stretch', key="reset_but"):
                            st.session_state.reset_triggered = True
                            st.rerun() # On recharge la page pour appliquer le reset

                st.subheader("Résultats de la recherche")

                # On fait la pagination des résultats avec 20 films par page et 5 par ligne
                films_par_page = 30
                total_films = len(st.session_state.filtered_data)
                total_pages = total_films // films_par_page
                if total_films % films_par_page != 0:
                    total_pages += 1 # On ajoute une page supplémentaire pour les restants
                # Boutons de navigation (prcédente sur col1/suivante sur col3) --> Je l'ai déplacé ici car casse l'UX sinon
                
                def boutons_navigation(key_numb): # Attention il faudra à chaque fois rentrer un nouveau numéro pour recréer les boutons
                    if total_films > 0 :
                        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns([4,1,3,1,1,1,1,1,3,1,4])
                        with col2: # On va à la première page
                            if st.button("**<<**", key=f"back0_{key_numb}", disabled=(st.session_state.page_number == 0), width='stretch'):
                                st.session_state.page_number = 0 # On revient à la première page
                                st.rerun() # On recharge la page pour mettre à jour le contenu
                        with col3: # On revient à la page précédente
                            if st.button("Page Précédente", key=f"prev_{key_numb}", disabled=(st.session_state.page_number == 0), width='stretch'): # Désactivé si on est à la première page
                                st.session_state.page_number -= 1 
                                st.rerun() 
                        with col4: # On revient de deux pages
                            if st.session_state.page_number >= 2:
                                if st.button(f"{st.session_state.page_number-1}", key=f"twoback_{key_numb}", disabled=(st.session_state.page_number < 2), width='stretch'):
                                    st.session_state.page_number -= 2 
                                    st.rerun()
                        with col5: # On revient d'une page
                            if st.session_state.page_number >= 1:
                                if st.button(f"{st.session_state.page_number}", key=f"oneback_{key_numb}", disabled=(st.session_state.page_number < 1), width='stretch'):
                                    st.session_state.page_number -= 1
                                    st.rerun()
                        with col6: # Page actuelle
                            st.button(f"**{st.session_state.page_number + 1}**", key=f"page_actuelle_{key_numb}", disabled=True, width='stretch') # Page actuelle en gras
                        with col7: # On avance d'une page
                            if st.session_state.page_number + 1 < total_pages:
                                if st.button(f"{st.session_state.page_number+2}", key=f"oneforward_{key_numb}", disabled=(st.session_state.page_number + 1 >= total_pages), width='stretch'):
                                    st.session_state.page_number += 1 
                                    st.rerun()
                        with col8: # On avance de deux pages
                            if st.session_state.page_number + 2 < total_pages:
                                if st.button(f"{st.session_state.page_number+3}", key=f"twoforward_{key_numb}", disabled=(st.session_state.page_number + 2 >= total_pages), width='stretch'):
                                    st.session_state.page_number += 2
                                    st.rerun()
                        with col9: # On va à la page suivante
                            if st.button(f"Page Suivante",key=f"next_{key_numb}", disabled=(st.session_state.page_number >= total_pages - 1), width='stretch'): # Désactivé si on est à la dernière page
                                st.session_state.page_number += 1
                                st.rerun() 
                        with col10: # On va à la dernière page
                            if st.button("**>>**", key=f"forwardend_{key_numb}", disabled=(st.session_state.page_number == -1 or total_pages==1), width='stretch'):
                                st.session_state.page_number = total_pages - 1
                                st.rerun()
                        
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
                    films_par_ligne = 6
                    
                    for i in range(0, len(display_films), films_par_ligne):
                        ligne_films = display_films.iloc[i : i + films_par_ligne]
                        cols = st.columns(films_par_ligne)
                        # iterrow nous permet d'itérer sur les lignes d'un DF
                        for col_idx, (idx, film) in enumerate(ligne_films.iterrows()): # On itère sur les films de la ligne dans notre filtered_data[indexes]
                            with cols[col_idx]: # pour chacune des 5 colonnes de la ligne on affiche un film
                                poster_url = film['poster_url']
                                # On affiche un placeholder si l'URL est invalide
                                if pd.isna(poster_url) or poster_url == "Inconnu" :
                                    st.image(placeholder, width='stretch')
                                else:  # On affiche l'affiche du film
                                    poster = st.image(poster_sizing(poster_url), width='stretch')

                                film_title = f"""<div class='film-title'><span>{film['titre']}</span></div>"""
                                st.html(film_title) # Titre en gras
                                
                                # Ajout d'un bouton "Infos" pour chaque film
                                if st.button("DETAILS", key=f"film_{idx}", width='stretch'):
                                    st.session_state.selected_film = film['titre']
                                    st.rerun()
                
                # On remet les boutons de navigation en bas de la page pour l'user experience
                boutons_navigation(2)
                            
            # On ferme enfin la box principale
            st.markdown('</div>', unsafe_allow_html=True)

def statistiques():
    # On fait une liste des graphiques disponibles
    list_graphs = ["Genres les plus représentés", 
                    "Répartition des genres",
                    "Films les plus populaires", 
                    "Acteurs les plus populaires", 
                    "Distribution des notes des films", 
                    "Distribution des notes par genre",
                    "Evolution de la production de films", 
                    "Relation popularité-notes",
                    "Matrice de corrélation"
                    ]
    # Graph 1
    def genre_rep():
        tous_les_genres = pd.concat([bdd['genre_1'], bdd['genre_2'], bdd['genre_3']])
        tous_les_genres = tous_les_genres.dropna()
        tous_les_genres = tous_les_genres.astype(str).str.strip()
        tous_les_genres = tous_les_genres[tous_les_genres.str.len() > 1]
        compte_genres = tous_les_genres.value_counts().head(10)
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(x=compte_genres.values, y=compte_genres.index, palette='viridis')
        plt.title('Top 10 des Genres')
        plt.xlabel('Nombre de films')
        plt.tight_layout() # Pour ne pas couper le texte à gauche
        return fig
    
    # Graph 2
        # fonction importée depuis stackoverflow pour rendre les labels qui se chevauchent plus lisibles
    def fix_labels(mylabels, tooclose=0.1, sepfactor=2):
        vecs = np.zeros((len(mylabels), len(mylabels), 2))
        dists = np.zeros((len(mylabels), len(mylabels)))
        for i in range(0, len(mylabels)-1):
            for j in range(i+1, len(mylabels)):
                a = np.array(mylabels[i].get_position())
                b = np.array(mylabels[j].get_position())
                dists[i,j] = np.linalg.norm(a-b)
                vecs[i,j,:] = a-b
                if dists[i,j] < tooclose:
                    mylabels[i].set_x(a[0] + sepfactor*vecs[i,j,0])
                    mylabels[i].set_y(a[1] + sepfactor*vecs[i,j,1])
                    mylabels[j].set_x(b[0] - sepfactor*vecs[i,j,0])
                    mylabels[j].set_y(b[1] - sepfactor*vecs[i,j,1])
    
    def repart_genre():
        tous_les_genres = bdd[['genre_1', 'genre_2', 'genre_3']].melt(value_name='Genre')
        tous_les_genres['Genre'] = tous_les_genres['Genre'].astype(str).str.strip()
        imposteurs = ['nan', 'NaN', '', ' ', 'None', '\\N', '-']
        tous_les_genres = tous_les_genres[~tous_les_genres['Genre'].isin(imposteurs)]
        compte_genres = tous_les_genres['Genre'].value_counts()
        top_10_genres = compte_genres[:10]
        autres = compte_genres[10:].sum()
        if autres > 0:
            top_10_genres = top_10_genres.copy()
            top_10_genres['Autres'] = autres
        # Create pie chart with legend to the right to avoid label overlap
        fig, ax = plt.subplots(figsize=(12, 6))
        wedges, labels, autopct = ax.pie(top_10_genres,
                                        labels=top_10_genres.index,
                                        autopct='%1.1f%%',
                                        startangle=140,
                                        colors=plt.cm.Set3.colors,
                                        wedgeprops={'edgecolor': 'white'})
        fix_labels(autopct, sepfactor=3)
        fix_labels(labels, sepfactor=3)
        ax.set_title('Répartition des Genres', fontsize=16)
        ax.axis('equal')  # keep as circle
        plt.subplots_adjust(right=0.75)
        # ensure percentages are legible

        return fig

    # Graph 3
    def films_pop():
        df_plus_votes = bdd.sort_values(by='nombre de votes', ascending=False)
        top_10_films = df_plus_votes.head(10)
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(data=top_10_films, x='nombre de votes', y='titre', palette='viridis')
        plt.title('Top 10 des Films les plus populaires (par Votes)')
        plt.ticklabel_format(style='scientific')
        plt.xlabel('Nombre de votes')
        plt.ylabel('Titre du film')
        plt.tight_layout()
        return fig

    # Graph 4
    def acteurs_pop():
        df_acteurs = bdd[['acteurs', 'nombre de votes']].copy()
        df_acteurs['nombre de votes'] = pd.to_numeric(df_acteurs['nombre de votes'], errors='coerce').fillna(0)
        df_acteurs_explose = df_acteurs.explode('acteurs')
        df_acteurs_explose['acteurs'] = df_acteurs_explose['acteurs'].astype(str).str.strip()
        df_acteurs_explose = df_acteurs_explose[df_acteurs_explose['acteurs'].astype(bool)]
        top_acteurs = df_acteurs_explose.groupby('acteurs')['nombre de votes'].sum().sort_values(ascending=False).head(10)
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=top_acteurs.values, y=top_acteurs.index, palette='rocket')
        plt.title('Top 10 des Acteurs cumulant le plus de votes (Carrière)')
        plt.xlabel('Nombre total de votes cumulés')
        plt.ylabel('Acteur')
        plt.tight_layout()
        return fig
    
    # Graph 5
    def distrib_notes():
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(data=bdd, x='votes', bins=20, kde=True, color='skyblue')
        plt.title('Distribution des Notes (Est-ce que les gens sont sévères ?)')
        plt.xlabel('Note moyenne')
        plt.ylabel('Nombre de films')
        return fig
    
    # Graph 6
    def distrib_notes_genre():
        df_graph = bdd[['votes', 'genre_1', 'genre_2', 'genre_3']]
        df_melted = df_graph.melt(id_vars=['votes'],
                                value_vars=['genre_1', 'genre_2', 'genre_3'],
                                value_name='Genre_Global')
        df_melted['Genre_Global'] = df_melted['Genre_Global'].astype(str).str.strip()
        imposteurs = ['nan', 'NaN', '', ' ', 'None', '\\N', '-']
        df_melted = df_melted[~df_melted['Genre_Global'].isin(imposteurs)]
        fig = plt.figure(figsize=(16, 8))
        ordre_alphabetique = sorted(df_melted['Genre_Global'].unique())

        sns.boxplot(x='Genre_Global', y='votes', data=df_melted,
                    order=ordre_alphabetique,
                    palette="Set3")
        plt.title('Distribution des Notes par Genre', fontsize=16)
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Note (Votes)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        return fig
    
    # Graph 7
    def evo_prod_films():
        bdd2 = bdd.copy()
        bdd2['année_clean'] = bdd2['année'].astype(str).str.extract(r'(\d{4})')
        bdd2['année_clean'] = pd.to_numeric(bdd2['année_clean'], errors='coerce')
        films_par_annee = bdd2.groupby('année_clean').size()
        films_par_annee.index = films_par_annee.index.astype(int)
        films_par_annee = films_par_annee.sort_index()
        moyenne_mobile = films_par_annee.rolling(window=5).mean()
        fig = plt.figure(figsize=(12, 6))
        moyenne_mobile.plot(kind='line', color='red', linewidth=3, label='Tendance Moyenne (sur 5 ans)')
        plt.title('Évolution de la Production de Films A&E (Chronologique et Lissé)')
        plt.xlabel('Année')
        plt.ylabel('Nombre de films sortis')
        plt.legend()
        plt.grid(True)
        return fig
    
    # Graph 8
    def rel_pop_notes():
        fig = plt.figure(figsize=(12, 6))
        sns.scatterplot(x='nombre de votes',
                        y='votes',
                        data=bdd,
                        alpha=0.6,
                        color='darkblue')
        plt.title('Relation entre Popularité et Qualité des films')
        plt.xlabel('Nombre de votes (Popularité)')
        plt.ylabel('Note moyenne (Qualité)')
        plt.xscale('log')
        plt.grid(True, linestyle='--', alpha=0.3)
        return fig
    
    # Graph 9
    def matrix():
        bdd2 = bdd.copy()
        bdd2['année_num'] = bdd2['année'].astype(str).str.extract(r'(\d{4})')
        bdd2['année_num'] = pd.to_numeric(bdd2['année_num'], errors='coerce')
        chiffres = bdd2[['temps', 'votes', 'nombre de votes', 'année_num']].dropna()
        fig = plt.figure(figsize=(10, 8))
        matrice_corr = chiffres.corr()
        sns.heatmap(matrice_corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matrice de Corrélation')
        return fig
    
    # Début de la pagination
    st.markdown("""<p style='font-size:30px'><br><br>""", unsafe_allow_html=True)
    with st.container(border=False, width='stretch', horizontal_alignment="center", vertical_alignment="center"):
        with st.container(border=False, width=1485, horizontal_alignment="center", vertical_alignment="center"):
            # Titre H1
            st.markdown("""<h1 class='page2-title' style='text-align: center;'>Statistiques de la base de données</h1>""", unsafe_allow_html=True)
            st.subheader("Visualisation de la base de donnée des films d'Art & d'Essai")
            # Colonnes pour la mise en page
            col1, col2 = st.columns([1,4])
            with col1:
                # Liste déroulante des graphs disponibles
                box = st.selectbox("Quel graphique veux-tu visionner ?", options=list_graphs)
            # Affichage des graphs en fonction de l'option choisie
            coll1, coll2, coll3 = st.columns([1,4,1])
            with coll2:
                if box == "Genres les plus représentés":
                    st.pyplot(genre_rep(), width="content")
                elif box == "Répartition des genres":
                    st.pyplot(repart_genre(), width=700)
                elif box == "Films les plus populaires":
                    st.pyplot(films_pop(), width="content")
                elif box == "Acteurs les plus populaires":
                    st.pyplot(acteurs_pop(), width="content")
                elif box == "Distribution des notes des films":
                    st.pyplot(distrib_notes(), width="content")
                elif box == "Distribution des notes par genre":
                    st.pyplot(distrib_notes_genre(), width="content")
                elif box == "Evolution de la production de films":
                    st.pyplot(evo_prod_films(), width="content")
                elif box == "Relation popularité-notes":
                    st.pyplot(rel_pop_notes(), width="content")
                elif box == "Matrice de corrélation":
                        st.pyplot(matrix(), width=700)


def page2():
    st.markdown("""<p style='font-size:30px'><br><br>""", unsafe_allow_html=True)
    with st.container(border=False, width='stretch', horizontal_alignment="center", vertical_alignment="center"):
        with st.container(border=False, width=1485, horizontal_alignment="center", vertical_alignment="center", height='content'):
            st.markdown("""<h1 class='page2-title' style='text-align: center;'>Le Ciné en Délire</h1>""", unsafe_allow_html=True)
            st.write("")
            with st.container(border=True):
                st.markdown("""<p style='text-align:justify; font-size:20px;'>Laïus sur le ciné :
                        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, 
                        totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 
                        Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui
                        ratione voluptatem sequi nesciunt."</p>""",unsafe_allow_html=True)

            with st.container(height='stretch', vertical_alignment="center"):
                col1, col2, col3 = st.columns([2,4,2])
                with col1:
                    with st.container(horizontal_alignment="left", vertical_alignment="center", height="stretch"):
                        st.image(logo_cine_en_delire, width="stretch")
                with col2:
                    with st.container(border=True, vertical_alignment="center", height="stretch"):
                        st.markdown("""<p style='text-align:center; font-size:30px;'>Laïus sur l'A&E :<br><br></p>""", unsafe_allow_html=True)
                        st.markdown("""<p style='text-align:justify; font-size:20px;'>Laïus sur le ciné :
                        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, 
                        totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 
                        Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui
                        ratione voluptatem sequi nesciunt."</p>""",unsafe_allow_html=True)
                with col3:
                    with st.container(horizontal_alignment="center", vertical_alignment='center', height="stretch"):
                        loc_tours = pd.DataFrame({"cine" : ["Ciné en délire"], "lat" : [47.383333], "lon" : [0.683333]})
                        st.map(data=loc_tours, latitude="lat", longitude="lon", zoom=10)
                    with st.container(vertical_alignment='center', height="stretch"):
                        st.write("")
            with st.container(height='stretch', vertical_alignment="center", horizontal_alignment="center"):
                with st.container(border=True):
                    st.markdown("""<p style='text-align:center; font-size:30px;'>Laïus sur Claire :<br><br></p>""", unsafe_allow_html=True)
                    st.markdown("""<p style='text-align:justify; font-size:20px;'>
                        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, 
                        totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 
                        Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui
                        ratione voluptatem sequi nesciunt."</p>""",unsafe_allow_html=True)
                with st.container(border=False, horizontal_alignment="center"):
                    st.image(cliente, width=600)


def page3():
    """Page A&E Tracker avec présentation du projet"""
    # CSS pour le thème noir /!\ Semble broken (Thomas) /!\
    st.markdown("""
        <style>
        .page3-title {
            color: #ffffff;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""<p style='font-size:30px'><br>""", unsafe_allow_html=True)
    with st.container(border=False, width='stretch', horizontal_alignment="center", vertical_alignment="center"):
        with st.container(border=False, width=1485, horizontal_alignment="center", vertical_alignment="center"):
            st.markdown("""<h1 class='page3-title'>"Des Essais et de l'art" un catalogue de films d'Art et Essais<br>par la Wild Comedy Show™</h1>""", unsafe_allow_html=True)
            st.write("")
            st.write("")    
            # Encadré principal - Introduction (NOIR)
            with st.container(border=True, horizontal=True, vertical_alignment="center"):
                st.markdown("<h3 style='text-align: center;'>Pourquoi ce moteur de recherche et recommandations ?</h3>", unsafe_allow_html=True)
                st.markdown("""
                    <p style='text-align: center; line-height: 1.7; font-size: 16px;'>
                        "Des Essais et de l'Art" répond à un besoin identifié par le cinéma d'Art et Essai 
                        "Le Ciné en Délire" : offrir aux spectateurs un outil de recherche et de 
                        recommandation adapté au catalogue spécifique des films d'Art et Essai. 
                        Notre objectif est de faciliter la découverte de films en fonction des 
                        préférences des utilisateurs, tout en valorisant la richesse du cinéma 
                        indépendant et d'auteur.
                    </p>
                    """, unsafe_allow_html=True)
            
            st.write("")
            # Trois colonnes pour le contenu principal
            with st.container(height='stretch', vertical_alignment="center"):
                col1, col2, col3 = st.columns([2, 1.2, 2])
                
                with col1:
                    with st.container(border=True, height='stretch', horizontal=True, vertical_alignment="center"):
                        st.markdown("""<h3 style='text-align: center; margin-bottom: -1rem'>Les fonctionnalités du site</h3>""", unsafe_allow_html=True)
                        st.markdown("""<p style='text-align: justify; line-height: 1.7;font-size: 16px;'>
                            • Trouvez rapidement vos films préférés grâce à nos filtres avancés<br>
                            • Naviguez facilement parmi des milliers de films<br>
                            • Découvrez des films similaires à chacun de vos coups de cœur<br>
                            • Explorez notre catalogue par genre, acteur ou réalisateur<br>
                            • Consultez toutes les infos : synopsis, casting, notes<br>                    
                            • Profitez d'une interface claire et intuitive<br>
                            • Une base de données enrichie avec des informations issues de IMDB, TMDB et AFCAE
                            <br><br>
                            <u>Notes :</u> Certains films peuvent ne pas avoir d'affiche disponible ou de résumé en raison de limitations dans les données sources.
                            Par ailleurs, les recommandations sont basées sur un algorithme KNN utilisant les genres, acteurs et réalisateurs pour suggérer des films similaires.
                            <br>Les films présentés peuvent parfois ne pas correspondre entièrement aux standards d'Art et Essai en raison de la diversité des données collectées.
                            <br>Certains des réalisateurs connus du milieu de l'Art et Essai ayant parfois également réalisé des films plus grand public, ceux-ci peuvent apparaître dans les résultats de recherche et recommandations. 
                            </p>""", unsafe_allow_html=True)
                
                with col2:
                    # Encadré noir pour le logo
                    with st.container(border=False, height='stretch', horizontal=False, vertical_alignment="center"):
                        if logo_WCS.exists():
                            st.image(logo_WCS, width='content')
                        else:
                            st.markdown("<h2 style='text-align: center; color: #ffffff;'>WCS LOGO</h2>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    with st.container(border=True, height='stretch', horizontal=True, vertical_alignment="center"):
                        st.markdown("""<h3 style='text-align: center;'>La WCS en quelques mots</h3>""", unsafe_allow_html=True)
                        st.markdown("""<p style='text-align: justify;  line-height: 1.7; font-size: 16px;'>
                            La Wild Comedy Show se positionne comme une société de services data, 
                            capable de transformer des données culturelles en leviers de décision et de découverte, 
                            avec une touche créative fidèle à l'univers du Ciné en Délire.<br><br>
                            Elle est composée d'une équipe de consultants expert en datas et passionnés de cinéma : 
                            Jenny, Solange, Thomas et Jérôme.</p>
                            """, unsafe_allow_html=True)
                        st.image(crew, width='stretch')
            
            st.write("")
                
            with st.container(border=True, horizontal=True, width='stretch', vertical_alignment="center"):
                st.markdown("<h3 style='text-align: center;'> Nous contacter </h3>", unsafe_allow_html=True)
                st.markdown("""
                    <p style='text-align: justify; line-height: 1.7; horizontal-align: center;font-size: 16px;'>
                        Vous êtes une entreprise et vous souhaitez développer des solutions 
                        data sur-mesure pour vos besoins spécifiques ? <br>
                        Contactez nous par email: 
                        contact@wildcomedyshow.fr ou venez nous rendre visite à notre agence:
                        1 rue de la Princesse Licorne 
                        00000 Royaume Arc-en-Ciel                
                    </p>
                    """, unsafe_allow_html=True)

pages = [
        st.Page(page1, icon="📽️", title="Recherche A&E", default=True),
        st.Page(statistiques, icon="✅", title="Statistiques BDD"),
        st.Page(page2, icon="🎭", title="Le ciné en délire"),
        st.Page(page3, icon="🤡", title="Recherche A&E by WCS"),
    ]
    # Setup de la navigation
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
current_page = st.navigation(pages=pages, position="hidden")

    # Setup du menu
@st.cache_data
def menu ():
    st.container(key="menu_container", height='content', border=False, width='stretch', horizontal=True)
    Menu_font = """<div class='Menu_test' style='text-align:center;'><span>Menu</span></div>"""
    with st.container(key="mymenu", height='content', vertical_alignment="center"):
        num_cols_menu = max(len(pages) + 1, 6)
        columns_menu = st.columns(num_cols_menu, vertical_alignment="bottom")
        columns_menu[0].html(Menu_font)
        for col, page in zip(columns_menu[1:-1], pages):
            col.page_link(page, icon=page.icon, width="stretch")

# On lance le menu puis la page
menu()
current_page.run()

# footer fixe en bas de page
@st.cache_data
def footer():
    st.write("<br><br><br>", unsafe_allow_html=True)  # espace pour le footer
    with st.container(border=False, vertical_alignment="center", height="content", width='stretch'):
        st.write("---", unsafe_allow_html=True)  # ligne de séparation
        footer_col1, footer_col2, footer_col3, footer_col4, footer_col5 = st.columns([1, 1, 3, 1, 1])
        with footer_col1:
            with st.container(horizontal_alignment="left", border=False):
                if logo_cine_en_delire.exists():
                    st.image(logo_cine_en_delire, width=220)
                else:
                    st.markdown("<p style='text-align: right; margin: 0; font-size: 20px; color: #c62828; font-weight: bold;'>Ciné en délire</p>", unsafe_allow_html=True)

        with footer_col3:
            with st.container(horizontal_alignment="center", vertical_alignment="center", height="stretch", border=False):
                st.markdown("""<p style='text-align: center; font-size: 17px; color: #555;'>
                            Application créée par la  Wild Comedy Show  pour Le ciné en délire. 
                            Données issus de IMDB, TMDB et AFCAE.<br><br>
                            L'abus de film d'A&E provoque des poussées d'intelligence et un gonflement des chevilles. 
                            A consommer avec modération.<br><br>
                            Pour toute question épineuse, veuillez contacter madame Claire Mercier du Ciné en Délire.<br></p>"""
                            , unsafe_allow_html=True)

        with footer_col5:
            with st.container(horizontal_alignment="right", border=False):
                if logo_WCS.exists():
                    st.image(logo_WCS, width=220)
                else:
                    st.markdown("<p style='text-align: right; margin: 0; font-size: 20px; color: #c62828; font-weight: bold;'>WCS</p>", unsafe_allow_html=True)

footer()
