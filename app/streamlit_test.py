import streamlit as st
import streamlit_authenticator as stauth
import streamlit_option_menu as som
from pathlib import Path

# On intègre un CSS personnalisé
css_path = Path(__file__).parent / "streamlit.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS file not found at: {css_path}")

# chargement du CSV avec conversion des listes string en listes réelles
def load_data():
    try:
        csv_path = Path(__file__).parent / 'films_final.csv'
        df = pd.read_csv(csv_path)
        for col in ['genres', 'acteurs', 'directeurs']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return pd.DataFrame()

# création du modèle KNN
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
        genres_weighted = genres_matrix * 3
        actors_weighted = actors_matrix * 2
        directors_weighted = directors_matrix * 5
        
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
        #  recommandations par genre simple
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

# Chargement des données + modèle
df = load_data()
if df.empty:
    st.error("Impossible de charger les données. Vérifiez que films_final.csv existe.")
    model_data = None
else:
    # Créer le modèle KNN
    model_data = create_recommendation_model(df)


# Affichage des détails d'un film sélectionné
def display_film_detail(film_data):
    """Affiche les détails d'un film avec ses recommandations"""
    
    # Bouton retour
    if st.button("← Retour à la recherche"):
        st.session_state.selected_film = None
        st.rerun()
    
    st.markdown("---")
    
    # Affichage du film
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Affiche")
        if pd.notna(film_data['poster_url']) and film_data['poster_url'] != 'Inconnu':
            try:
                st.image(film_data['poster_url'], use_container_width=True)
            except:
                st.info("Affiche non disponible")
        else:
            st.info("Affiche non disponible")
    
    with col2:
        st.title(film_data['titre'])
        
        # Réalisateurs
        if isinstance(film_data['directeurs'], list) and len(film_data['directeurs']) > 0:
            st.markdown(f"**Réalisé par :** {', '.join(film_data['directeurs'])}")
        
        # Acteurs
        if isinstance(film_data['acteurs'], list) and len(film_data['acteurs']) > 0:
            acteurs_str = ", ".join(film_data['acteurs'][:5])
            st.markdown(f"**Avec :** {acteurs_str}")
        
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
    st.markdown("---")
    
    # RECOMMANDATIONS KNN
    st.markdown("##  Si vous avez aimé ce film, vous aimerez probablement :")
    
    recommendations = get_recommendations(film_data['titre'], df, model_data, 5)
    
    if not recommendations.empty:
        cols = st.columns(5)
        for idx, (i, row) in enumerate(recommendations.iterrows()):
            with cols[idx]:
                # Affiche
                if pd.notna(row['poster_url']) and row['poster_url'] != 'Inconnu':
                    try:
                        st.image(row['poster_url'], use_container_width=True)
                    except:
                        st.info("Pas d'affiche")
                else:
                    st.info("Pas d'affiche")
                
                # Titre
                st.markdown(f"**{row['titre']}**")
                
                # Note
                if pd.notna(row['votes']):
                    st.caption(f"{row['votes']:.1f}/10")
                
                # Année
                if pd.notna(row['année']):
                    try:
                        annee_str = str(row['année'])
                        if '-' in annee_str:
                            annee = annee_str.split('-')[-1]
                        else:
                            annee = int(float(annee_str))
                        st.caption(f"{annee}")
                    except:
                        pass
                
                # Bouton voir ce film recommandé
                if st.button("Voir ce film", key=f"rec_{idx}", use_container_width=True):
                    st.session_state.selected_film = row['titre']
                    st.rerun()
    else:
        st.warning("Aucune recommandation disponible pour ce film.")

# Notre home page (recherche films) 
def page1():
    # Si un film est sélectionné, affiche détails du film + recommandations)
    if st.session_state.selected_film is not None:
        film_data = df[df['titre'] == st.session_state.selected_film].iloc[0]
        display_film_detail(film_data)
    
    # Je créé trois colonnes pour centrer le contenu
    lay_gauche, lay_centre, lay_droit = st.columns([1, 20, 1])

    # titre
    with lay_centre:
        # box stylisée pour le titre et les filtres
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-title">Recherche de films A&E</h1>', unsafe_allow_html=True)

        # Filtres container (inside the styled box)
        with st.container():
            st.subheader("Filtres")

            # colonnes des filtres principaux
            but_gauche, but_centre, but_droit = st.columns(3)
            with but_gauche:
                st.text_input("Mot-clef", key="filter_keyword")
                st.selectbox("Note", options=["Any", ">= 5", ">= 7", ">= 9"], key="filter_note")
            with but_centre:
                st.text_input("Acteur", key="filter_actor")
                st.selectbox("Popularité", options=["Any", "Low", "Medium", "High"], key="filter_popularity")
            with but_droit:
                st.text_input("Réalisateur", key="filter_director")
                st.number_input("Année", min_value=1900, max_value=2100, value=2020, step=1, key="filter_year")

            st.write("Genre")
            # Genres comme toggle button dans des petites colonnes
            but_a, but_b, but_c, but_d, but_e = st.columns(5)
            with but_a:
                st.checkbox("A", key="genre_a")
                st.checkbox("F", key="genre_f")
            with but_b:
                st.checkbox("B", key="genre_b")
                st.checkbox("G", key="genre_g")
            with but_c:
                st.checkbox("C", key="genre_c")
                st.checkbox("H", key="genre_h")
            with but_d:
                st.checkbox("D", key="genre_d")
                st.checkbox("I", key="genre_i")
            with but_e:
                st.checkbox("E", key="genre_e")
                st.checkbox("J", key="genre_j")

        # Close the main box
        st.markdown('</div>', unsafe_allow_html=True)
        
        
def page2():
    st.write("Un petit test pour du multipage app")
    st.image("https://media.makeameme.org/created/impressive-2y23ct.jpg", width=600)

def page3():
    st.write("Un autre petit test pour du multipage app")
    st.image("https://i.pinimg.com/564x/56/b9/a9/56b9a962f481a4212bce3f82b151433d.jpg", width=600)

pages = [
        st.Page(page1, icon="📽️", title="Recherche A&E"),
        st.Page(page2, icon="🎭", title="Le ciné en délire"),
        st.Page(page3, icon="🤡", title="A&E tracker by WCS"),
    ]
    # Setup de la navigation
st.set_page_config(layout="wide")
current_page = st.navigation(pages=pages, position="hidden")

    # Setup du menu
num_cols_menu = max(len(pages) + 1, 8)
columns_menu = st.columns(num_cols_menu, vertical_alignment="bottom")
columns_menu[0].write("**Menu**")
for col, page in zip(columns_menu[1:-1], pages):
    col.page_link(page, icon=page.icon)
current_page.run()
