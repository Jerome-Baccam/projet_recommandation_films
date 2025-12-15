import requests
import pandas as pd
from bs4 import BeautifulSoup
import ast
import numpy as np
import datetime

all_rows = []

# Boucle sur les pages
for page in range(1, 100):
    url = f"https://www.art-et-essai.org/les-films-recommandes?page={page}"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if not table:
        continue

    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    for tr in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) == len(headers):
            all_rows.append(cols)

# Créer DataFrame
df_afcae = pd.DataFrame(all_rows, columns=headers)

# Garder uniquement Titre et Réalisateur
df_titres = df_afcae[["Titre", "Réalisateur","Date","Distributeur"]]

api_key = "27340cce7eb082b0eda4d6682b81ab98"

movies_list = []
# Fusion partielle : ne télécharger que les détails des films AFCAE
for titre in df_titres['Titre']:
    # chercher le film par titre dans TMDB
    url_search = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={titre}&language=fr-FR"
    data = requests.get(url_search).json()
        
    # Si TMDB renvoie au moins un résultat
    if data.get("results"):
        movie = data["results"][0]  # prendre le premier résultat
        movie_id = movie["id"]

        # Détails du film
        url_details = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=fr-FR"
        details = requests.get(url_details).json()

        # Crédits
        url_credits = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}&language=fr-FR"
        credits = requests.get(url_credits).json()

        actors = [(actor["id"], actor["name"]) for actor in credits.get("cast", [])]
        directors = [(crew["id"], crew["name"]) for crew in credits.get("crew", []) if crew.get("job")=="Director"]
        production = [prod["name"] for prod in details.get("production_companies",[])]
        genres = [g["name"] for g in details.get("genres",[])]

        movies_list.append({
            "title": details.get("title"),
            "genres": genres,
            "overview": details.get("overview"),
            "runtime": details.get("runtime"),
            "year": details.get("release_date", ""),
            "vote_count": details.get("vote_count"),
            "vote_average": details.get("vote_average"),
            "actors": actors,
            "directors": directors,
            "production": production
        })

# Créer le DataFrame final à partir de la liste
df_afcae_tmdb = pd.DataFrame(movies_list)

# Sélection et renommage des colonnes
df_affichage = df_afcae_tmdb[[
    "title", "genres", "overview", "runtime", "year",
    "vote_count", "vote_average", "actors", "directors", "production"
]].rename(columns={
    "title": "titre",
    "overview": "résumé",
    "runtime": "temps",
    "year": "année",
    "vote_count": "nombre de votes",
    "vote_average": "votes",
    "actors": "acteurs",
    "directors": "directeurs"
})

df = df_affichage

def get_movie_info(title):
    url = (
        f"https://api.themoviedb.org/3/search/movie"
        f"?api_key={api_key}&query={title}&language=fr"
    )
    r = requests.get(url).json()

    if len(r.get("results", [])) == 0:
        return None, None

    film = r["results"][0]
    return film["id"], film.get("genre_ids", [])

#  Créer les listes d'abord, puis assigner
id_films = []
id_genres = []

for i, row in df.iterrows():
    title = row["titre"]
    id_film, id_genre = get_movie_info(title)
    id_films.append(id_film)
    id_genres.append(id_genre)

# Assigner les colonnes
df["id_film"] = id_films
df["id_genre"] = id_genres

# récupération image
BASE_IMG = "https://image.tmdb.org/t/p/w500"

def get_movie_poster(title):
    """Retourne l'ID du film + le poster_path + l'URL complète."""
    # ✅ CORRECTION 2: Utiliser api_key au lieu de API_KEY
    url = (
        f"https://api.themoviedb.org/3/search/movie"
        f"?api_key={api_key}&query={title}&language=fr"
    )
    r = requests.get(url).json()

    if len(r.get("results", [])) == 0:
        return None, None, None

    data = r["results"][0]

    id_film = data["id"]
    poster_path = data.get("poster_path")

    if poster_path:
        full_url = BASE_IMG + poster_path
    else:
        full_url = None

    return id_film, poster_path, full_url

#  listes
id_films_poster = []
poster_paths = []
poster_urls = []

for i, row in df.iterrows():
    title = row["titre"]
    id_film, path, img_url = get_movie_poster(title)
    
    id_films_poster.append(id_film)
    poster_paths.append(path)
    poster_urls.append(img_url)

# Assigner les colonnes
df["poster_path"] = poster_paths
df["poster_url"] = poster_urls

# Convertir en datetime
df['année'] = pd.to_datetime(df['année'], errors='coerce')

# Formater seulement les dates valides (non-NaT)
df['année'] = df['année'].apply(
    lambda x: x.strftime('%d-%m-%Y') if pd.notna(x) else None
)

def liste(x):
    """
    Transforme la cellule en liste Python :
    - NaN → []
    - string représentant liste → liste Python
    - array/numpy → liste Python
    - déjà une liste → inchangée
    """
    if x is None:
        return []
    elif isinstance(x, list):
        return x
    elif isinstance(x, np.ndarray):
        return list(x)
    elif isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    else:
        return []

# Appliquer uniquement sur la colonne 'acteurs'
df['acteurs'] = df['acteurs'].apply(liste)

# Ne garder que les noms si ce sont des tuples (id, nom)
df['acteurs'] = df['acteurs'].apply(
    lambda x: [name for _, name in x] if all(isinstance(i, tuple) and len(i) == 2 for i in x) else x
)

# Appliquer uniquement sur la colonne 'directeurs'
df['directeurs'] = df['directeurs'].apply(liste)

# Si la liste contient des tuples (id, nom), on ne garde que les noms
df['directeurs'] = df['directeurs'].apply(
    lambda x: [name for _, name in x] if all(isinstance(i, tuple) and len(i) == 2 for i in x) else x
)

# Remplacer les crochets
df['production'] = df['production'].str.strip("[]").str.replace("'", "")

# Remplacer les chaînes vides par "Inconnu"
df['production'] = df['production'].replace('', 'Inconnu')

# supprimer lignes où l'année est manquante
df = df.dropna(subset=['année'])
# supprimer lignes où les résumés sont vides
df["résumé"] = df["résumé"].fillna("Non disponible")
# remplacer valeur vide des colonnes défini par Inconnu
colonnes = ["id_film", "id_genre", "poster_path", "poster_url"]

for col in colonnes:
    df[col] = df[col].replace("", "Inconnu").fillna("Inconnu")

df.to_csv("films.csv", index=False)

df.head()