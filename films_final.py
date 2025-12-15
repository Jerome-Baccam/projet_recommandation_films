import pandas as pd
from pathlib import Path

# Find and resolve the path of films_final.csv
csv_path = Path(__file__).parent / "films_final.csv"
resolved_path = csv_path.resolve()

print(f"Script location: {Path(__file__).resolve()}")
print(f"CSV file path: {resolved_path}")
print(f"CSV file exists: {csv_path.exists()}\n")

films = pd.read_csv("Projets_Data/Projet_2/projet_recommandation_films/films_final.csv")
print(films.head(5))
