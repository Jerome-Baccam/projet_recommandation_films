# app.py - Ceci est un commentaire, Streamlit ne le lit pas.

import streamlit as st

# 1. Ajoutons un titre principal
st.title("Projet de Recommandation de Films")

# 2. Ajoutons un simple message de bienvenue
st.write("Bienvenue Solange ! Cette application Streamlit est lancée avec Python 3.11.6.")

# 3. Ajoutons un petit élément interactif (un bouton)
if st.button('Cliquez-moi !'):
    st.write('Bravo, le bouton fonctionne !')
    