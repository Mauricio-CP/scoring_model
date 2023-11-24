import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_URL = 'https://scoringmodelapi.azurewebsites.net'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}


def plot_hist(target_data, client_data):
    # Instanciation de la figure et de ses dimensions
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Histogramme
    ax.hist(target_data, bins=50, density=True)
    
    # Droite représentant la valeur du client
    ax.axvline(client_data.loc[target_data.columns].values, color='red', linestyle='-')
    
    # Titre, labels, grille et affichage
    ax.set_title(f'Distribution de la variable {target_data.columns[0]}', fontsize=14)
    ax.set_ylabel('Densité', fontsize=12)
    ax.set_xlabel(f'{target_data.columns[0]}', fontsize=12)
    ax.grid(linestyle=':')
    st.pyplot(fig)

def plot_feat_imp(n, feat_imp):
    # Sélection des plus grandes feature importances,
    # en valeur absolue par ordre croissant.
    best_feat_imp = abs(feat_imp).sort_values(na_position='first')[-n:]

    # Instanciation de la figure et de ses dimensions
    fig, ax = plt.subplots(figsize=(7, slider*0.5))

    # Coloration des bâtons en fonction des valeurs
    color = np.where(feat_imp[best_feat_imp.index] >= 0, 'red', 'steelblue')

    # Bar plot horizontal, titre, label, grille et affichage
    ax.barh(y=best_feat_imp.index,
            width=feat_imp[best_feat_imp.index],
            color=color)
    ax.set_title('Main feature importances', fontsize=14)
    ax.set_xlabel('Feature importance', fontsize=12)
    ax.grid(axis='x', linestyle=':')
    st.pyplot(fig)


# Utilisation de toute la largeur de l'écran
st.set_page_config(layout='wide')

# Titre de la page Web
st.title("Dashboard pour la visualisation des données et l'interprétation des prédictions")

# Instanciation d'une barre latérale
with st.sidebar:
    # Saisie de l'ID du client
    client_id = st.number_input(label='SK_ID_CURR', step=1, format='%d')

    if client_id:
        # Prédiction de la classe du client (0 ou 1)
        target = requests.get(
            f'{BASE_URL}/predict_target/{client_id}', headers=HEADERS)
        target = target.json()

        # Prédiction des probabilités d'appartenance aux classes 0 et 1
        proba = requests.get(
            f'{BASE_URL}/predict_proba/{client_id}', headers=HEADERS)
        proba = proba.json()

        # Probabilité (pourcentage) d'appartenance à la classe du client
        proba_perc = round(proba[str(target)]*100, 2)

        # Affiche la classe prédite du client
        st.title('Prédiction')
        if target == 0:
            st.markdown(f':green[**Crédit accordé** ({proba_perc}%)]')
        elif target == 1:
            st.markdown(f':red[**Crédit refusé** ({proba_perc}%)]')

# Si l'ID du client a été entrée
if client_id:
    # Insertion de 2 conteneurs disposés en colonnes côte à côte
    col1, col2 = st.columns(2)

    # Instanciation du premier conteneur
    with col1:
        # Titre du conteneur
        st.header('Données')

        # Données moyennes
        mean_data = pd.read_json(
            f'{BASE_URL}/mean_data', orient='index', storage_options=HEADERS)
        mean_data.index = ['Mean']  # Modification de l'index
        mean_data.drop('TARGET', axis=1, inplace=True)  # Amputation
        mean_data = mean_data.T  # Transposition

        # Données médianes
        median_data = pd.read_json(
            f'{BASE_URL}/median_data', orient='index', storage_options=HEADERS)
        median_data.index = ['Median']  # Modification de l'index
        median_data.drop('TARGET', axis=1, inplace=True)  # Amputation
        median_data = median_data.T  # Transposition

        # Données personnelles
        client_data = pd.read_json(
            f'{BASE_URL}/client_data/{client_id}', storage_options=HEADERS)
        client_data.index = [client_id]  # Modification de l'index
        client_data.drop(['SK_ID_CURR', 'TARGET'], axis=1,
                         inplace=True)  # Amputation
        client_data = client_data.T  # Transposition

        # Concaténation des données et affichage du tableau
        data = pd.concat([mean_data, median_data, client_data], axis=1)
        st.dataframe(data)

        # Description du tableau
        table_text = \
            """
            Le tableau est constitué de 4 colonnes :
            - la première contient les noms des variables 
            - la deuxième contient les valeurs moyennes
            - la troisième contient les valeurs médianes
            - la quatrième contient les valeurs pour le client sélectionné

            """ \
            "Il est possible d’ordonner les données par ordre croissant ou décroissant, " \
            "en fonction d’une colonne en cliquant sur son en-tête."
        st.markdown(table_text)

        # Sous-titre du conteneur
        st.subheader('Distribution')

        # Choix de la variable à visualiser
        feat_options = ['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
        choice_feat = st.selectbox(label='Variable à visualiser :', 
                                   options=feat_options)
        
        # Choix des individus à représenter
        target_options = ['accordé', 'refusé']
        choice_target = st.radio(label='Individus dont le crédit a été :', 
                                 options=target_options)
        
        # Crédit accordé
        if choice_target == target_options[0]:
            target_data = pd.read_json(
                f'{BASE_URL}/target_data/{choice_feat}/0', orient='index', storage_options=HEADERS)
            target_data.columns = [choice_feat]
            plot_hist(target_data, client_data)  # Histogramme
        # Crédit refusé
        elif choice_target == target_options[1]:
            target_data = pd.read_json(
                f'{BASE_URL}/target_data/{choice_feat}/1', orient='index', storage_options=HEADERS)
            target_data.columns = [choice_feat]
            plot_hist(target_data, client_data)  # Histogramme

        # Description de la figure
        hist_text = \
            "Le graphique est un histogramme représentant la distribution d’une variable " \
            "selon un groupe d’individus. " \
            "L’axe des ordonnées mesure la densité, " \
            "alors que l’axe des abscisses mesure la valeur de la variable. " \
            "La droite verticale rouge représente la valeur de la variable pour le client sélectionné."
        st.markdown(hist_text)


    # Instanciation du second conteneur
    with col2:
        # Titre du conteneur
        st.header('Feature importances')

        # Définition de la feature importance
        feat_text = \
            "La feature importance est un score calculé pour une variable du modèle de prédiction. " \
            "Ce score reflète l’importance de la variable pour la prédiction. " \
            "La feature importance peut être globale ou locale. " \
            "Elle est globale lorsqu’elle prend en compte toutes les prédictions, " \
            "et locale lorsqu’elle se concentre sur une prédiction spécifique."
        st.markdown(feat_text)

        # Feature importances globales ou locales
        imp_options = ['globales', 'locales']
        choice_feat_imp = st.radio(label='Feature importances à visualiser :',
                                   options=imp_options)

        # Nombre de features à représenter sur le graphique
        slider = st.slider(label='Nombre de features principales à représenter :',
                           min_value=5,
                           max_value=20,
                           step=1,
                           format='%d')

        if choice_feat_imp == imp_options[0]:
            # Feature importances globales
            global_imp = pd.read_json(
                f'{BASE_URL}/global_imp', typ='series', storage_options=HEADERS)
            plot_feat_imp(n=slider, feat_imp=global_imp)  # Bar plot
        elif choice_feat_imp == imp_options[1]:
            # Feature importances locales
            local_imp = pd.read_json(
                f'{BASE_URL}/local_imp/{client_id}', typ='series', storage_options=HEADERS)
            plot_feat_imp(n=slider, feat_imp=local_imp)  # Bar plot

        # Description de la figure
        fig_text = \
            "Le graphique est un diagramme à barres représentant les principales feature importances, " \
            "allant de la barre la plus longue (en haut) vers la barre la plus courte (en bas). " \
            "L’axe des abscisses mesure la valeur de chaque barre, qui est proportionnelle à sa longueur. " \
            "Les barres sont bleues lorsqu’elles sont négatives et rouges lorsqu’elles sont positives. " \
            "Sur l’axe des ordonnées se trouve pour chaque barre le nom de la variable correspondante. " \
            "Lorsqu'une variable a une importance (globale ou locale) positive, " \
            "cela indique qu'elle augmente la probabilité de refus de crédit. " \
            "Au contraire, lorsqu’elle a une importance négative, " \
            "cela indique qu’elle diminue cette probabilité."
        st.markdown(fig_text)