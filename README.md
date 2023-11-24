# scoring_model

L’objectif de ce projet est de développer un algorithme de classification permettant de catégoriser une demande de crédit par un client en accordé ou refusé. L’autre but du projet est de développer un dashboard interactif, afin d’expliquer les décisions d’octroi de crédit et d’explorer les facilement les données des clients.

Les données utilisées dans ce projet sont disponibles sur le lien suivant : https://www.kaggle.com/c/home-credit-default-risk/data

## Description du répertoire

**Dossiers :**
- _.github/workflows_ contient le fichier _main\_scoringmodelapi.yml_ permettant le déploiement de l’API REST sur un serveur Azure.
- _input_ contient trois fichiers au format pickle. Le fichier _df.pkl_ contient le jeu de données, le fichier _model.pkl_ contient le modèle choisi pour la réalisation du projet et le fichier _shap\_values_ contient les SHAP values de chaque individu.
- _unit\_testing_ contient le script Python _unit\_test.py_ permettant d’effectuer les tests unitaires.

**Fichiers :**
- _.gitattributes_ permet la gestion des fichiers au format pickle “*.pkl” par Git LFS.
- _API.py_ est un script Python générant l’API REST.
- _Dashboard.py_ est un script Python générant le dashboard interactif.
- _Data Drift Report.html_ est le tableau HTML de l’analyse du Data Drift pour les principales features.
- _Modelisation.ipynb_ est le notebook Jupyter permettant l’élaboration du modèle.
- _requirements.txt_ contient la liste des packages utilisés pour la réalisation du projet.

## Installation

Le projet a été réalisé sur un environnement Python 3.11. Les packages peuvent être installés grâce à la ligne de commande suivante : 

`pip install -r requirements.txt`

## Déploiement de l’API REST

L’API a été déployé sur un serveur Azure avec une pile d’exécution Python 3.11.

## Déploiement du dashboard interactif

Le dashboard a été déployé sur la plateforme Streamlit Community Cloud avec la version 3.11 de Python. Les données et les prédictions du modèle affichés par le dashboard sont récupéré via des requêtes vers l'API REST.

Le dashboard interactif est disponible sur le lien suivant : https://scoringmodel-fu6ptfj5hcfsn3hqrrsmpb.streamlit.app/