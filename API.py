import pickle
import json

import pandas as pd
from flask import Flask


app = Flask(__name__)


# Chargement des données
with open('input/df.pkl', 'rb') as f:
    df = pickle.load(f)

# Chargement du modèle
with open('input/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Chargement des feature importances locales (SHAP values)
with open('input/shap_values.pkl', 'rb') as f:
    sv_df = pickle.load(f)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict_target/<int:client_id>')
def predict_target(client_id):
    # Sélection du client en fonction de son ID
    client = df[df['SK_ID_CURR'] == client_id]
    
    # Amputation des colonnes "SK_ID_CURR" et "TARGET"
    client = client.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    
    # Prédiction de la classe du client (0 ou 1)
    pred = model.predict(client.values)
    
    # Conversion de la prédiction en un scalaire Python standard
    pred = pred.item()
    
    # Renvoie la prédiction au format JSON
    return json.dumps(pred)

@app.route('/predict_proba/<int:client_id>')
def predict_proba(client_id):
    # Sélection du client en fonction de son ID
    client = df[df['SK_ID_CURR'] == client_id]
    
    # Amputation des colonnes "SK_ID_CURR" et "TARGET"
    client = client.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    
    # Prédiction des probabilités d'appartenance aux classes 0 et 1
    pred = model.predict_proba(client.values)
    
    # Création d'un dictionnaire à partir de la prédiction
    # Les clés sont converties en un scalaire Python standard
    pred_dict = {model.classes_[0].item(): pred[0][0],
                 model.classes_[1].item(): pred[0][1]}
    
    # Renvoie le dictionnaire converti au format JSON
    return json.dumps(pred_dict)

@app.route('/client_data/<int:client_id>')
def client_data(client_id):
    # Sélection du client en fonction de son ID
    client = df[df['SK_ID_CURR'] == client_id]
    
    # Renvoie le DataFrame au format JSON
    return client.to_json()

@app.route('/target_data/<feat>/<int:target>')
def target_data(target, feat):
    # Sélection des individus appartenant à la classe choisie de la target,
    # pour la variable choisie
    data = df.loc[df['TARGET']==target, feat]
    
    # Renvoie la sélection au format JSON
    return data.to_json()

@app.route('/mean_data')
def mean_data():
    # Calcul de la moyenne de chaque variable (à l'exception de "SK_ID_CURR")
    means = df.drop('SK_ID_CURR', axis=1).mean()
    
    # Conversion en DataFrame
    means = means.to_frame()
    
    # Renvoie le DataFrame au format JSON
    return means.to_json()

@app.route('/median_data')
def median_data():
    # Calcul de la médiane de chaque variable (à l'exception de "SK_ID_CURR")
    medians = df.drop('SK_ID_CURR', axis=1).median()
    
    # Conversion en DataFrame
    medians = medians.to_frame()
    
    # Renvoie le DataFrame au format JSON
    return medians.to_json()

@app.route('/global_imp')
def global_imp():
    # Récupération des feature importances globales
    imp = model.named_steps['logistic'].coef_

    # Conversion des feature importances en Series, avec les noms des features en index
    imp = pd.Series(imp[0].T, index=model.feature_names_in_)
    
    # Renvoie la Series au format JSON
    return imp.to_json()

@app.route('/local_imp/<int:client_id>')
def local_imp(client_id):
    # Sélection des SHAP values du client en fonction de son ID
    client_sv = sv_df[sv_df['SK_ID_CURR'] == client_id]

    # Amputation de la colonne "SK_ID_CURR"
    client_sv = client_sv.drop('SK_ID_CURR', axis=1)

    # Conversion des SHAP values du client en Series
    client_sv = client_sv.squeeze()
    
    # Renvoie la Series au format JSON
    return client_sv.to_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000')