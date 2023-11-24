import pickle

import pytest
import pandas as pd


@pytest.fixture
def create_df():
    # Chargement des données dans un DataFrame
    df = pd.read_pickle('input/df.pkl')
    return df

@pytest.fixture
def create_model():
    # Chargement du modèle
    with open('input/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def test_shape(create_df):
    # Assurance de la forme du DataFrame
    assert create_df.shape == (356251, 247)

def test_id(create_df):
    # Assurance de la présence de la colonne "SK_ID_CURR" dans le DataFrame
    assert 'SK_ID_CURR' in create_df.columns

def test_target(create_df):
    # Assurance de la présence de la colonne "TARGET" dans le DataFrame
    assert 'TARGET' in create_df.columns

def test_steps(create_model):
    # Etapes de référence pour le pipeline
    ref_steps = ['imputer', 'scaler', 'oversampling', 'logistic']

    # Etapes du pipeline du modèle
    pipe_steps = list(create_model.named_steps.keys())

    # Assurance de la conformité des étapes du pipeline
    assert ref_steps == pipe_steps

def test_params(create_model):
    # Paramètres de référence pour la régression logistique
    ref_params = {'C': 10, 
                  'fit_intercept': True}
    
    # Paramètres de la régression logistique
    model_params = create_model.named_steps['logistic'].get_params()
    
    # Pour chaque paramètre de référence
    for param in ref_params:
        # Assurance de la conformité du paramètre de la régression logistique
        assert ref_params[param] == model_params[param]

def test_predict(create_df, create_model):
    # Sélection du client 100002 (premier du jeu de données)
    client = create_df[create_df['SK_ID_CURR'] == 100002]

    # Amputation des colonnes "SK_ID_CURR" et "TARGET"
    client = client.drop(['SK_ID_CURR', 'TARGET'], axis=1)

    # Prédiction de la classe du client
    pred = create_model.predict(client.values)

    # Assurance que la classe prédite est 1
    assert pred == 1