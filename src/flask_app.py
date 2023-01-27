import json
import pickle
import re
import zipfile
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from lime import lime_tabular
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Creation de l'instance Flask.

app = Flask(__name__)

# On charge les données
data_train = pd.read_csv('../train.zip', compression='zip', low_memory=False)
data_test = pd.read_csv('../test.zip', compression='zip', low_memory=False)



# On crée deux variables en attente qui deviendront
# des variables globales après l'initialisation de l'API.
# Ces variables sont utilisées dans plusieurs fonctions de l'API.
train1 = None
train_a = None
test_b = None
model = None

# On crée la liste des ID clients qui nous servira dans l'API
id_client = data_test["SK_ID_CURR"].values
id_client = pd.DataFrame(id_client)


# routes /  creation des fonctions qui génèrent des pages
# Entraînement du modèle
@app.route("/init_model", methods=["GET"])
def init_model():
    # 1/ On prépare les données.
    df_train, df_test = features_engineering(data_train, data_test)
    print("Features engineering done")

    # 2/ On fait le préprocessing des données.
    df_train, df_test = preprocesseur(df_train, df_test)

    # 3/ On transforme les datasets préparés en variabe globale.
    global train_a
    train_a = df_train.copy()
    global test_b
    test_b = df_test.copy()
    print("Preprocessing done")

    global train1
    train1 = df_train.copy()
    # 4/ On entraîne le modèle et on le transforme en variable globale pour la fonction predict.

    global clf
    clf = load_model(df_train, data_train)

    print("Training clf done")

    # 5/ On sélectionne les voisins les plus proches.
    global knn
    knn = entrainement_knn(df_train)
    print("Training knn done")

    # 5/ Features locales Lime
    global exp
    exp = lime_model(df_train)
    print("exp done")

    return jsonify(["Initialisation terminée."])


# Chargement de l'ID client
@app.route("/load_data", methods=["GET"])
def load_data():
    return id_client.to_json(orient='values')


# Chargement d'informations générales sur le client
@app.route("/infos_client", methods=["GET"])
def infos_client():
    id = request.args.get("id_client")

    data_client = data_test[data_test["SK_ID_CURR"] == int(id)]

    dict_infos = {
        "Id Client": data_client["SK_ID_CURR"].item(),
        "Moyenne des échéances restant dues des crédits antérieurs": int(
            data_client["PREV_APPL_MEAN_POS_MEAN_CNT_INSTALMENT_FUTURE"].values),
        "Nombre d'enfants": data_client["CNT_CHILDREN"].item(),
        "Age (ans)": int(data_client["DAYS_BIRTH"].values / -365),
        "Ancienneté de l'emploi (ans)": int(data_client["DAYS_EMPLOYED"].values / -365),
        "Revenu total": data_client["AMT_INCOME_TOTAL"].item(),
        "Montant du credit": data_client["AMT_CREDIT"].item(),
        "Annuites": data_client["AMT_ANNUITY"].item(),
        "Valeur du bien": data_client["AMT_GOODS_PRICE"].item()
    }
    return dict_infos


# Chargement des global feature importance
@app.route("/load_features", methods=["GET"])
def load_features():
    lgbm_features = clf.feature_importances_
    feature_imp = pd.DataFrame(sorted(zip(train1.columns, lgbm_features)), columns=['Feature', 'Value'])

    print(feature_imp)

    data_feat = json.loads(feature_imp.to_json(orient='index'))

    return data_feat


# Chargement des prédiction (Scoring credit)
@app.route("/predict", methods=["GET"])
def predict():
    id = request.args.get("id_client")

    print("Analyse data_test :")
    print(data_test.shape)
    print(data_test[data_test["SK_ID_CURR"] == int(id)])

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    print(index[0])

    data_client = test_b.iloc[index].values

    print(data_client)

    prediction = clf.predict_proba(data_client)

    prediction = prediction[0].tolist()

    return jsonify(prediction)


# Chargement des infos gen sur les 10 proches voisins
@app.route("/load_voisins", methods=["GET"])
def load_voisins():
    id = request.args.get("id_client")

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    data_client = test_b.iloc[index].values

    distances, indices = knn.kneighbors(data_client)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_voisins = data_train.iloc[indices[0], :]

    response = json.loads(df_voisins.to_json(orient='index'))

    return response


# Chargement des locales features importance
@app.route("/load_localfeat", methods=["GET"])
def load_localfeat():
    id = request.args.get("id_client")

    idx = data_test[data_test["SK_ID_CURR"] == int(id)].index

    local_feat = exp.explain_instance(
        data_row=test_b.iloc[idx[0]],
        predict_fn=clf.predict_proba,
        num_features=15,
    )
    features_local = pd.DataFrame(local_feat.as_list(), columns=['Feature_name', "Value"])
    data_local = json.loads(features_local.to_json(orient='index'))
    return data_local


def features_engineering(data_train, data_test):
    # Cette fonction regroupe toutes les opérations de features engineering
    # mises en place sur les sets train & test

    data_train = pd.get_dummies(data_train)
    data_test = pd.get_dummies(data_test)

    train_labels = data_train['TARGET']
    # Align the training and testing data, keep only columns present in both dataframes
    data_train, data_test = data_train.align(data_test, join='inner', axis=1)
    # Add the target back in
    data_train['TARGET'] = train_labels

    # Traitement des valeurs négatives
    data_train['DAYS_BIRTH'] = abs(data_train['DAYS_BIRTH'])

    return data_train, data_test


def preprocesseur(df_train, df_test):
    # Cette fonction permet d'imputer les valeurs manquantes
    # et aussi d'appliquer un MinMaxScaler

    if "TARGET" in df_train:
        train = df_train.drop(columns=["TARGET"])

    else:
        train = df_train.copy()

    # Feature names
    features = list(train.columns)

    # Imputation par la médiane des Nan
    imputer = SimpleImputer(strategy='median')

    scaler = MinMaxScaler(feature_range=(0, 1))

    imputer.fit(train)

    # Imputation par la médiane des Nan des données du jeu d'entrainement & test

    imputed_train = imputer.transform(train)
    imputed_test = imputer.transform(df_test)

    test = df_test.copy()

    train[train.isnull()] = imputed_train
    df_test[df_test.isnull()] = imputed_test

    # appliquer MinMaxScaler aux valeurs des datasets : train & test
    test = df_test.copy()

    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    train = pd.DataFrame(scaled_train, index=train.index, columns=train.columns)
    test = pd.DataFrame(scaled_test, index=test.index, columns=test.columns)

    return train, test


def load_model(df_train, target):
    # Chargement du modèle de prédiction
    df_train = df_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    with open('../model_pkl', 'rb') as l:
        clf = pickle.load(l)
    clf.fit(df_train, target['TARGET'])

    return clf


def entrainement_knn(df):
    # Cette fonction permet de selection les voisins les plus proches d'un client donné

    print("En cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)

    return knn


def lime_model(df_train):
    # Cette fonction permet de selectionner les features locales d'une prédiction donnée

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(df_train),
        feature_names=df_train.columns,
        class_names=["Solvable", "Non Solvable"],
        mode='classification')

    return explainer


if __name__ == "__main__":
    app.run(debug=True)
