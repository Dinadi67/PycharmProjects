# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:17:25 2022

@author: didkh
"""

from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import dash
import json
import plotly.express as px
import requests

import pandas as pd
import plotly.graph_objects as go

app = Dash(__name__)
server = app.server

URL_API = "http://127.0.0.1:5000/"


# Récuperer la Liste des clients


def load_selectbox():

    # Requête permettant de récupérer la liste des ID clients
    data_json = requests.get(URL_API + "load_data",verify=False)
    data = data_json.json()

    # Récupération des valeurs sans les [] de la réponse
    lst_id = []
    for i in data:
        lst_id.append(i[0])

    return lst_id



lst_id = load_selectbox()


# 0/ Features Globales (Graphe)

def load_features():

    features = requests.get(URL_API + "load_features",verify=False)
    dataf = features.json()
    return dataf

lst_features = load_features()

lst_features = pd.DataFrame.from_dict(lst_features).T
lst_features = lst_features.sort_values(by="Value", ascending=False)
lst_features = lst_features[:20]

fig = px.bar(lst_features, x="Feature", y="Value",color ="Value")

# 1/On affiche les informations descriptives de l'ID client choisi

infos_clients = requests.get(URL_API + "infos_client",params={"id_client":"100001"},verify=False)
infos_clients =json.loads(infos_clients.content.decode("utf-8"))

infos_clients = pd.DataFrame.from_dict([infos_clients])
infos_clients = infos_clients[["Id Client","Moyenne des échéances restant dues des crédits antérieurs","Nombre d'enfants","Age (ans)","Ancienneté de l'emploi (ans)",
                         "Revenu total","Montant du credit","Annuites","Valeur du bien"]]


basic_table = dash_table.DataTable(infos_clients.to_dict('records'), [{"name": i, "id": i} for i in infos_clients], 
                                   id='table-dropdown',)


# 2/Graphe : pie-chart(prédiction) de l'ID client choisi

prediction = requests.get(URL_API + "predict", params={"id_client":"100001"},verify=False)
prediction = json.loads(prediction.content.decode("utf-8"))


# 3/Tableau permettant de récupérer les 10 clients similaires


# les plus proches de l'ID client choisi

voisins = requests.get(URL_API + "load_voisins", params={"id_client":"100001"},verify=False)
voisins = json.loads(voisins.content.decode("utf-8"))
 
# On transforme le dictionnaire en dataframe
voisins = pd.DataFrame.from_dict(voisins).T

voisins = voisins[["SK_ID_CURR","PREV_APPL_MEAN_POS_MEAN_CNT_INSTALMENT_FUTURE","CNT_CHILDREN","DAYS_BIRTH","DAYS_EMPLOYED",
                    "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"]]
voisins["DAYS_BIRTH"] = (voisins["DAYS_BIRTH"] / -365).astype(int)
voisins["DAYS_EMPLOYED"] = (voisins["DAYS_EMPLOYED"] / -365).astype(int)
voisins["PREV_APPL_MEAN_POS_MEAN_CNT_INSTALMENT_FUTURE"] = voisins["PREV_APPL_MEAN_POS_MEAN_CNT_INSTALMENT_FUTURE"].astype(int)
voisins.rename(columns={"SK_ID_CURR": "Id Client", 
                        "PREV_APPL_MEAN_POS_MEAN_CNT_INSTALMENT_FUTURE": "Moyenne des échéances restant dues des crédits antérieurs",
                        "CNT_CHILDREN" : "Nombre d'enfants", "DAYS_BIRTH" : "Age (ans)",
                        "DAYS_EMPLOYED" : "Ancienneté de l'emploi (ans)",
                        "AMT_INCOME_TOTAL" : "Revenu total",
                        "AMT_CREDIT" : "Montant du credit","AMT_ANNUITY" :"Annuites",
                        "AMT_GOODS_PRICE" : "Valeur du bien"}, inplace=True)

voisins_table = dash_table.DataTable(voisins.to_dict('records'), [{"name": j, "id": j} for j in voisins.columns], 
                                   id='table-voisins',)

# 4/ Locales features

local_feat = requests.get(URL_API + "load_localfeat", params={"id_client":"100001"},verify=False)
features_loc = json.loads(local_feat.content.decode("utf-8"))
features_loc = pd.DataFrame.from_dict(features_loc).T

features_table = dash_table.DataTable(features_loc.to_dict('records'), [{"name": l, "id": l} for l in features_loc.columns],style_cell={'textAlign': 'left'},
                                   id='table-feat',)

app.layout = html.Div(children=[
    html.H1("Prêt à dépenser", style={'textAlign': 'center'}),
    html.Div(''' Id Client '''),
    html.Div(dcc.Dropdown(lst_id,'100001', id='dropdown')),
    html.Div(''' Scoring crédit '''),
    html.Div(dcc.Graph(id='target-graph')),
    html.Div(''' Global Feature importance '''),
    html.Div(dcc.Graph(id='feat-graph',figure = fig)),
    html.Div(''' Local Feature importance '''),
    html.Div(features_table),
    html.H1('''Les informations descriptives relatives au client''', style={'textAlign': 'center'}),
    html.Div(basic_table),
    html.H1('''Le top 10 des clients similaires''', style={'textAlign': 'center'}),
    html.Div(voisins_table),
])

@app.callback(
    Output('table-dropdown','data'),Input('dropdown','value'))
def load_infos_client(value):

    infos_clients = requests.get(URL_API + "infos_client",params={"id_client":value},verify=False)
    infos_clients = json.loads(infos_clients.content.decode("utf-8"))

    # On transforme le dictionnaire en dataframe
    infos_clients = pd.DataFrame.from_dict([infos_clients])
    
    return infos_clients.to_dict('records')

@app.callback(
    Output('target-graph','figure'),Input('dropdown','value'))
def load_prediction(value):
    
    # Requête permettant de récupérer la prédiction
    # de faillite du client sélectionné

    prediction = requests.get(URL_API + "predict", params={"id_client":value},verify=False)
    prediction = json.loads(prediction.content.decode("utf-8"))
    labels = ['Solvable','Non solvable']
    values = prediction
    return go.Figure(data=[go.Pie(labels=labels, values=values)])



@app.callback(
    Output('table-voisins','data'),Input('dropdown','value'))
def load_voisins(value):
    
    # Requête permettant de récupérer les 10 dossiers
    # les plus proches de l'ID client choisi

    voisins = requests.get(URL_API + "load_voisins", params={"id_client":value},verify=False)

    # On transforme la réponse en dictionnaire python
    voisins = json.loads(voisins.content.decode("utf-8"))
    
    # On transforme le dictionnaire en dataframe
    voisins = pd.DataFrame.from_dict(voisins).T
    voisins = voisins[["SK_ID_CURR","PREV_APPL_MEAN_POS_MEAN_CNT_INSTALMENT_FUTURE","CNT_CHILDREN","DAYS_BIRTH","DAYS_EMPLOYED",
                       "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"]]
    voisins["DAYS_BIRTH"] = (voisins["DAYS_BIRTH"] / -365).astype(int)
    voisins["DAYS_EMPLOYED"] = (voisins["DAYS_EMPLOYED"] / -365).astype(int)
    voisins["PREV_APPL_MEAN_POS_MEAN_CNT_INSTALMENT_FUTURE"] = voisins["PREV_APPL_MEAN_POS_MEAN_CNT_INSTALMENT_FUTURE"].astype(int)

    voisins.rename(columns={"SK_ID_CURR": "Id Client", 
                            "PREV_APPL_MEAN_POS_MEAN_CNT_INSTALMENT_FUTURE": "Moyenne des échéances restant dues des crédits antérieurs",
                            "CNT_CHILDREN" : "Nombre d'enfants", "DAYS_BIRTH" : "Age (ans)",
                            "DAYS_EMPLOYED" : "Ancienneté de l'emploi (ans)",
                            "AMT_INCOME_TOTAL" : "Revenu total",
                            "AMT_CREDIT" : "Montant du credit","AMT_ANNUITY" :"Annuites",
                            "AMT_GOODS_PRICE" : "Valeur du bien"}, inplace=True)
    
    
    return voisins.to_dict('records')

@app.callback(
    Output('table-feat','data'),Input('dropdown','value'))
def local_features(value):
    
    # Requête permettant de récupérer les features 

    local_feat = requests.get(URL_API + "load_localfeat", params={"id_client":value},verify=False)
    features_loc = json.loads(local_feat.content.decode("utf-8"))
    features_loc = pd.DataFrame.from_dict(features_loc).T
    return features_loc.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)
