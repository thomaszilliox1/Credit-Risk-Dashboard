import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import lightgbm as lgb
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle
from matplotlib import cm
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Arc
import shap

# Indicateur de risque du client
def degree_range(n):
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points

def rot_text(ang):
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation

def gauge(arrow=0.4, labels=['Faible', 'Modéré', 'Elevé', 'Très élevé'],
          title='', min_val=0, max_val=100, threshold=-1.0,
          colors='RdYlGn_r', n_colors=-1, ax=None, figsize=(2, 1.3)):
    N = len(labels)
    n_colors = n_colors if n_colors > 0 else N
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, n_colors)
        cmap = cmap(np.arange(n_colors))
        colors = cmap[::-1]
    if isinstance(colors, list):
        n_colors = len(colors)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ang_range, _ = degree_range(n_colors)

    for ang, c in zip(ang_range, colors):
        ax.add_patch(Wedge((0., 0.), .4, *ang, width=0.10, facecolor='w', lw=2, alpha=0.5))
        ax.add_patch(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=0, alpha=0.8))

    _, mid_points = degree_range(N)
    labels = labels[::-1]
    a = 0.45
    for mid, lab in zip(mid_points, labels):
        ax.text(a * np.cos(np.radians(mid)), a * np.sin(np.radians(mid)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=12, \
                fontweight=None, rotation=rot_text(mid))

    ax.add_patch(Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2))
    ax.text(0, -0.10, title, horizontalalignment='center', verticalalignment='center', fontsize=20, fontweight='bold')

    # Seuils de l'indicateur
    if threshold > min_val and threshold < max_val:
        pos = 180 * (max_val - threshold) / (max_val - min_val)
        a = 0.25;
        b = 0.18;
        x = np.cos(np.radians(pos));
        y = np.sin(np.radians(pos))
        ax.arrow(a * x, a * y, b * x, b * y, width=0.01, head_width=0.0, head_length=0, ls='--', fc='r', ec='r')

    # Flèche 
    pos = 180 - (180 * (max_val - arrow) / (max_val - min_val))
    pos_normalized = (arrow - min_val) / (max_val - min_val)
    angle_range = 180
    pos_degrees = angle_range * (1 - pos_normalized)

    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos_degrees)), 0.225 * np.sin(np.radians(pos_degrees)), \
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    return ax

# Streamlit (v1.25)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

# Logo
logo_path = 'logo.jpg'
st.sidebar.image(logo_path, use_column_width=True)

# Colonnes
categorical_columns = ['Type contrat',
                       'Genre',
                       'Diplôme etude supérieure',
                       'Ville travail différente ville_résidence']

# Modèle
# model_path = './saved_model/'
with open('LightGBM_smote_tuned.pckl', 'rb') as f:
    model = pickle.load(f)

# Data
df_feature_importance = pd.read_csv('df_feature_importance_25.csv')
df_feature_importance.drop('Unnamed: 0', axis=1, inplace=True)
df_dashboard_final = pd.read_csv('df_dashboard_final.csv')
df_dashboard_final.drop('Unnamed: 0', axis=1, inplace=True)
df_données_dashboard = pd.read_csv('données_dashboard.csv')

# Titre
st.title('Tableau de bord : risque client')

# Marge
st.sidebar.title('Client')
selected_client = st.sidebar.selectbox('Identifiant :', df_données_dashboard['ID client'])
predict_button = st.sidebar.button('Calculer risque')

import requests  

# URL de l'API Render
API_URL = "https://model-predict-risk-scoring.onrender.com/predict" 

if predict_button:
    st.subheader("Résultat de la prédiction via l'API Render")

    try:
        # Prépare la requête JSON avec l'identifiant client
        payload = {"SK_ID_CURR": int(selected_client)}
        response = requests.post(API_URL, json=payload)

        # Vérifie la réponse
        if response.status_code == 200:
            result = response.json()
            proba = result["PRED_PROBA"]
            pred = result["PRED_TARGET"]

            # Affiche les résultats
            st.success(f"Client {selected_client} → Risque : {'ÉLEVÉ' if pred == 1 else 'FAIBLE'}")
            st.metric("Probabilité de défaut", f"{proba*100:.1f} %")

            # Affiche la jauge avec le score API
            st.subheader("Indicateur de risque (API)")
            fig, ax = plt.subplots(figsize=(5, 3))
            gauge(arrow=proba*100, ax=ax)
            st.pyplot(fig)

        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"⚠️ Erreur de connexion à l'API : {e}")

# Index
index = df_données_dashboard[df_données_dashboard['ID client'] == selected_client].index[0]

# Affichage 
client_info = df_données_dashboard[df_données_dashboard['ID client'] == selected_client]
st.subheader('Informations sur le client :')
client_info.index = client_info['ID client']
st.write(client_info[['Prédiction crédit', 'Score client (sur 100)', 'Type contrat', 'Genre', 'Âge']])

# Prédiction
selected_client_cat = df_données_dashboard.loc[index, 'Prédiction crédit']

# DataFrame prédiction
df_customer = df_données_dashboard[df_données_dashboard['Prédiction crédit'] == selected_client_cat].copy()

# Affichage de la jauge score client
st.subheader('Niveau de risque :')
score = client_info['Score client (sur 100)'].values[0]  # Récupérer le score du client
fig, ax = plt.subplots(figsize=(5, 3))
gauge(arrow=score, ax=ax)  # Appeler la fonction gauge() en passant le score du client
st.pyplot(fig)

# Ecran principal
st.sidebar.title('Métriques')
univariate_options = [col for col in df_dashboard_final.columns if col not in ['ID client', 'Prédiction crédit']]
bivariate_options = [col for col in df_dashboard_final.columns if col not in ['ID client', 'Prédiction crédit']]

# Graphique univarié
univariate_feature = st.sidebar.selectbox('Variable univariée :', univariate_options)
df_customer.replace([np.inf, -np.inf], 0, inplace=True)
st.subheader('Analyse univariée (population restreinte) :')
plt.figure()
plt.hist(df_customer[univariate_feature], color='skyblue', label='Population')
plt.xlabel(univariate_feature)
plt.axvline(client_info[univariate_feature].values[0], color='salmon', linestyle='--', label='Client sélectionné')
plt.legend()
st.pyplot(plt.gcf())

# Graphique bivarié
bivariate_feature1 = st.sidebar.selectbox('Variable 1 (bivariée) :', bivariate_options)
bivariate_feature2 = st.sidebar.selectbox('Variable 2 (bivariée) :', bivariate_options)
st.subheader('Analyse bivariée (population complète) :')
plt.figure()
sns.scatterplot(data=df_dashboard_final, x=bivariate_feature1, y=bivariate_feature2,
                c=df_dashboard_final['Score client'], cmap='viridis',
                alpha=0.5, label='Population')
sns.scatterplot(data=client_info, x=bivariate_feature1, y=bivariate_feature2,
                color='salmon', marker='o', s=100, label='Client sélectionné')
plt.xlabel(bivariate_feature1)
plt.ylabel(bivariate_feature2)
plt.legend()
st.pyplot(plt.gcf())

# Graphique feature importance globale
df_sorted = df_feature_importance.sort_values('Features_importance_shapley', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Features_importance_shapley', y='Features', data=df_sorted, color='skyblue')
plt.xlabel('Importance SHAP')
plt.ylabel('Variable')
st.subheader('Importance des variables :')
st.pyplot(plt.gcf())
