import streamlit as st
# Configuración de página DEBE ser lo primero
st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*missing ScriptRunContext.*")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.title("Dashboard de Análisis de Rotación de Empleados")

# 1. Carga y Preprocesamiento de Datos
st.header("1. Preprocesamiento de Datos")

# Cargar datos
uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    
    # Mostrar datos originales
    with st.expander("Datos Originales", expanded=True):
        st.write(f"Registros originales: {df.shape[0]}, Columnas originales: {df.shape[1]}")
        st.dataframe(df.head())

    # Eliminar filas con valores nulos
    df_cleaned = df.dropna()
    
    # Convertir variable objetivo
    df_cleaned['Attrition'] = df_cleaned['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Eliminar columnas irrelevantes
    st.subheader("Selección de Características")
    columns_to_drop = st.multiselect(
        "Seleccionar columnas a eliminar", 
        df_cleaned.columns,
        default=[]
    )
    if columns_to_drop:
        df_cleaned = df_cleaned.drop(columns=columns_to_drop)
    
    # Mostrar datos procesados
    with st.expander("Datos Procesados", expanded=False):
        st.write(f"Registros después de limpieza: {df_cleaned.shape[0]}, Columnas: {df_cleaned.shape[1]}")
        st.dataframe(df_cleaned.head())
    
    # Separar características y variable objetivo
    X = df_cleaned.drop('Attrition', axis=1)
    y = df_cleaned['Attrition']
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Visualización de Datos
    st.header("2. Análisis Exploratorio")
    
    # Histogramas
    st.subheader("Distribución de Variables")
    fig, ax = plt.subplots(figsize=(10, 8))
    X.hist(bins=20, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Correlación
    st.subheader("Matriz de Correlación")
    corr_matrix = X.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    
    # 3. Algoritmos de Clusterización
    st.header("3. Clusterización")
    
    # Selección de algoritmo
    clustering_algorithm = st.selectbox(
        "Seleccionar algoritmo de clusterización",
        ("K-Means", "DBSCAN"),
        key="cluster_algo"
    )
    
    # K-Means
    if clustering_algorithm == "K-Means":
        # Selección de clusters
        n_clusters = st.slider("Número de clusters", 2, 10, 3, key="kmeans_clusters")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Silhouette Score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        st.success(f"Silhouette Score: `{silhouette_avg:.4f}`")
        
        # Visualización con PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax.set_title("Clusters (PCA)")
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    
    # DBSCAN
    elif clustering_algorithm == "DBSCAN":
        eps = st.slider("EPS", 0.1, 2.0, 0.5, step=0.1, key="dbscan_eps")
        min_samples = st.slider("Mínimo de muestras", 2, 20, 5, key="dbscan_min_samples")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        # Información de clusters
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        st.info(f"Clusters detectados: {n_clusters}")
        
        # Silhouette Score (solo si hay más de 1 cluster)
        if len(unique_clusters) > 1:
            silhouette_avg = silhouette_score(X_scaled, clusters)
            st.success(f"Silhouette Score: `{silhouette_avg:.4f}`")
        else:
            st.warning("Silhouette Score no disponible (solo 1 cluster detectado)")
        
        # Visualización con t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax.set_title("Clusters (t-SNE)")
        ax.set_xlabel("Dimensión 1")
        ax.set_ylabel("Dimensión 2")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    
    # 4. Algoritmos de Clasificación
    st.header("4. Clasificación")
    
    # División de datos
    test_size = st.slider("Tamaño de prueba (%)", 10, 40, 20, key="test_size")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=test_size/100, 
        random_state=42,
        stratify=y
    )
    
    # Selección de modelo
    classifier = st.selectbox(
        "Seleccionar clasificador",
        ("Random Forest", "Logistic Regression"),
        key="classifier"
    )
    
    # Random Forest
    if classifier == "Random Forest":
        n_estimators = st.slider("Número de árboles", 50, 300, 100, key="rf_estimators")
        max_depth = st.slider("Profundidad máxima", 2, 20, 5, key="rf_depth")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced'
        )
    
    # Logistic Regression
    elif classifier == "Logistic Regression":
        C = st.slider("Parámetro de regularización (C)", 0.01, 10.0, 1.0, key="lr_c")
        
        model = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
    
    # Entrenamiento y evaluación
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas de evaluación
    st.subheader("Métricas de Evaluación")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Matriz de confusión
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusión')
    st.pyplot(fig)
    plt.close(fig)
    
    # Curva ROC
    if classifier == "Logistic Regression":
        st.subheader("Curva ROC")
        from sklearn.metrics import RocCurveDisplay
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle='--')
        st.pyplot(fig)
        plt.close(fig)
    
else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis")

# Instrucciones para ejecutar
st.sidebar.header("Instrucciones")
st.sidebar.markdown("""
1. Sube un archivo CSV con datos de empleados
2. Selecciona columnas a eliminar (opcional)
3. Explora las visualizaciones
4. Configura y ejecuta algoritmos de clusterización
5. Configura y ejecuta algoritmos de clasificación
""")


# In[ ]:





# In[ ]:




