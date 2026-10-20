import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# 1. CARGA DEL CONJUNTO DE DATOS IRIS

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in y]

# Primeras filas del conjunto de datos
print(df.head())

# Información estadística
print(df.describe())


# 2. FIGURA 1 - GRÁFICO DE DISPERSIÓN

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species"
)

plt.title("Relación entre longitud y anchura del pétalo")
plt.show()

# 3. FIGURA 2 - DIAGRAMA DE CAJA

sns.boxplot(
    data=df,
    x="species",
    y="petal length (cm)"
)

plt.title("Distribución de la longitud del pétalo por especie")
plt.show()

# 4. MODELO SUPERVISADO - SVC

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

modelo = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", C=1)
)

# Entrenamiento
modelo.fit(X_train, y_train)

# Predicciones
pred = modelo.predict(X_test)

# Resultados
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# 5. FIGURA 3 - MATRIZ DE CONFUSIÓN

ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    display_labels=iris.target_names
)

plt.title("Matriz de confusión - SVC")
plt.show()

# 6. VALIDACIÓN CRUZADA DEL MODELO SUPERVISADO

scores = cross_val_score(
    modelo,
    X,
    y,
    cv=5
)

print("Validación cruzada:", scores)
print("Media:", scores.mean())

# 7. MODELO NO SUPERVISADO - K-MEANS

escalador = StandardScaler()

X_escalado = escalador.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=0,
    n_init=10
)

clusters = kmeans.fit_predict(X_escalado)

# Resultados de K-Means
print("Etiquetas:", kmeans.labels_)
print("Centroides:", kmeans.cluster_centers_)
print("Inercia:", kmeans.inertia_)
print("Iteraciones:", kmeans.n_iter_)

# Coeficiente Silhouette
print(
    "Silhouette:",
    silhouette_score(X_escalado, clusters)
)

# 8. FIGURA 4 - MÉTODO DEL CODO

inercias = []

for k in range(1, 9):

    modelo_k = KMeans(
        n_clusters=k,
        random_state=0,
        n_init=10
    )

    modelo_k.fit(X_escalado)

    inercias.append(modelo_k.inertia_)


plt.plot(
    range(1, 9),
    inercias,
    marker="o"
)

plt.xlabel("Número de agrupamientos K")
plt.ylabel("Inercia")
plt.title("Método del codo para K-Means")

plt.show()

# 9. PCA PARA REDUCIR LOS DATOS A DOS DIMENSIONES

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_escalado)

# 10. FIGURA 5 - REPRESENTACIÓN DE LOS CLUSTERS K-MEANS

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title("K-Means sobre Iris")

plt.show()