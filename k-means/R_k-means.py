from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Datos de ejemplo
dict_documentos = {
    "doc1": "Este es el primer documento",
    "doc2": "Este es el segundo documento",
    "doc3": "Este es el tercer documento",
    "doc4": "Este es el cuarto documento",
    "doc5": "Este es el quinto documento",
    "doc6": "El sexto documento es diferente",
    "doc7": "El séptimo documento también es diferente",
    "doc8": "El octavo documento es diferente al sexto y séptimo"
}

# Representación de documentos como vectores binarios
vectorizador = CountVectorizer(binary=True, stop_words='english')
vectores = vectorizador.fit_transform(list(dict_documentos.values()))
dict_Vectores = {}
for vector in vectores.toarray().tolist():
    dict_Vectores[list(dict_documentos.keys())[vectores.toarray().tolist().index(vector)]] = vector
unique_words = vectorizador.get_feature_names_out()

# Aplicación del algoritmo k-means
k = 3  # número de clústeres
kmeans = KMeans(n_clusters=k,n_init=10, random_state=42).fit(vectores)

# Interpretación de los clústeres
documentos_cluster={}
for i in range(k):
    indices = kmeans.labels_ == i
    documentos_cluster[i] = [list(dict_documentos.values())[j] for j in range(len(dict_documentos)) if indices[j]]
print(f"Clústeres: {documentos_cluster}")

# Evaluación del modelo
cohesion = kmeans.inertia_  # medida de la cohesión intra-cluster
separacion = kmeans.score(vectores)  # medida de la separación inter-cluster
print(f"Cohesión: {cohesion}\n")
print(f"Separación: {separacion}\n")
print(f"Palabras únicas: {unique_words}")
print(f"vectores: {dict_Vectores}")
