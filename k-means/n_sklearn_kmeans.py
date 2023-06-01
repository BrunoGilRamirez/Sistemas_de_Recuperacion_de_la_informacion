import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import mode
dict_documentos = {
    "doc1": "Este es el primer documento",
    "doc2": "Este es el segundo documento",
    "doc3": "Este es el tercer documento",
    "doc4": "Este es el cuarto documento",
    "doc5": "Este es el quinto documento",
    "doc6": "El sexto documento es diferente",
    "doc7": "El séptimo documento es diferente",
    "doc8": "El octavo documento es diferente al sexto y séptimo"
}
class KMeans_DocsRecovery:
    def __init__(self,dict_documentos, k=3):
        self.dict_documentos=dict_documentos # Diccionario de documentos
        self.X,self.unique_words=self.vectorize_documents(dict_documentos) # Representación de documentos como vectores binarios
        self.k=k # Número de clústeres
    def addDoc(self, docs: dict): # Agregar un documento al diccionario de documentos
        self.dict_documentos = self.dict_documentos | docs # Unir los diccionarios de documentos
       
    def vectorize_documents(self,dict_documentos):
        # Representación de documentos como vectores binarios
        vectorizador = CountVectorizer(binary=True, stop_words='english')
        self.X= vectorizador.fit_transform(list(dict_documentos.values())) # Representación de documentos como vectores binarios
        self.X=self.X.toarray() # Convertir a array
        self.dict_Vectores = {}; i=0 # Diccionario de vectores
        for vector in self.X: # Recorrer los vectores
            self.dict_Vectores[list(dict_documentos.keys())[i]] = vector.tolist() # Agregar el vector al diccionario de vectores
            i+=1
        self.unique_words = vectorizador.get_feature_names_out() # Palabras únicas
        return self.X, self.unique_words # Devolver la representación de documentos como vectores binarios y las palabras únicas
    

    def k_medias(self, max_iter=100):
        # Inicialización aleatoria de los centroides
        self.centroides = self.X[np.random.choice(range(self.X.shape[0]), self.k, replace=False), :]
        
        # Bucle principal del algoritmo
        for _ in range(max_iter):
            # Asignación de cada vector al centroide más cercano
            self.distancias = np.linalg.norm(self.X[:, np.newaxis, :] - self.centroides, axis=-1) #:, np.newaxis, :] es para que se pueda restar el array de centroides a cada vector de X, np.linalg.norm es la norma euclidea, esto hace que se calcule la distancia de cada vector a cada centroide, distancias es un array de distancias, cada distancia corresponde a un cluster.
            self.etiquetas = np.argmin(self.distancias, axis=1)
            
            # Actualización de los centroides
            for i in range(self.k):
                self.centroides[i] = np.mean(self.X[self.etiquetas == i], axis=0)
        
        # Devolución de las etiquetas finales y los centroides finales
        return self.etiquetas, self.centroides

    def k_medias_mejorado(self, max_iter=100, tol=1e-4, seed=0):
        # Inicialización robusta de los centroides
        centroides = []
        np.random.seed(seed) #Esto hace que se fije la semilla para que los resultados sean reproducibles.
        centroides.append(self.X[np.random.choice(range(self.X.shape[0]), 1), :][0]) #Esto hace que se elija un vector aleatorio de X y se añada a centroides.
        #Tecnica de inicializacion k-means++
        for i in range(1, self.k):
            distancias = np.array([min([np.linalg.norm(x-c)**2 for c in centroides]) for x in self.X]) #Esto hace que se calcule la distancia de cada vector a cada centroide, distancias es un array de distancias, cada distancia corresponde a un vector.
            probs = distancias / distancias.sum() #Esto hace que se calcule la probabilidad de que cada vector sea un centroide, probs es un array de probabilidades, cada probabilidad corresponde a un vector.
            cumprobs = probs.cumsum() #Esto hace que se calcule la probabilidad acumulada de que cada vector sea un centroide, cumprobs es un array de probabilidades acumuladas, cada probabilidad acumulada corresponde a un vector.
            r = np.random.rand() #Esto hace que se elija un número aleatorio entre 0 y 1, r es un número aleatorio entre 0 y 1.
            for j, p in enumerate(cumprobs): #Este bucle hace que se elija un vector aleatorio de X y se añada a centroides basandonos en probs y r.
                if r < p: #Esto hace que se elija un vector aleatorio de X y se añada a centroides basandonos en probs y r.
                    centroides.append(self.X[j]) #Esto hace que se elija un vector aleatorio de X y se añada a centroides basandonos en probs y r.
                    break #Esto hace que se elija un vector aleatorio de X y se añada a centroides basandonos en probs y r.
        
        
        for j in range(max_iter): #Bucle principal del algoritmo
            # Asignación de cada vector al centroide más cercano
            distancias = np.linalg.norm(self.X[:, np.newaxis, :] - centroides, axis=-1) #np.linalg.norm es la norma euclidea, esto hace que se calcule la distancia de cada vector a cada centroide, distancias es un array de distancias, cada distancia corresponde a un cluster.
            etiquetas = np.argmin(distancias, axis=1) #np.argmin devuelve el indice del valor minimo de un array, Etiqutas es un array de indices, cada indice corresponde a un cluster.
            
            # Actualización de los centroides
            nuevos_centroides = np.zeros_like(centroides) #np.zeros_like devuelve un array de ceros con la misma forma y tipo que el array de entrada, esto hace que se cree un array de ceros con la misma forma y tipo que centroides.
            for i in range(self.k): #Este bucle hace que se calcule el centroide de cada cluster basandonos en X[etiquetas == i]
                nuevos_centroides[i] = np.mean(self.X[etiquetas == i], axis=0) #np.mean es la media aritmética, esto hace que se calcule el centroide de cada cluster basandonos en X[etiquetas == i]
            
            # Verificación de convergencia temprana
            if np.allclose(centroides, nuevos_centroides, rtol=0, atol=tol): #np.allclose devuelve True si dos arrays son iguales dentro de una tolerancia, rtol es la tolerancia relativa y atol es la tolerancia absoluta, esto hace que se compruebe si los centroides y los nuevos centroides son iguales dentro de una tolerancia.
                print(f"Convergencia temprana en la iteración {j}") 
                break
            centroides = nuevos_centroides #Esto hace que se actualicen los centroides.
        
        # Devolución de las etiquetas finales y los centroides finales
        return etiquetas, centroides


    def k_mode(self, max_iter=100, tol=1e-4,seed=0): #este algoritmo es el mismo que el anterior, solo cambia np.linalg.norm por scipy.stats.mode, que calcula la moda de un array.
        # Inicialización robusta de los centroides
        centroides = []
        np.random.seed(seed) #Esto hace que se fije la semilla para que los resultados sean reproducibles.
        centroides.append(self.X[np.random.choice(range(self.X.shape[0]), 1), :][0])
        #Tecnica de inicializacion k-means++
        for i in range(1, self.k):
            distancias = np.array([min([np.linalg.norm(x-c)**2 for c in centroides]) for x in self.X])
            probs = distancias / distancias.sum()
            cumprobs = probs.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cumprobs):
                if r < p:
                    centroides.append(self.X[j])
                    break
        
        # Bucle principal del algoritmo
        for j in range(max_iter):
            # Asignación de cada vector al centroide más cercano
            distancias = np.linalg.norm(self.X[:, np.newaxis, :] - centroides, axis=-1)
            etiquetas = np.argmin(distancias, axis=1)
            
            # Actualización de los centroides
            nuevos_centroides = np.zeros_like(centroides)
            for i in range(self.k):
                nuevos_centroides[i] = mode(self.X[etiquetas == i], axis=0,keepdims=True).mode[0]
            
            # Verificación de convergencia temprana
            if np.allclose(centroides, nuevos_centroides, rtol=0, atol=tol):
                print(f"Convergencia temprana en la iteración {j}")
                break
            centroides = nuevos_centroides
        
        # Devolución de las etiquetas finales y los centroides finales
        return etiquetas, centroides


    def k_mean_fit(self, Algoritmo='k_medias_mejorado', max_iter=100, tol=1e-4, seed=0): #Este metodo hace que se ejecute el algoritmo elegido.
        
        
        if Algoritmo == 'k_medias_mejorado': #Este if hace que se ejecute el algoritmo elegido.
            self.etiquetas, self.centroides = self.k_medias_mejorado( max_iter, tol, seed)
        elif Algoritmo == 'k_medias':
            self.etiquetas, self.centroides = self.k_medias( max_iter)
        elif Algoritmo == 'k_mode':
            self.etiquetas, self.centroides = self.k_mode( max_iter, tol, seed)

        # Impresión de los clústeres
        dict_cluster={}
        for i in range(self.k): #Este bucle hace que se cree un diccionario con los clusters y sus vectores.
            indices = self.etiquetas == i #Este bucle hace que se cree un diccionario con los clusters y sus vectores.
            vectores_cluster = self.X[indices] #Esto hace que se cree un diccionario con los clusters y sus vectores.
            dict_cluster[i] = vectores_cluster.tolist() #Esto hace que se cree un diccionario con los clusters y sus vectores.

        for name,clusters in list(dict_cluster.items()): #Este bucle hace que se imprima el cluster y sus vectores.
            print(f"\nClúster {name}: valores: {clusters}") 
            for cluster in clusters: 
                for key,value in list(self.dict_Vectores.items()): 
                    if value == cluster: 
                        print(f"Documento: {self.dict_documentos[key]}") 
        
    def query_k_mean(self, query, Algoritmo='k_medias_mejorado', max_iter=100, tol=1e-4, seed=0):
        self.dict_documentos['query'] = query
        self.vectorize_documents(self.dict_documentos)
        self.k_mean_fit(Algoritmo, max_iter, tol, seed)
    def asignar_cluster(self, vector_nuevo):
        distancias = np.linalg.norm(vector_nuevo - self.centroides, axis=1)
        cluster = np.argmin(distancias)
        return cluster
    