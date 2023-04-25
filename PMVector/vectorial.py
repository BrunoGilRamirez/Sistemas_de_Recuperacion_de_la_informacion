from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class ModeloVectorial:
    corpus = None
    term_document_matrix = None
    feature_names = None
    rep_dict = None
    def __init__(self, corpus):
        self.corpus = corpus #diccionario de documentos
        self.term_document_matrix, self.feature_names = self.build_term_document_matrix() #se construye la matriz de representaciones vectoriales
        self.Representative_dict() #se construye el diccionario de representaciones vectoriales


    # Preprocesamiento de texto y construcción de la matriz término-documento
    def build_term_document_matrix(self):
        # Inicializar el vectorizador y tokenizar el texto
        #CountVectorizer es un objeto que permite convertir una coleccion de documentos de texto en una matriz de representaciones vectoriales
        vectorizer = CountVectorizer() #se crea el objeto vectorizador
        term_document_matrix = vectorizer.fit_transform(self.corpus.values())#se construye la matriz de representaciones vectoriales
        return term_document_matrix, vectorizer.get_feature_names_out()#se retorna la matriz y el diccionario de palabras y sus indices

    # Agregar documentos al corpus y reconstruir la matriz término-documento
    def add_documents(self, new_documents):
        self.corpus.update(new_documents)#se actualiza el diccionario de documentos
        self.term_document_matrix, self.feature_names = self.build_term_document_matrix()#se reconstruye la matriz
        self.Representative_dict()#se reconstruye el diccionario de representaciones vectoriales

    # Recuperación de información a partir de una consulta
    def retrieve_documents(self, query):
        # Construir el vector de la consulta
        #vectorizer es el objeto que contiene el diccionario de palabras y sus indices
        vectorizer = CountVectorizer(vocabulary=self.feature_names)#vocabulary es el diccionario de palabras y sus indices
        query_vector = vectorizer.transform([query]).toarray()#query_vector es la matriz de la consulta 
        # Calcular la similitud del coseno entre la consulta y los documentos
        cosine_similarities = self.cosine_similarity_explicit(query_vector, self.term_document_matrix.toarray())#toarray() es igual a .todense() pero mas eficiente

        # Ordenar los resultados por similitud descendente
        results = [(idx, cosine_similarities[idx]) for idx in cosine_similarities.argsort()[::-1]]#argsort() es igual a sort() pero devuelve los indices de los elementos ordenados
        results = {list(self.corpus.keys())[idx]: score for idx, score in results if score > 0} # Eliminar los documentos con similitud 0
        return results
    
    
    def Representative_dict(self):
        #Funcion que construye un diccionario con los documentos y sus representaciones vectoriales
        self.rep_dict = {}#diccionario de representaciones vectoriales
        j=0
        for rep  in self.term_document_matrix.toarray(): #recorre la matriz de representaciones vectoriales
            i = 0
            aux ={} 
            for score in rep: #recorre las representaciones vectoriales de cada documento
                aux[self.feature_names[i]] = score #diccionario con las palabras y sus frecuencias
                i += 1#contador de palabras
            self.rep_dict[list(self.corpus.keys())[j]] = aux#diccionario con los documentos y sus representaciones vectoriales
            j += 1#contador de documentos

    def cosine_similarity_explicit(self,query_vector, term_document_matrix):
        # Calcular el producto punto entre la consulta y los documentos
        #es decir np.dot(query_vector, term_document_matrix.T) es igual a np.sum(query_vector * term_document_matrix, axis=1)
        #.T es igual a .transpose() para transponer la matriz
        # pero la primera es mas eficiente  
        dot_products = np.dot(query_vector, term_document_matrix.T)#se calcula el producto punto entre la consulta y los documentos

        # Calcular las normas de la consulta y los documentos
        #np.linalg.norm(query_vector) es igual a np.sqrt(np.sum(query_vector**2)) o la norma euclidiana
        query_norm = np.linalg.norm(query_vector) #se calcula la norma de la consulta
        #np.linalg.norm(term_document_matrix, axis=1) es igual a np.sqrt(np.sum(term_document_matrix**2, axis=1)) o la norma euclidiana
        doc_norms = np.linalg.norm(term_document_matrix, axis=1) #se calcula la norma de los documentos axis=1 es para que calcule la norma de cada fila

        # Calcular la similitud del coseno entre la consulta y los documentos
        #np.divide(dot_products, np.outer(query_norm, doc_norms)) es igual a dot_products / (query_norm * doc_norms)
        # y np.outer(query_norm, doc_norms) es igual a query_norm * doc_norms 
        #ademas 1e-10 es un numero muy pequeño para evitar la division entre 0
        cosine_similarities = np.divide(dot_products, np.outer(query_norm, doc_norms) + 1e-10) #se calcula la similitud del coseno

        return cosine_similarities.flatten() #se retorna la matriz de similitudes


