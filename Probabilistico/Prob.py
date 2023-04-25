import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from math import ceil
from pdfminer.high_level import extract_text
import math
import numpy as np
nltk.download('stopwords', quiet=True)

documents = {
    "d1":"El perro marrón saltó sobre el Pato perezoso.",
    "d2":"El gato perezoso se  en la alfombra.",
    "d3":"El perro marrón ladra al gato a menudo.",
    "d4":"El gato perezoso ignora al perro marrón."
}
class Probabilistico: 
    d = None # Lista de documentos
    documents = None # Diccionario de documentos
    term_dict = None # Diccionario de términos
    def __init__(self):
        self.d= [] # Lista de documentos
        self.documents = {} # Diccionario de documentos
        self.stop_words = set(stopwords.words('spanish')) # Lista de palabras vacías
        self.term_dict = {} # Diccionario de términos
        self.unique_words = set() # Conjunto de palabras únicas
        self.relevant_docs = []

    def get_textfrom_file(self,path='2591-0.txt', encoding="utf-8-sig"):
        filtered_words = [] # Lista de palabras filtradas
        if ".txt" in path: # Si el archivo es un .txt
            with open(path, 'r') as text: # Abrir el archivo
                text=text.read() # Leer el archivo
                filtered_words=self.split_text(text.lower())    # Filtrar el texto
        if ".pdf" in path: # Si el archivo es un .pdf
            text = extract_text(path) # Extraer el texto del archivo
            filtered_words=self.split_text(text.lower())    # Filtrar el texto
        self.documents[path]=filtered_words # Agregar el documento al diccionario de documentos
    def addDoc(self, docs: dict): # Agregar un documento al diccionario de documentos
        self.documents = self.documents | docs # Unir los diccionarios de documentos
    def tokenize (self):
        # Crear un conjunto de palabras clave únicas
        for name,string in list(self.documents.items()): # Recorrer el diccionario de documentos
            self.documents[name]=str.upper(string) # Convertir el texto a mayúsculas
        #print(f"\nDocumentos Procesados: {self.documents}\n") # Imprimir los documentos procesados
        unique_words = set(word_tokenize(' '.join(self.documents.values()), language='spanish')) # Crear un conjunto de palabras clave únicas
        unique_words.remove(".")    # Eliminar el punto
        # Eliminar las palabras vacías
        unique_words = unique_words - self.stop_words # Eliminar las palabras vacías
        self.unique_words = unique_words # Asignar el conjunto de palabras únicas
        print(f"\nPalabras unicas: {unique_words}\n") # Imprimir las palabras únicas
        for word in unique_words: # Recorrer el conjunto de palabras únicas
    
            for name, text in list(self.documents.items()): # Recorrer el diccionario de documentos
                if name not in self.term_dict: # Si el nombre del documento no está en el diccionario de términos
                    self.term_dict[name] = {} # Agregar el nombre del documento al diccionario de términos
                self.term_dict[name][word] = self.boolean_eval(word, text) # Agregar el término al diccionario de términos
        #self.Print()
        self.sort()
        return self.term_dict 
    
    def Print(self):  #Imprimir el diccionario de términos
        for name, pesos in list(self.term_dict.items()):
            print(f"\nDatos Procesados: \n{name}: {list(pesos.items())}\n")

    def sort(self): # Ordenar el diccionario de términos
        for name, pesos in list(self.term_dict.items()):
            self.term_dict[name]=dict(sorted(pesos.items(), key=lambda item: item[1], reverse=True))
        #self.Print()

    def boolean_eval (self, term, text): # Evaluar si el término está en el texto
            text=word_tokenize(text, language='spanish') # Tokenizar el texto
            text.remove(".") # Eliminar el punto
            if  term in text: # Si el término está en el texto
                return 1    # Retornar 1
            else:  # Si el término no está en el texto
                return 0   # Retornar 0
    
    def Importance_ofATerm(self,term): # Calcular la importancia de un término
        relevant_docs={}
        peso=0
        for name, pesos in list(self.term_dict.items()): # Recorrer el diccionario de términos
            if pesos.get(term)==1: # Si el término está en el documento
                peso+=pesos.get(term) # Sumar el peso del término
                relevant_docs[name] = peso   # Agregar el documento al diccionario de documentos relevantes
        return relevant_docs # Retornar el diccionario de documentos relevantes
    
    def Relevancy_calculation(self,termn,type): # Calcular la relevancia de un término
        if type=="r": # Si el tipo es relevante
            r=0;
            R_r=0
            for doc in self.relevant_docs: # Recorrer el diccionario de documentos relevantes
                if self.term_dict[doc].get(termn)==1: #si el termino esta en el documento
                    r+=1 #El termino si esta en los relevantes y se cuenta.
                else: #si el termino no esta en el documento
                    R_r+=1 #El termino no esta en los relevantes y se cuenta.
            return r,(r+R_r)
        if type=="nr": # Si el tipo es no relevante
            nr=0; 
            N_R_ntr=0
            no_relevants= self.documents.keys()-self.relevant_docs #no_relevants= documentos no relevantes
            for doc in no_relevants: # Recorrer el diccionario de documentos no relevantes
                if self.term_dict[doc].get(termn)==1: #si el termino esta en el documento
                    nr+=1 #El termino si esta en los no relevantes y se cuenta.
                else:
                    N_R_ntr+=1 #El termino no esta en los no relevantes y se cuenta. 
            return nr,(nr+N_R_ntr) #nr=numero de documentos relevantes que contienen el termino
            
    def P_ti_nr(self,termn,relevant_docs):
        #esta funcion calcula la probabilidad de que un termino no relevante este en un documento
        if len(self.relevant_docs)>0: #si hay documentos relevantes
            nr,Nra=self.Relevancy_calculation(termn, "nr") #nr=numero de documentos relevantes que contienen el termino
            q=nr/Nra #q=probabilidad de que un documento no relevante contenga el termino
            l_q=1-q #l_q=probabilidad de que un documento no relevante no contenga el termino
            return q,l_q #q=probabilidad de que un documento no relevante contenga el termino
        else: #si no hay documentos relevantes
            suma=self.suma(relevant_docs,termn) #suma=numero de documentos que contienen el termino
            prob_condicional=(suma/len(self.documents)) # prob_condicional=probabilidad de que un documento contenga el termino
            return prob_condicional # prob_condicional=probabilidad de que un documento contenga el termino
    
    def P_ti_r(self,termn,relevant_docs, incertidumbre=0.5):
        if len(self.relevant_docs)>0: #si hay documentos relevantes
            r,Ra=self.Relevancy_calculation(termn, "r") #r=numero de documentos relevantes que contienen el termino
            p=r/Ra #p=probabilidad de que un documento relevante contenga el termino
            l_p=1-p #l_p=probabilidad de que un documento relevante no contenga el termino
            return p,l_p #p=probabilidad de que un documento relevante contenga el termino
        else:
            prob_condicional=incertidumbre #prob_condicional=probabilidad de que un documento contenga el termino
            return prob_condicional #prob_condicional=probabilidad de que un documento contenga el termino
    
    def relevant_docs_from_a_survey(self,relevant_docs): #relevant_docs_from_a_survey es un diccionario de documentos relevantes
        self.relevant_docs=relevant_docs #relevant_docs es un diccionario de documentos relevantes y se guarda en la clase
        return relevant_docs #relevant_docs es un diccionario de documentos relevantes

    def Query_processing(self,query): #esta funcion procesa la consulta
        query=query.upper() #query=consulta en mayusculas
        query=word_tokenize(query, language='spanish') #query=consulta tokenizada
        try:
            query.remove(".") #query=consulta sin puntos
        except:
            pass

        for word in query: #recorrer la consulta
            relevant_docs=self.Importance_ofATerm(word) # se obtienen los documentos relevantes de la consulta
        conditional = {} #conditional es un diccionario vacio
        sem_di=0 #sem_di= similaridad del termino con el documento
        for word in query: #recorrer la consulta
            prob_condicional_nr = self.P_ti_nr(word,relevant_docs) #prob_condicional_nr es la probabilidad de que un termino  este en un documento no relevante
            prob_condicional_r = self.P_ti_r(word, relevant_docs) #prob_condicional_r es la probabilidad de que un termino este en un documento relevante
            if type(prob_condicional_nr)==tuple and type(prob_condicional_r)==tuple: #si los dos son tuplas
                q,l_q=prob_condicional_nr #prob_condicional_nr tiene el valor de incertidumbre
                p,l_p=prob_condicional_r #prob_condicional_r tiene el valor de incertidumbre
                sem_di=(1+(p*l_q))/(1+(q*l_p))
            else:
                sem_di=prob_condicional_r/prob_condicional_nr #prob_condicional_r tiene el valor de incertidumbre 
                #prob_condicional_nr es el numero de veces que aparece el termino en los documentos/numero de documentos de la coleccion
            result = {}
            for name in relevant_docs: #recorrer los documentos relevantes
                if name not in result: #si el documento no esta en el diccionario
                    result[name] = 0 #se agrega el documento al diccionario
                if self.term_dict[name].get(word)==1: #si el termino esta en el documento
                    result[name] = result[name]+ sem_di #se suma la similaridad del termino con el documento
            conditional[word] = result #se agrega el termino al diccionario
        nuevo_diccionario = self.union_resultados(conditional) #nuevo_diccionario es el diccionario de documentos ordenados por similaridad
        return nuevo_diccionario #nuevo_diccionario es el diccionario de documentos ordenados por similaridad
    
    def union_resultados(self,conditional): #esta funcion une los resultados de los terminos de la consulta
        nuevo_diccionario = {}
        for valores in conditional.values():
            for clave, valor in valores.items():
                if clave in nuevo_diccionario:
                    nuevo_diccionario[clave] += valor
                else:
                    nuevo_diccionario[clave] = valor
        nuevo_diccionario_ordenado = dict(sorted(nuevo_diccionario.items(), key=lambda item: item[1], reverse=False))
        return nuevo_diccionario_ordenado
    
    def suma(self,name,word): #esta funcion suma la similaridad de un termino con un documento
        suma=0
        for nam, pesos in list(self.term_dict.items()):
            if nam in name and pesos.get(word)==1:
                suma+=pesos.get(word)
        return suma


