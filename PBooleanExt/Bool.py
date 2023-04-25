import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from math import ceil
from pdfminer.high_level import extract_text
import math
nltk.download('stopwords', quiet=True)

documents = {
    "d1":"El perro marrón saltó sobre el Pato perezoso.",
    "d2":"El gato perezoso se durmió en la alfombra.",
    "d3":"El perro marrón ladra al gato a menudo.",
    "d4":"El gato perezoso ignora al perro marrón."
}
class boolean: 
    d = None
    documents = None
    term_dict = None
    def __init__(self):
        self.d= []
        self.documents = {}
        self.stop_words = set(stopwords.words('spanish'))
        self.term_dict = {}

    def get_textfrom_file(self,path='2591-0.txt', encoding="utf-8-sig"):
        filtered_words = []
        if ".txt" in path:
            with open(path, 'r') as text:
                text=text.read()
                filtered_words=self.split_text(text.lower())
        if ".pdf" in path:
            text = extract_text(path)
            filtered_words=self.split_text(text.lower())
        self.documents[path]=filtered_words
    def addDoc(self, docs: dict):
        self.documents = self.documents | docs
        
    def tokenize (self):
        # Crear un conjunto de palabras clave únicas
        for name,string in list(self.documents.items()):
            self.documents[name]=str.upper(string)
        print(f"\nDocumentos Procesados: {self.documents}\n")
        unique_words = set(word_tokenize(' '.join(self.documents.values()), language='spanish'))
        unique_words.remove(".")
        # Eliminar las palabras vacías
        unique_words = unique_words - self.stop_words
        print(f"\nPalabras unicas: {unique_words}\n")
        for word in unique_words:
    
            for name, text in list(self.documents.items()):
                if name not in self.term_dict:
                    self.term_dict[name] = {}
                self.term_dict[name][word] = self.ponderacion(word, text)
        #self.Print()
        self.sort()
        return self.term_dict 
    def sort(self):
        for name, pesos in list(self.term_dict.items()):
            self.term_dict[name]=dict(sorted(pesos.items(), key=lambda item: item[1], reverse=True))
        self.Print()      
    
    def ponderacion (self,word, text:str):
        i=0
        j=0
        for name, document in list(self.documents.items()):
            if word in document:
                j=j+1
        s_Et=text.split()
        try:    
            s_Et.remove(".")
            s_Et=s_Et-self.stop_words
        except:
            pass
        for token in s_Et:
            if word==token:
                i=i+1
        if j>0:
            idf=math.log(len(self.documents)/j)+1
        else:
            idf=0
        peso= (idf)*(i/len(s_Et))
        return peso
    
    def Print(self):
        for name, pesos in list(self.term_dict.items()):
            print(f"\nDatos Procesados: \n{name}: {list(pesos.items())}\n")

    def boolean_search(self,query,relevant_docs=None):
        query_words = word_tokenize(str.upper(query))

        try: 
                query_words.remove("(")
                query_words.remove(")")
        except:
            pass
        query_words = [word for word in query_words if word not in self.stop_words]
        if relevant_docs is None:
            relevant_docs = set(self.documents.keys())
        else: 
            relevant_docs = set(relevant_docs)
        if "AND" in query_words:
            query_words.remove("AND")
            for word in query_words:
                doc_ids = set([i for i, val in list(self.term_dict.items())if val[word]])
                relevant_docs = relevant_docs.intersection(doc_ids)
            result= self.rank_docs("AND",query_words,relevant_docs)
        if "OR" in query_words:
            query_words.remove("OR")
            for word in query_words:
                doc_ids = set([i for i, val in list(self.term_dict.items())if val[word]])
                relevant_docs = relevant_docs.union(doc_ids)
            result= self.rank_docs("OR",query_words,relevant_docs)
        
        if "NOT" in query_words:
            query_words.remove("NOT")
            excluded_docs = set()
            for word in query_words:
                doc_ids = set([i for i, val in list(self.term_dict.items())if val[word]])
                excluded_docs = excluded_docs.union(doc_ids)
            relevant_docs =  relevant_docs - excluded_docs
            result= self.rank_docs("NOT",query_words,relevant_docs)

        result= dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

        return result
    
    def rank_docs(self,key, query_words,relevant_docs):
        
        scores = {}
        if key=="NOT":
            for doc in relevant_docs:
                scores[doc] = 0
                for word in query_words:
                    if self.term_dict[doc][word]>0:
                        scores[doc] = 0
                    elif self.term_dict[doc][word]==0:
                        scores[doc] = 100
        if key=="AND":
            for doc in relevant_docs:
                scores[doc] = 1
                for word in query_words:
                    scores[doc] = scores[doc] * (self.term_dict[doc][word]*100)
                    print (f"relevant_docs: {relevant_docs} {doc}: word: {word}score{scores[doc]}")
        if key=="OR":
            for doc in relevant_docs:
                scores[doc] = 0
                for word in query_words:
                    scores[doc] += (self.term_dict[doc][word]*100)
        return scores
    def process_query(self,query):
        query = str.upper(query)
        result = None
        query = query.replace("(", " ( ")
        query = query.replace(")", " ) ")
        tokens = query.split()
        stack = []
        for token in tokens:
            if token == "(":
                stack.append(token)
            elif token == "AND" or token == "OR":
                stack.append(token)
            elif token == "NOT":
                stack.append(token)
            else: 
                stack.append(token)
            if token == ")":
                sub_expr = []
                while len(stack)> 0:
                    top = stack.pop()
                    if top == "(": 
                        sub_expr.append(top)
                        break
                    else: sub_expr.append(top)
                sub_expr.reverse()
                print(f"sub: {sub_expr}")
                text= " ".join(sub_expr)
                print(f"text: {text}")
                result = self.boolean_search(text,result)
        return result


#Ejemplo de búsqueda
query = "Not perro"
extend = boolean()
extend.addDoc(documents)
print(f"\nDocumentos Originales {extend.documents}\n")
extend.tokenize()
results = extend.boolean_search(query)
print(f"\nConsulta {query}\n resultados{results}\n")  # salida: [0, 2]


query = "((not perro)and gato)"
results = extend.process_query(query)
print(f"\nConsulta {query}\n resultados{results}\n")

