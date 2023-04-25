import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')

class BooleanRetrieval:
    def __init__(self, documents):
        self.documents = documents
        self.term_dict = {}
        self.unique_words = set()
        self.stop_words = set(stopwords.words('spanish'))
        self.stemmer = SnowballStemmer('spanish')
        self.index()

    def index(self):
        for i, document in enumerate(self.documents):
            self.index_document(i, document)

    def index_document(self, doc_id, document):
        """Indexa un documento"""
        # Tokenizar el documento y eliminar las palabras vacías
        words = word_tokenize(document.lower())
        words = [word for word in words if word not in self.stop_words]

        # Obtener el conjunto de palabras únicas y aplicar el stemmer
        unique_words = set([self.stemmer.stem(word) for word in words])
        self.unique_words.update(unique_words)

        # Crear un diccionario de términos para el documento actual
        term_dict = {}
        for word in unique_words:
            term_dict[word] = [i for i, val in enumerate(words) if self.stemmer.stem(val) == word]

        self.term_dict[doc_id] = term_dict

    def search(self, query):
        """Realiza una búsqueda booleana"""
        # Tokenizar la consulta y eliminar las palabras vacías
        query_words = word_tokenize(query.lower())
        query_words = [word for word in query_words if word not in self.stop_words]

        # Aplicar el stemmer a las palabras de la consulta
        query_words = [self.stemmer.stem(word) for word in query_words]

        relevant_docs = set(range(len(self.documents)))
        for i, word in enumerate(query_words):
            if i == 0 or query_words[i-1] != "not":
                doc_ids = set([i for i, val in self.term_dict.items() if word in val])
            else:
                doc_ids = set(range(len(self.documents))) - set([i for i, val in self.term_dict.items() if word in val])
            if not doc_ids:
                return []

            if i > 0 and query_words[i-1] == "and":
                relevant_docs = relevant_docs.intersection(doc_ids)
            elif i > 0 and query_words[i-1] == "or":
                relevant_docs = relevant_docs.union(doc_ids)
            else:
                relevant_docs = doc_ids

        # Verificar si los documentos relevantes contienen todos los términos de la consulta
        results = []
        for doc_id in relevant_docs:
            document_terms = self.term_dict[doc_id]
            doc_result = True
            for word in query_words:
                if word not in document_terms and word != "and" and word != "or" and word != "not":
                    doc_result = False
                    break
            if doc_result:
                results.append(doc_id)

        return results

def boolean_search(query, model):
    """Realiza una búsqueda booleana en el modelo especificado"""
    results = model.search(query)
    return results
documents = [
    "El perro marrón saltó sobre el gato perezoso.",
    "El gato perezoso se durmió en la alfombra.",
    "El perro marrón ladra al gato a menudo.",
    "El gato perezoso ignora al perro marrón."
]
r=boolean_search("perro AND gato", model = BooleanRetrieval(documents))
print(r)