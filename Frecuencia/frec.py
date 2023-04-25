import re
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from math import ceil
from pdfminer.high_level import extract_text
from matplotlib import pyplot as plt
class Analizer:

    def __init__(self):
        self.documentsets= {}
        self.dic = {}
        self.recordsets = {}
    
    def deploy_grafic(self):
        i=0
        for name,eset in list(self.recordsets.items()):
            plt.title(f"frecuencia de palabras en el archivo: {name}")
            plt.xlabel("Palabras")
            plt.ylabel("Veces que aparece en el texto")
            plt.xticks(rotation='vertical')
            for element in list(eset.items())[0:30]:
                #i=i+1
                #print(f"top {i}. elemento: {element[1]} frecuencia {element[0]}")
                plt.scatter(element[0], element[1])
            plt.show()

    def split_text(self,text):
        stop_words = set(stopwords.words('english'))
        words = re.split(r'\W+', text)
        filtered_words = []
        for w in words:
            if w not in stop_words:
                filtered_words.append(w)
        print(f"Palabras sin Stopwords {filtered_words[100:200]}")
        return filtered_words

    def get_setfrom_text(self,path='2591-0.txt', encoding="utf-8-sig"):
        filtered_words = []
        if ".txt" in path:
            with open(path, 'r') as text:
                text=text.read()
                filtered_words=self.split_text(text.lower())
        if ".pdf" in path:
            text = extract_text(path)
            filtered_words=self.split_text(text.lower())
        self.documentsets[path]=filtered_words

    def Frequency(self):
        for name,dataset in list(self.documentsets.items()):
            unique_words = set(dataset)
            dictonary={}
            if not(name in self.recordsets):
                for word in unique_words :
                    #print('Frequency of ', words , 'is :', filtered_words.count(words))
                    cont=dataset.count(word)
                    if  word in self.dic:
                        self.dic[word]+=cont
                    else:
                        self.dic[word]=cont
                    dictonary[word]=cont
                self.dic= dict(sorted(self.dic.items(),reverse=True,key=lambda x:x[1]))
                dictonary= dict(sorted(dictonary.items(),reverse=True,key=lambda x:x[1]))
                self.recordsets[name]=dictonary

    def inverse_document_Frec (self,frec, ndoc):
        None;
    def get_normalized (self,set):
        for element in set[0:30]:
            None
        #i=i+1
        #print(f"top {i}. elemento: {element[1]} frecuencia {element[0]}")