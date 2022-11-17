# Pedro Bianchini de Quadros
from bs4 import BeautifulSoup
import requests
import spacy
import string
import numpy as np
'''
Enunciado:
1. Sua tarefa será gerar a matriz termo-documento usando TF-IDF por meio da aplicação das 
fórmulas  TF-IDF  na  matriz  termo-documento  criada  com  a  utilização  do  algoritmo  Bag of 
Words. Sobre o Corpus que recuperamos anteriormente. O entregável desta tarefa é uma 
matriz termo-documento onde a primeira linha são os termos e as linhas subsequentes são 
os vetores calculados com o TF-IDF. 
Sua tarefa será gerar uma matriz de distância, computando o cosseno do ângulo entre todos 
os vetores que encontramos usando o tf-idf. Para isso use a seguinte fórmula para o cálculo 
do  cosseno  use  a  fórmula  apresentada  em  Word2Vector  (frankalcantara.com) 
(https://frankalcantara.com/Aulas/Nlp/out/Aula4.html#/0/4/2)  e  apresentada  na  figura  a 
seguir:  
    <Imagem>
O resultado deste trabalho será uma matriz que relaciona cada um dos vetores já calculados 
com todos os outros vetores disponíveis na matriz termo-documento mostrando a distância 
entre cada um destes vetores. 
'''

# Declarando as funções que serão utilizadas no código
# corpus
def corpus():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('sentencizer')

    urls = [
        "https://www.ibm.com/cloud/learn/natural-language-processing",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://www.techtarget.com/searchenterpriseai/definition/natural-language-processing-NLP",
        "https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1",
        "https://www.tableau.com/learn/articles/natural-language-processing-examples"
    ]
    
    listaDeCorpus = []

    for url in urls:

        page = requests.get(url)

        if page.status_code != 200:
            continue

        soup = BeautifulSoup(page.content, 'html.parser')

        corpus = [] 
        for tagsParagrafo in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
            paragrafo = tagsParagrafo.get_text()
            stripParagrafo = paragrafo.strip()

            if len(stripParagrafo) > 0:
                documentos = nlp(stripParagrafo)

                for sentence in documentos.sents:
                    corpus.append(sentence.text.translate(str.maketrans('', '', string.punctuation)))

        listaDeCorpus.append(corpus)
        print(f"Número de palavras: {len(' '.join(corpus).split(' '))}")
        print(corpus)
    return listaDeCorpus

# bag of words
def bagOfWords(listaDeCorpus):
    # Bag of Words
    NumeroDeSentencas = 0
    for corpus in listaDeCorpus:

        for sentences in corpus:
            NumeroDeSentencas += 1

            for lex in sentences.split(' '):

                if lex not in lexemas:
                  lexemas.append(lex)

    bagOfWords = np.zeros((NumeroDeSentencas,len(lexemas)))

    sentencaAtual = 0
    for corpus in listaDeCorpus:

        for sentences in corpus:

            for lex in sentences.split(' '):
                bagOfWords[sentencaAtual][lexemas.index(lex)] += 1

            sentencaAtual += 1

    print("NÚMERO DE SENTENÇAS: " , NumeroDeSentencas)
    print(bagOfWords)
    return bagOfWords

#tf e idf
def tfIdf(listaDeCorpus, bOw):
    # TF
    tf = np.zeros((len(bOw),len(bOw[0])))
    sentencaAtual = 0
    for corpus in listaDeCorpus:

        for sentences in corpus:

            numeroDeLexemas = len(sentences.split(' '))
            for lexeme in sentences.split(' '):
                tf[sentencaAtual][lexemas.index(lexeme)] = bOw[sentencaAtual][lexemas.index(lexeme)] / numeroDeLexemas

            sentencaAtual += 1
    print("\nTF: ")
    print(tf)

    # IDF
    idfs = []
    for lex in range(len(lexemas)):
        lexRate = 0

        for sentenceRates in bOw:

            if sentenceRates[lex] > 0: 
                lexRate += 1
        
        idfs.append(np.log10(len(bOw)/lexRate))
    
    print("\nIDF: ")
    print(len(bOw))
    print(idfs)

    tfidf = np.zeros((len(bOw),len(bOw[0])))
    for i in range(len(tfidf)):

        for j in range(len(tfidf[0])):
            tfidf[i][j] = tf[i][j] * idfs[j]

    print('\nTF-IDF: ')
    print(tfidf)
    return tfidf

# cossine_similarity
def cossine_similarity():
    vetoresTfIdf = tfIdf(listaDeCorpus, bOw)
    vetorAtual = 0
    distanciasMatriz = np.zeros((len(vetoresTfIdf),len(vetoresTfIdf)))

    for vector in vetoresTfIdf:

        i = vetorAtual
        while i < len(vetoresTfIdf):

            # Aplicando o calculo do cossine similarity
            distancia_tfidf = np.dot(vector,vetoresTfIdf[i])/(np.linalg.norm(vector)*np.linalg.norm(vetoresTfIdf[i]))
            distanciasMatriz[vetorAtual][i] = distancia_tfidf
            distanciasMatriz[i][vetorAtual] = distancia_tfidf
            i += 1

        vetorAtual += 1
    print('\nDISTÂNCIAS ENTRE VETORES')
    print(distanciasMatriz)


lexemas = []
print("CORPUS")
listaDeCorpus = corpus()
print("\nBAG OF WORDS")
bOw = bagOfWords(listaDeCorpus)
print("\nTF-IDF E COSINE SIMILARITY")
cossine_similarity()