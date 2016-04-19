# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 08:42:54 2016

@author: canalli
"""

from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
#Pretty printer, imprime matrizes de forma mais legível
from pprint import pprint

def matriz_termo_documento(vetorizador, documentos):
	# Aqui o vetorizador identifica todas as palavras que vai precisar e o tamanho da matriz resultante (#documentos x #palavras)
	vetorizador.fit(documentos)
	
	# Aqui ele preenche as frequencias 
	matriz_termo_documento = vetorizador.transform(documentos)
	
	print("\nVocabulário aprendido:")
	pprint(vetorizador.vocabulary_)
	
	# O scikit-learn coloca num formato de matriz esparça (armazena apenas valores não nulos) para economizar espaço
	# Em um caso real isso é importante porque a matriz resultante não caberia em nenhuma memória
	# o método toarray() transforma a matriz esparça numa matriz convencional, para fins didáticos
	return matriz_termo_documento.toarray()
	
def vetoriza_queries(vetorizador, queries):
	queries_vetorizada = vetorizador.transform(queries)
	return queries_vetorizada.toarray()
	
# Produto interno de dois vetores quaisquer
def produto_interno(u, v):
	acc = 0
	n = len(u)
	m = len(v)
	
	if n != m:
		print("Vetores com dimensões distintas! Imporssível realizar produto interno.")
		return None
		
	for i in range(n):
		acc += u[i] * v[i]
		
	# Valor absoluto (sem sinal) do somatório
	return abs(acc)

# Norma euclidiana, ou L2, dada pela raiz da soma dos quadrados de um vetor. Mede o tamanho de um vetor
def norma_euclidiana(u):
	acc = 0
	n = len(u)
	# realiza a soma dos quadrados
	for i in range(n):
		acc += u[i] ** 2
		
	
	return sqrt(acc)
		
def similaridade_do_coseno(d, q):
	return produto_interno(d, q)/ (norma_euclidiana(d) * norma_euclidiana(q))
	
def realiza_consulta(matriz_termo_documento, q):
	similaridades = []
	
	# calcula similaridade da consulta com todos os documentos
	# neste formato de for numa matriz, cada item será uma linha
	for d in matriz_termo_documento:
		s = similaridade_do_coseno(d, q)
		similaridades.append(s)
		
	return similaridades
		
		
vetorizador = CountVectorizer()

#Cria uma lista dinâmica
documentos = []

documentos.append('Os céus declaram a glória de Deus e o firmamento anuncia a obra das suas mãos')
documentos.append('Os preceitos do Senhor são retos e alegram o coração o mandamento do Senhor é puro e ilumina os olhos')
documentos.append('Um dia faz declaração a outro dia e uma noite mostra sabedoria a outra noite')
documentos.append('O temor do Senhor é limpo, e permanece eternamente os juízos do Senhor são verdadeiros e justos juntamente')

matriz = matriz_termo_documento(vetorizador, documentos)
print("\nmatriz termo documento:")
pprint(matriz)

queries = []
queries.append('dia noite')
queries.append('temor do Senhor')


queries_vetorizadas = vetoriza_queries(vetorizador, queries)
for q in queries_vetorizadas:
	resultado = realiza_consulta(matriz, q)
	print("\nSimilaridades para query:")
	pprint(resultado)