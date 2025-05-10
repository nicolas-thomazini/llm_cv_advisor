import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


class Index:
    """
    Um índice de busca simples usando TF-IDF e similaridade de cosseno para campos de texto e correspondência exata para campos de palavras-chave.

    Atributos:
        text_fields (list): Lista de nomes de campos de texto a serem indexados.
        keyword_fields (list): Lista de nomes de campos de palavras-chave a serem indexados.
        vectorizers (dict): Dicionário de instâncias de TfidfVectorizer para cada campo de texto.
        keyword_df (pd.DataFrame): DataFrame contendo dados de campos de palavras-chave.
        text_matrices (dict): Dicionário de matrizes TF-IDF para cada campo de texto.
        docs (list): Lista de documentos indexados.
    """

    def __init__(self, text_fields, keyword_fields, vectorizer_params={}):
        """
        Inicializa o Índice com os campos de texto e palavras-chave especificados.

        Argumentos:
            text_fields (list): Lista de nomes de campos de texto a serem indexados.
            keyword_fields (list): Lista de nomes de campos de palavras-chave a serem indexados.
            vectorizer_params (dict): Parâmetros opcionais a serem passados ​​para TfidfVectorizer.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields

        self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in text_fields}
        self.keyword_df = None
        self.text_matrices = {}
        self.docs = []

    def fit(self, docs):
        """
        Ajusta o índice com os documentos fornecidos.

        Argumentos:
        docs (lista de dicionários): Lista de documentos a serem indexados. Cada documento é um dicionário.
        """

        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in docs]
            self.text_matrices[field] = self.vectorizers[field].fit_transform(texts)

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)

        return self

    def search(self, query, filter_dict={}, boost_dict={}, num_results=10):
        """
        Pesquisa o índice com a consulta, os filtros e os parâmetros de reforço fornecidos.

        Argumentos:
            query (str): A string da consulta de pesquisa.
            filter_dict (dict): Dicionário de campos de palavras-chave para filtrar. As chaves são nomes de campos e os valores são os valores para filtrar.
            boost_dict (dict): Dicionário de pontuações de reforço para campos de texto. As chaves são nomes de campos e os valores são as pontuações de reforço.
            num_results (int): O número de resultados principais a serem retornados. O padrão é 10.

        Retorna:
            list of dict: Lista de documentos que correspondem aos critérios de pesquisa, classificados por relevância.
        """

        query_vecs = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        scores = np.zeros(len(self.docs))

        # Compute cosine similarity for each text field and apply boost
        for field, query_vec in query_vecs.items():
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            scores += sim * boost

        # Apply keyword filters
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

        # Use argpartition to get top num_results indices
        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        # Filter out zero-score results
        top_docs = [self.docs[i] for i in top_indices if scores[i] > 0]

        return top_docs