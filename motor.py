

import re
import math

from collections import Counter


class MotorRecuperacion:

    def __init__(self, datos):

        self.indice = datos["indice"]
        self.longitudes = datos["longitudes"]
        self.dfreq = datos["dfreq"]

        self.tfidf_docs = datos["tfidf_docs"]
        self.normas_docs = datos["normas_docs"]

        self.documentos_info = datos["documentos_info"]

        self.N = datos["N"]
        self.promedio_longitud = datos["promedio_longitud"]

        self.stop_words = set(datos["stop_words"])

        self.k1 = datos["k1"]
        self.b = datos["b"]

        self.vocabulario = set(self.indice.keys())

    # --------------------------------
    # PREPROCESAMIENTO
    # --------------------------------

    def procesar_texto(self, texto):

        tokens = re.findall(
            r"[a-z0-9]+(?:['-][a-z0-9]+)*",
            texto.lower().replace("’", "'")
        )

        tokens = [
            t for t in tokens
            if t not in self.stop_words
        ]

        from nltk.stem import SnowballStemmer

        stemmer = SnowballStemmer("english")

        return [stemmer.stem(t) for t in tokens]

    # --------------------------------
    # TF-IDF
    # --------------------------------

    def calcular_idf(self, termino):

        return math.log(
            (self.N + 1) / (self.dfreq[termino] + 1)
        ) + 1

    def buscar_tfidf(self, query_tokens, top_k=10):

        frecuencias = Counter(query_tokens)

        vector_query = {}

        for termino, tf in frecuencias.items():

            if termino in self.vocabulario:

                vector_query[termino] = (
                    (1 + math.log(tf)) *
                    self.calcular_idf(termino)
                )

        norma_query = math.sqrt(
            sum(p ** 2 for p in vector_query.values())
        )

        if norma_query == 0:
            return []

        candidatos = set()

        for termino in vector_query:

            candidatos.update(
                self.indice[termino].keys()
            )

        resultados = []

        for docid in candidatos:

            producto = sum(
                peso_query *
                self.tfidf_docs[docid].get(termino, 0)

                for termino, peso_query in vector_query.items()
            )

            norma_doc = self.normas_docs[docid]

            if norma_doc > 0:

                similitud = producto / (
                    norma_query * norma_doc
                )

                resultados.append((docid, similitud))

        resultados.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return resultados[:top_k]

    # --------------------------------
    # BM25
    # --------------------------------

    def buscar_bm25(self, query_tokens, top_k=10):

        frecuencias_query = Counter(query_tokens)

        candidatos = set()

        for termino in frecuencias_query:

            if termino in self.indice:

                candidatos.update(
                    self.indice[termino].keys()
                )

        resultados = []

        for docid in candidatos:

            score = 0.0

            for termino, qf in frecuencias_query.items():

                if termino not in self.indice:
                    continue

                tf = self.indice[termino].get(docid, 0)

                if tf == 0:
                    continue

                df = self.dfreq[termino]

                idf = math.log(
                    1 + (
                        self.N - df + 0.5
                    ) / (
                        df + 0.5
                    )
                )

                numerador = tf * (self.k1 + 1)

                denominador = tf + self.k1 * (
                    1 - self.b +
                    self.b * self.longitudes[docid] /
                    self.promedio_longitud
                )

                score += idf * (
                    numerador / denominador
                ) * qf

            if score > 0:

                resultados.append((docid, score))

        resultados.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return resultados[:top_k]

    # --------------------------------
    # MODELO HIBRIDO
    # --------------------------------

    def buscar_hibrido(
        self,
        query_tokens,
        top_k=10,
        alpha=0.5
    ):

        tfidf = dict(
            self.buscar_tfidf(query_tokens, self.N)
        )

        bm25 = dict(
            self.buscar_bm25(query_tokens, self.N)
        )

        max_tfidf = max(tfidf.values(), default=0)
        max_bm25 = max(bm25.values(), default=0)

        candidatos = set(tfidf) | set(bm25)

        resultados = []

        for docid in candidatos:

            score_tfidf = (
                tfidf.get(docid, 0) / max_tfidf
                if max_tfidf > 0 else 0
            )

            score_bm25 = (
                bm25.get(docid, 0) / max_bm25
                if max_bm25 > 0 else 0
            )

            score = (
                alpha * score_tfidf +
                (1 - alpha) * score_bm25
            )

            resultados.append((docid, score))

        resultados.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return resultados[:top_k]

    # --------------------------------
    # BUSQUEDA GENERAL
    # --------------------------------

    def buscar(
        self,
        query,
        modelo="bm25",
        top_k=10,
        alpha=0.5
    ):

        tokens = self.procesar_texto(query)

        if modelo == "tfidf":

            resultados = self.buscar_tfidf(
                tokens, top_k
            )

        elif modelo == "bm25":

            resultados = self.buscar_bm25(
                tokens, top_k
            )

        elif modelo == "hibrido":

            resultados = self.buscar_hibrido(
                tokens, top_k, alpha
            )

        else:

            raise ValueError("Modelo no válido")

        respuesta = []

        for posicion, (docid, score) in enumerate(
            resultados, 1
        ):

            documento = self.documentos_info.get(
                docid, {}
            )

            respuesta.append({

                "rank": posicion,
                "document_id": docid,
                "title": documento.get("title", ""),
                "text": documento.get("text", ""),
                "score": round(float(score), 6)

            })

        return respuesta

