"""Parte I - Análise estrutural obrigatória."""
import random

import numpy as np

from .algoritmos import bfs, componentes_conexas, tarjan_scc


# ----------------------------------------------------------------------
# Graus e densidade
# ----------------------------------------------------------------------
def _resumo(x):
    x = np.asarray(x)
    return {
        "min": int(x.min()),
        "max": int(x.max()),
        "media": float(x.mean()),
        "mediana": float(np.median(x)),
        "desvio_padrao": float(x.std(ddof=0)),
    }


def graus(g):
    """Vetores de grau (entrada, saída, total = ent+sai, não-dirigido) e resumos."""
    ent = np.array([len(a) for a in g.in_adj])
    sai = np.array([len(a) for a in g.out_adj])
    und = np.array([len(a) for a in g.und_adj])
    vetores = {"entrada": ent, "saida": sai, "total": ent + sai, "nao_dirigido": und}
    return vetores, {k: _resumo(v) for k, v in vetores.items()}


def distribuicao(vetor):
    """{grau: nº de vértices}, ordenado por grau."""
    valores, cont = np.unique(np.asarray(vetor), return_counts=True)
    return {int(k): int(c) for k, c in zip(valores, cont)}


def densidade(g):
    n = g.n
    return {
        "dirigida": g.m / (n * (n - 1)),
        "nao_dirigida": 2 * g.m_und / (n * (n - 1)),
    }


# ----------------------------------------------------------------------
# Componentes
# ----------------------------------------------------------------------
def componentes(g):
    rot_w, tam_w = componentes_conexas(g.und_adj)
    rot_s, k_s = tarjan_scc(g.out_adj)
    tam_s = np.bincount(rot_s, minlength=k_s).tolist()
    maior_w = int(np.argmax(tam_w))
    maior_s = int(np.argmax(tam_s))
    return {
        "rot_fracas": rot_w,
        "rot_fortes": rot_s,
        "tamanhos_fracas": sorted(tam_w, reverse=True),
        "tamanhos_fortes": sorted(tam_s, reverse=True),
        "n_fracas": len(tam_w),
        "n_fortes": k_s,
        "vertices_maior_fraca": [v for v in range(g.n) if rot_w[v] == maior_w],
        "vertices_maior_forte": [v for v in range(g.n) if rot_s[v] == maior_s],
    }


# ----------------------------------------------------------------------
# Distâncias: diâmetro, raio, comprimento médio, diâmetro efetivo
# ----------------------------------------------------------------------
def _diametro_efetivo(hist, q=0.9):
    """Menor d tal que a fração q dos pares está a distância <= d (interpolação linear,
    como no SNAP). `hist[d]` = nº de pares a distância d (d >= 1)."""
    total = hist[1:].sum()
    if total == 0:
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(hist[1:]) / total])  # cum[d] = P(dist <= d)
    d = int(np.argmax(cum >= q))
    if d == 0:
        return 0.0
    return float(d - 1 + (q - cum[d - 1]) / (cum[d] - cum[d - 1]))


def estatisticas_caminhos(adj, fontes, mascara_alvos=None, max_fontes=None, semente=0):
    """BFS a partir de cada fonte.

    - Considera apenas pares (s, t) com t alcançável a partir de s (t != s).
    - `mascara_alvos` (array bool de tamanho n) restringe os destinos (usado p/ a maior SCC).
    - `max_fontes`: se dado e menor que len(fontes), usa amostra aleatória (resultado aproximado).
    """
    n = len(adj)
    fontes = list(fontes)
    exato = True
    if max_fontes is not None and len(fontes) > max_fontes:
        fontes = random.Random(semente).sample(fontes, max_fontes)
        exato = False
    hist = np.zeros(n + 1, dtype=np.int64)
    ecc = []
    for s in fontes:
        d = np.asarray(bfs(adj, s))
        if mascara_alvos is not None:
            d = d[mascara_alvos]
        d = d[d > 0]
        if d.size == 0:
            ecc.append(0)
            continue
        ecc.append(int(d.max()))
        hist += np.bincount(d, minlength=n + 1)[: n + 1]
    pares = int(hist.sum())
    soma = int((np.arange(n + 1) * hist).sum())
    return {
        "fontes_usadas": len(fontes),
        "exato": exato,
        "pares_alcancaveis": pares,
        "diametro": int(max(ecc)) if ecc else 0,
        "raio": int(min(ecc)) if (ecc and exato) else None,
        "comprimento_medio": soma / pares if pares else float("nan"),
        "diametro_efetivo_90": _diametro_efetivo(hist),
        "histograma_distancias": {int(d): int(c) for d, c in enumerate(hist) if d > 0 and c > 0},
    }


def caminhos(g, comp, max_fontes=None):
    """Calcula distâncias em três contextos (ver README, seção 'Tratamento')."""
    n = g.n
    # (a) grafo dirigido, pares alcançáveis
    dirigido = estatisticas_caminhos(g.out_adj, range(n), max_fontes=max_fontes)
    dirigido["raio"] = None  # indefinido: vértices sem arestas de saída têm excentricidade 0 / grafo não é fortemente conexo
    # (b) versão não-dirigida, maior componente fracamente conexa
    vw = comp["vertices_maior_fraca"]
    nao_dirigido = estatisticas_caminhos(g.und_adj, vw, max_fontes=max_fontes)
    # (c) maior componente fortemente conexa (dirigida)
    vs = comp["vertices_maior_forte"]
    mask = np.zeros(n, dtype=bool)
    mask[vs] = True
    forte = estatisticas_caminhos(g.out_adj, vs, mascara_alvos=mask, max_fontes=max_fontes)
    return {"dirigido_alcancaveis": dirigido, "nao_dirigido_maior_WCC": nao_dirigido,
            "dirigido_maior_SCC": forte}


# ----------------------------------------------------------------------
# Triângulos e clustering (grafo não-dirigido)
# ----------------------------------------------------------------------
def triangulos_clustering(und_adj):
    """Triângulos, clustering local médio e transitividade (versão não-dirigida)."""
    n = len(und_adj)
    conj = [set(a) for a in und_adj]
    cont = [0] * n  # cada triângulo conta 2 em cada um dos seus 3 vértices
    for u in range(n):
        for v in und_adj[u]:
            if v > u:
                c = len(conj[u] & conj[v])
                cont[u] += c
                cont[v] += c
    triangulos = sum(cont) // 6
    c_local = np.zeros(n)
    for i in range(n):
        k = len(und_adj[i])
        if k >= 2:
            c_local[i] = cont[i] / (k * (k - 1))   # = 2 t_i / (k (k-1)), com t_i = cont[i]/2
    triplas = sum(len(a) * (len(a) - 1) // 2 for a in und_adj)
    return {
        "triangulos": int(triangulos),
        "clustering_medio_todos_vertices": float(c_local.mean()),
        "clustering_medio_grau_ge_2": float(c_local[[len(a) >= 2 for a in und_adj]].mean()),
        "transitividade_global": float(3 * triangulos / triplas) if triplas else 0.0,
        "clustering_local": c_local,
    }
