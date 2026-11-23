"""Algoritmos da disciplina (Parte II), implementados à mão.

Convenções
  - Vértices são inteiros 0..n-1.
  - `adj` é lista de listas de inteiros (adjacência sem pesos).
  - `adj_w` é lista de listas de pares (v, peso).
  - Tudo é iterativo (sem recursão) para não estourar a pilha com n ~ 10^4.

Complexidades (n = |V|, m = |E|)
  BFS / DFS / Tarjan / componentes / Euler : O(n + m)
  Dijkstra (heap binário)                  : O((n + m) log n)
  Bellman-Ford                             : O(n m)   (pior caso)
  Floyd-Warshall                           : O(n^3)
  Kruskal                                  : O(m log m)
  Prim (heap binário)                      : O(m log n)
"""
import heapq
from collections import deque

import numpy as np

INF = float("inf")


# ----------------------------------------------------------------------
# BFS / DFS
# ----------------------------------------------------------------------
def bfs(adj, s):
    """Distâncias (em número de arestas) a partir de s; -1 = inalcançável."""
    dist = [-1] * len(adj)
    dist[s] = 0
    fila = deque([s])
    while fila:
        u = fila.popleft()
        du = dist[u] + 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = du
                fila.append(v)
    return dist


def dfs(adj, s=None):
    """DFS iterativa. Se `s` é None, percorre a floresta de todos os vértices.

    Retorna (ordem_de_visita, pre, pos, pai). `pre`/`pos` são os tempos de
    descoberta e término; -1 = não visitado.
    """
    n = len(adj)
    pre = [-1] * n
    pos = [-1] * n
    pai = [-1] * n
    ordem = []
    t = 0
    raizes = [s] if s is not None else range(n)
    for r in raizes:
        if pre[r] != -1:
            continue
        pre[r] = t
        t += 1
        ordem.append(r)
        pilha = [(r, 0)]
        while pilha:
            u, i = pilha[-1]
            if i < len(adj[u]):
                pilha[-1] = (u, i + 1)
                v = adj[u][i]
                if pre[v] == -1:
                    pre[v] = t
                    t += 1
                    pai[v] = u
                    ordem.append(v)
                    pilha.append((v, 0))
            else:
                pos[u] = t
                t += 1
                pilha.pop()
    return ordem, pre, pos, pai


def componentes_conexas(und_adj):
    """Componentes conexas de um grafo não-dirigido (BFS). Retorna (rotulo, tamanhos)."""
    n = len(und_adj)
    rot = [-1] * n
    tamanhos = []
    for r in range(n):
        if rot[r] != -1:
            continue
        c = len(tamanhos)
        rot[r] = c
        fila = deque([r])
        cont = 0
        while fila:
            u = fila.popleft()
            cont += 1
            for v in und_adj[u]:
                if rot[v] == -1:
                    rot[v] = c
                    fila.append(v)
        tamanhos.append(cont)
    return rot, tamanhos


# ----------------------------------------------------------------------
# Tarjan (componentes fortemente conexas)
# ----------------------------------------------------------------------
def tarjan_scc(adj):
    """Componentes fortemente conexas (Tarjan, versão iterativa).

    Retorna (comp, k): comp[v] = id da componente de v, k = nº de componentes.
    """
    n = len(adj)
    indice = [-1] * n
    low = [0] * n
    na_pilha = [False] * n
    pilha = []
    comp = [-1] * n
    contador = 0
    k = 0
    for raiz in range(n):
        if indice[raiz] != -1:
            continue
        indice[raiz] = low[raiz] = contador
        contador += 1
        pilha.append(raiz)
        na_pilha[raiz] = True
        trabalho = [(raiz, 0)]
        while trabalho:
            v, i = trabalho[-1]
            if i < len(adj[v]):
                trabalho[-1] = (v, i + 1)
                w = adj[v][i]
                if indice[w] == -1:
                    indice[w] = low[w] = contador
                    contador += 1
                    pilha.append(w)
                    na_pilha[w] = True
                    trabalho.append((w, 0))
                elif na_pilha[w]:
                    if indice[w] < low[v]:
                        low[v] = indice[w]
            else:
                trabalho.pop()
                if trabalho:
                    u = trabalho[-1][0]
                    if low[v] < low[u]:
                        low[u] = low[v]
                if low[v] == indice[v]:
                    while True:
                        w = pilha.pop()
                        na_pilha[w] = False
                        comp[w] = k
                        if w == v:
                            break
                    k += 1
    return comp, k


# ----------------------------------------------------------------------
# Eulerianidade
# ----------------------------------------------------------------------
def eulerianidade(g):
    """Verifica condições de circuito/caminho euleriano (dirigido e não-dirigido).

    Dirigido:      circuito  <=> grau_ent = grau_sai em todo vértice e os vértices
                                  com aresta estão numa só componente (fraca);
                   caminho   <=> idem, exceto um vértice com sai-ent = 1 e outro com ent-sai = 1.
    Não-dirigido:  circuito  <=> todos os graus pares (+ conectividade);
                   caminho   <=> 0 ou 2 vértices de grau ímpar (+ conectividade).
    """
    n = g.n
    rot, _ = componentes_conexas(g.und_adj)
    comps_com_aresta = {rot[v] for v in range(n) if g.und_adj[v]}
    conexo = len(comps_com_aresta) <= 1

    dif = [len(g.out_adj[v]) - len(g.in_adj[v]) for v in range(n)]
    desbalanceados = sum(1 for d in dif if d != 0)
    sai_a_mais = sum(1 for d in dif if d == 1)
    ent_a_mais = sum(1 for d in dif if d == -1)
    muito = sum(1 for d in dif if abs(d) > 1)
    impares = sum(1 for v in range(n) if len(g.und_adj[v]) % 2 == 1)

    dir_circ = desbalanceados == 0 and conexo
    dir_cam = (desbalanceados == 0 or (sai_a_mais == 1 and ent_a_mais == 1 and muito == 0)) and conexo
    und_circ = impares == 0 and conexo
    und_cam = impares in (0, 2) and conexo
    return {
        "componentes_com_aresta": len(comps_com_aresta),
        "vertices_desbalanceados_dirigido": desbalanceados,
        "vertices_sai_a_mais_1": sai_a_mais,
        "vertices_ent_a_mais_1": ent_a_mais,
        "vertices_desbalanco_maior_que_1": muito,
        "vertices_grau_impar_nao_dirigido": impares,
        "dirigido_tem_circuito_euleriano": dir_circ,
        "dirigido_tem_caminho_euleriano": dir_cam,
        "nao_dirigido_tem_circuito_euleriano": und_circ,
        "nao_dirigido_tem_caminho_euleriano": und_cam,
    }


# ----------------------------------------------------------------------
# Caminhos mínimos com pesos
# ----------------------------------------------------------------------
def com_pesos(adj, peso=1):
    """Converte adjacência simples em adjacência com pesos (peso constante)."""
    return [[(v, peso) for v in vs] for vs in adj]


def dijkstra(adj_w, s):
    """Dijkstra com heap binário (pesos não-negativos)."""
    dist = [INF] * len(adj_w)
    dist[s] = 0
    heap = [(0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj_w[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def bellman_ford(n, arestas_w, s, parada_antecipada=True):
    """Bellman-Ford. `arestas_w` = lista de (u, v, w).

    Retorna (dist, ciclo_negativo, rodadas). Com `parada_antecipada` o laço
    externo termina quando uma rodada não altera nada (o pior caso continua
    O(n m); na prática costuma terminar em ~ diâmetro rodadas).
    """
    dist = [INF] * n
    dist[s] = 0
    rodadas = 0
    for _ in range(n - 1):
        mudou = False
        rodadas += 1
        for u, v, w in arestas_w:
            du = dist[u]
            if du != INF and du + w < dist[v]:
                dist[v] = du + w
                mudou = True
        if parada_antecipada and not mudou:
            break
    ciclo = any(dist[u] != INF and dist[u] + w < dist[v] for u, v, w in arestas_w)
    return dist, ciclo, rodadas


def floyd_warshall(n, arestas_w):
    """Floyd-Warshall O(n^3). O laço em k é explícito; as atualizações (i, j)
    de cada k são vetorizadas com numpy (mesma recorrência, mais rápido).

    Retorna matriz n x n (float32, inf = inalcançável).
    """
    D = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(D, 0)
    for u, v, w in arestas_w:
        if w < D[u, v]:
            D[u, v] = w
    for k in range(n):
        np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :], out=D)
    return D


# ----------------------------------------------------------------------
# Árvore geradora mínima
# ----------------------------------------------------------------------
class UniaoBusca:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def achar(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def unir(self, a, b):
        a, b = self.achar(a), self.achar(b)
        if a == b:
            return False
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]:
            self.r[a] += 1
        return True


def kruskal(n, arestas_w):
    """Kruskal. `arestas_w` = lista de (u, v, w). Retorna (peso_total, arestas_escolhidas).
    Em grafo desconexo devolve a floresta geradora mínima."""
    uf = UniaoBusca(n)
    escolhidas = []
    total = 0
    for u, v, w in sorted(arestas_w, key=lambda e: e[2]):
        if uf.unir(u, v):
            escolhidas.append((u, v, w))
            total += w
    return total, escolhidas


def prim(adj_w):
    """Prim com heap binário; reinicia em cada componente (floresta geradora mínima)."""
    n = len(adj_w)
    visto = [False] * n
    total = 0
    escolhidas = []
    for r in range(n):
        if visto[r]:
            continue
        visto[r] = True
        heap = [(w, r, v) for v, w in adj_w[r]]
        heapq.heapify(heap)
        while heap:
            w, u, v = heapq.heappop(heap)
            if visto[v]:
                continue
            visto[v] = True
            total += w
            escolhidas.append((u, v, w))
            for x, wx in adj_w[v]:
                if not visto[x]:
                    heapq.heappush(heap, (wx, v, x))
    return total, escolhidas
