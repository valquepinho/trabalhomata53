"""Leitura, limpeza e representação do grafo.

O grafo original (p2p-Gnutella08) é DIRIGIDO. Esta classe guarda:
  - out_adj : lista de adjacência de saída   (u -> v)
  - in_adj  : lista de adjacência de entrada (v <- u)
  - und_adj : vizinhança não-dirigida (união de entrada e saída, sem repetição)

Tratamento aplicado na leitura (registrado em `info` para o relatório):
  1. linhas de comentário (#) ignoradas;
  2. laços (u -> u) removidos;
  3. arestas dirigidas duplicadas removidas;
  4. vértices reindexados para 0..n-1 (ordem crescente do id original).
"""
import gzip
import io


class Grafo:
    def __init__(self, n, arestas, ids_originais=None):
        """`arestas`: iterável de pares (u, v) com 0 <= u, v < n, já sem laços/duplicatas."""
        self.n = n
        self.ids = ids_originais if ids_originais is not None else list(range(n))
        out_sets = [set() for _ in range(n)]
        in_sets = [set() for _ in range(n)]
        for u, v in arestas:
            out_sets[u].add(v)
            in_sets[v].add(u)
        self.out_adj = [sorted(s) for s in out_sets]
        self.in_adj = [sorted(s) for s in in_sets]
        self.und_adj = [sorted(out_sets[i] | in_sets[i]) for i in range(n)]
        self.m = sum(len(a) for a in self.out_adj)            # arestas dirigidas
        self.m_und = sum(len(a) for a in self.und_adj) // 2   # arestas não-dirigidas

    # ------------------------------------------------------------------
    def arestas_dirigidas(self):
        return [(u, v) for u in range(self.n) for v in self.out_adj[u]]

    def arestas_nao_dirigidas(self):
        return [(u, v) for u in range(self.n) for v in self.und_adj[u] if u < v]

    def subgrafo(self, vertices):
        """Subgrafo induzido pelos `vertices`, reindexado em 0..k-1."""
        vs = sorted(set(vertices))
        novo = {v: i for i, v in enumerate(vs)}
        arestas = [(novo[u], novo[v]) for u in vs for v in self.out_adj[u] if v in novo]
        return Grafo(len(vs), arestas, [self.ids[v] for v in vs])

    def sem_vertices(self, removidos):
        """Subgrafo induzido após remover o conjunto `removidos`."""
        rem = set(removidos)
        return self.subgrafo([v for v in range(self.n) if v not in rem])


def _abrir(caminho):
    if str(caminho).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(caminho, "rb"), encoding="utf-8")
    return open(caminho, "r", encoding="utf-8")


def carregar(caminho):
    """Lê uma lista de arestas (formato SNAP: 'u<TAB>v', '#' = comentário).

    Retorna (Grafo, info) onde `info` documenta o tratamento realizado.
    """
    brutas = 0
    lacos = 0
    vistos = set()
    ids = set()
    with _abrir(caminho) as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            partes = linha.split()
            u, v = int(partes[0]), int(partes[1])
            brutas += 1
            ids.add(u)
            ids.add(v)
            if u == v:
                lacos += 1
                continue
            vistos.add((u, v))
    duplicadas = brutas - lacos - len(vistos)
    ids_ord = sorted(ids)
    idx = {x: i for i, x in enumerate(ids_ord)}
    arestas = [(idx[u], idx[v]) for u, v in vistos]
    g = Grafo(len(ids_ord), arestas, ids_ord)
    info = {
        "linhas_de_aresta_lidas": brutas,
        "lacos_removidos": lacos,
        "arestas_duplicadas_removidas": duplicadas,
        "vertices": g.n,
        "arestas_dirigidas": g.m,
        "arestas_nao_dirigidas": g.m_und,
        "pares_reciprocos": g.m - g.m_und,
    }
    return g, info
