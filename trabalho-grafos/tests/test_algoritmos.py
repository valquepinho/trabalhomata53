"""Valida as implementações próprias contra o networkx (usado SÓ nos testes)."""
import random
import unittest

import networkx as nx
import numpy as np

from src import algoritmos as A
from src import metricas as M
from src import parte3 as P3
from src.grafo import Grafo


def grafo_aleatorio(n, m, seed):
    rng = random.Random(seed)
    arestas = set()
    while len(arestas) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            arestas.add((u, v))
    return Grafo(n, arestas), nx.DiGraph(list(arestas)), arestas


class Testes(unittest.TestCase):
    def setUp(self):
        self.g, self.D, self.arestas = grafo_aleatorio(120, 260, 3)
        self.D.add_nodes_from(range(120))
        self.U = self.D.to_undirected()

    def test_bfs(self):
        for s in (0, 7, 50):
            d = A.bfs(self.g.out_adj, s)
            ref = nx.single_source_shortest_path_length(self.D, s)
            for v in range(120):
                self.assertEqual(d[v], ref.get(v, -1))

    def test_dfs_alcancaveis(self):
        ordem, pre, pos, pai = A.dfs(self.g.out_adj, 0)
        self.assertEqual(set(ordem), set(nx.descendants(self.D, 0)) | {0})

    def test_tarjan(self):
        comp, k = A.tarjan_scc(self.g.out_adj)
        ref = list(nx.strongly_connected_components(self.D))
        self.assertEqual(k, len(ref))
        for c in ref:
            self.assertEqual(len({comp[v] for v in c}), 1)

    def test_componentes_fracas(self):
        _, tam = A.componentes_conexas(self.g.und_adj)
        self.assertEqual(sorted(tam), sorted(len(c) for c in nx.connected_components(self.U)))

    def test_dijkstra_bf_fw(self):
        adjw = A.com_pesos(self.g.out_adj)
        aw = [(u, v, 1) for u, v in self.arestas]
        FW = A.floyd_warshall(120, aw)
        for s in (0, 5, 33):
            dd = A.dijkstra(adjw, s)
            bf, neg, _ = A.bellman_ford(120, aw, s)
            self.assertFalse(neg)
            bfs = A.bfs(self.g.out_adj, s)
            for v in range(120):
                esp = bfs[v] if bfs[v] >= 0 else float("inf")
                self.assertEqual(dd[v], esp)
                self.assertEqual(bf[v], esp)
                self.assertEqual(FW[s, v], esp)

    def test_bellman_ford_peso_negativo(self):
        d, neg, _ = A.bellman_ford(3, [(0, 1, 2), (1, 2, -1), (0, 2, 5)], 0)
        self.assertEqual(d, [0, 2, 1])
        self.assertFalse(neg)
        _, neg, _ = A.bellman_ford(3, [(0, 1, 1), (1, 2, -3), (2, 1, 1)], 0)
        self.assertTrue(neg)

    def test_mst(self):
        rng = random.Random(1)
        nd = [(u, v) for u, v in self.U.edges()]
        w = [(u, v, rng.randint(1, 9)) for u, v in nd]
        adjw = [[] for _ in range(120)]
        W = nx.Graph()
        W.add_nodes_from(range(120))
        for u, v, x in w:
            adjw[u].append((v, x))
            adjw[v].append((u, x))
            W.add_edge(u, v, weight=x)
        ref = sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(W, data=True))
        self.assertEqual(A.kruskal(120, w)[0], ref)
        self.assertEqual(A.prim(adjw)[0], ref)

    def test_eulerianidade(self):
        ciclo = Grafo(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        self.assertTrue(A.eulerianidade(ciclo)["dirigido_tem_circuito_euleriano"])
        self.assertTrue(A.eulerianidade(ciclo)["nao_dirigido_tem_circuito_euleriano"])
        caminho = Grafo(3, [(0, 1), (1, 2)])
        e = A.eulerianidade(caminho)
        self.assertFalse(e["dirigido_tem_circuito_euleriano"])
        self.assertTrue(e["dirigido_tem_caminho_euleriano"])
        # K4 não-dirigido: 4 vértices de grau 3 (ímpar) => nada
        k4 = Grafo(4, [(u, v) for u in range(4) for v in range(4) if u != v])
        self.assertFalse(A.eulerianidade(k4)["nao_dirigido_tem_caminho_euleriano"])

    def test_triangulos_clustering(self):
        t = M.triangulos_clustering(self.g.und_adj)
        self.assertEqual(t["triangulos"], sum(nx.triangles(self.U).values()) // 3)
        self.assertAlmostEqual(t["clustering_medio_todos_vertices"], nx.average_clustering(self.U))
        self.assertAlmostEqual(t["transitividade_global"], nx.transitivity(self.U))

    def test_distancias(self):
        comp = M.componentes(self.g)
        c = M.caminhos(self.g, comp)
        vs = comp["vertices_maior_fraca"]
        H = self.U.subgraph(vs)
        self.assertEqual(c["nao_dirigido_maior_WCC"]["diametro"], nx.diameter(H))
        self.assertEqual(c["nao_dirigido_maior_WCC"]["raio"], nx.radius(H))
        self.assertAlmostEqual(c["nao_dirigido_maior_WCC"]["comprimento_medio"],
                               nx.average_shortest_path_length(H))
        S = self.D.subgraph(comp["vertices_maior_forte"])
        self.assertEqual(c["dirigido_maior_SCC"]["diametro"], nx.diameter(S))
        self.assertEqual(c["dirigido_maior_SCC"]["raio"], nx.radius(S))
        self.assertAlmostEqual(c["dirigido_maior_SCC"]["comprimento_medio"],
                               nx.average_shortest_path_length(S))

    def test_intermediacao(self):
        bc = P3.intermediacao(self.g.und_adj, list(range(120)))
        ref = nx.betweenness_centrality(self.U, normalized=False)
        for v in range(120):
            self.assertAlmostEqual(bc[v], ref[v], places=6)

    def test_rewire_preserva_graus(self):
        rng = random.Random(0)
        adj = P3.grafo_rewire(self.g.und_adj, rng)
        self.assertEqual([len(a) for a in adj], [len(a) for a in self.g.und_adj])
        for u, vs in enumerate(adj):
            self.assertEqual(len(set(vs)), len(vs))
            self.assertNotIn(u, vs)

    def test_lei_de_potencia_recupera_alpha(self):
        rng = np.random.default_rng(0)
        # amostra discreta ~ Zipf(alpha=2.5)
        x = rng.zipf(2.5, 20000)
        r = P3.lei_de_potencia(x)
        self.assertAlmostEqual(r["alpha"], 2.5, delta=0.15)
        self.assertEqual(r["comparacoes"]["exponencial"]["favorece"], "lei de potência")


if __name__ == "__main__":
    unittest.main()
