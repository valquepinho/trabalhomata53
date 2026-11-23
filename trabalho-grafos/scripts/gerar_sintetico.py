#!/usr/bin/env python3
"""Gera um grafo SINTÉTICO dirigido com tamanho parecido com o p2p-Gnutella08.

Serve APENAS para testar o pipeline. Os números dele NÃO devem ir para o relatório.
Uso: python scripts/gerar_sintetico.py data/sintetico.txt [n] [m_alvo] [semente]
"""
import random
import sys

destino = sys.argv[1] if len(sys.argv) > 1 else "data/sintetico.txt"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6301
m_alvo = int(sys.argv[3]) if len(sys.argv) > 3 else 20777
semente = int(sys.argv[4]) if len(sys.argv) > 4 else 1

rng = random.Random(semente)
arestas = set()
alvos = [0, 1]            # lista com repetição => escolha proporcional ao grau (preferencial)
arestas.add((0, 1))
for v in range(2, n - 2):  # os 2 últimos vértices formam uma componente à parte
    k = max(1, min(8, int(rng.expovariate(1 / 2.6)) + 1))
    for _ in range(k):
        u = rng.choice(alvos) if rng.random() < 0.6 else rng.randrange(v)
        if u != v:
            arestas.add((v, u) if rng.random() < 0.75 else (u, v))
            alvos += [u, v]
while len(arestas) < m_alvo - 1:      # completa com arestas aleatórias
    u, v = rng.randrange(n - 2), rng.randrange(n - 2)
    if u != v:
        arestas.add((u, v))
arestas.add((n - 2, n - 1))
with open(destino, "w") as f:
    f.write("# GRAFO SINTETICO para teste do pipeline (nao e o p2p-Gnutella08)\n")
    f.write(f"# Nodes: {n} Edges: {len(arestas)}\n# FromNodeId\tToNodeId\n")
    for u, v in sorted(arestas):
        f.write(f"{u}\t{v}\n")
print(f"{destino}: n={n} m={len(arestas)}")
