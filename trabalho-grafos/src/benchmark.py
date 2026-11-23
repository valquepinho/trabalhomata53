"""Parte II - análise de complexidade teórica x tempo real (média, desvio padrão e IC).

Regra do IC (conforme o enunciado), nível de confiança 95% por padrão:
    n >= 30            -> Normal:    x̄ ± z_{α/2} · s / √n
    n <  30, σ desc.   -> t-Student: x̄ ± t_{α/2, n-1} · s / √n
`s` é o desvio padrão amostral (ddof = 1).
"""
import math
import random
import time
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t as t_student

from .algoritmos import (bellman_ford, bfs, com_pesos, dfs, dijkstra, eulerianidade,
                         floyd_warshall, kruskal, prim, tarjan_scc)


# ----------------------------------------------------------------------
def intervalo_confianca(amostras, confianca=0.95):
    x = np.asarray(amostras, dtype=float)
    n = len(x)
    media = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else float("nan")
    alfa = 1 - confianca
    if n >= 30:
        crit, dist = float(norm.ppf(1 - alfa / 2)), "Normal (z)"
    else:
        crit, dist = float(t_student.ppf(1 - alfa / 2, n - 1)), f"t-Student (gl={n - 1})"
    h = crit * s / math.sqrt(n) if n > 1 else float("nan")
    return {"n_amostras": n, "media": media, "desvio": s, "ic_inf": media - h, "ic_sup": media + h,
            "valor_critico": crit, "distribuicao": dist}


def medir(fn, repeticoes):
    """Mede `fn(i)` (i = índice da repetição) com perf_counter, após 1 execução de aquecimento."""
    fn(0)
    tempos = []
    for i in range(repeticoes):
        t0 = time.perf_counter()
        fn(i)
        tempos.append(time.perf_counter() - t0)
    return tempos


# ----------------------------------------------------------------------
def amostra_bfs(g, k, semente=0):
    """Os k primeiros vértices na ordem de BFS (não-dirigida) a partir do vértice de maior grau;
    preserva conectividade. Se a componente tem menos de k vértices, completa com sorteio."""
    raiz = max(range(g.n), key=lambda v: len(g.und_adj[v]))
    visto = [False] * g.n
    ordem = []
    fila = deque([raiz])
    visto[raiz] = True
    while fila and len(ordem) < k:
        u = fila.popleft()
        ordem.append(u)
        for v in g.und_adj[u]:
            if not visto[v]:
                visto[v] = True
                fila.append(v)
    if len(ordem) < k:
        resto = [v for v in range(g.n) if not visto[v]]
        random.Random(semente).shuffle(resto)
        ordem += resto[: k - len(ordem)]
    return g.subgrafo(ordem)


# Modelos teóricos (n = |V|, m = |E| dirigido, mu = |E| não-dirigido)
MODELOS = {
    "BFS": ("O(n+m)", lambda n, m, mu: n + m),
    "DFS": ("O(n+m)", lambda n, m, mu: n + m),
    "Eulerianidade": ("O(n+m)", lambda n, m, mu: n + m),
    "Tarjan": ("O(n+m)", lambda n, m, mu: n + m),
    "Dijkstra": ("O((n+m) log n)", lambda n, m, mu: (n + m) * math.log2(max(n, 2))),
    "Bellman-Ford": ("O(n·m)", lambda n, m, mu: n * m),
    "Kruskal": ("O(m log m)", lambda n, m, mu: mu * math.log2(max(mu, 2))),
    "Prim": ("O(m log n)", lambda n, m, mu: mu * math.log2(max(n, 2))),
    "Floyd-Warshall": ("O(n^3)", lambda n, m, mu: n ** 3),
}


def _casos(h, semente):
    rng = random.Random(semente)
    n = h.n
    adjw = com_pesos(h.out_adj)
    arestas_w = [(u, v, 1) for u, v in h.arestas_dirigidas()]
    # pesos inteiros aleatórios (1..10) para a MST ser não-trivial
    nd = h.arestas_nao_dirigidas()
    pesos = {e: rng.randint(1, 10) for e in nd}
    nd_w = [(u, v, pesos[(u, v)]) for u, v in nd]
    nd_adjw = [[] for _ in range(n)]
    for u, v, w in nd_w:
        nd_adjw[u].append((v, w))
        nd_adjw[v].append((u, w))
    fontes = [rng.randrange(n) for _ in range(10000)]
    rodadas_bf = []

    def f_bf(i):
        rodadas_bf.append(bellman_ford(n, arestas_w, fontes[i])[2])

    casos = {
        "BFS": lambda i: bfs(h.out_adj, fontes[i]),
        "DFS": lambda i: dfs(h.out_adj, fontes[i]),
        "Eulerianidade": lambda i: eulerianidade(h),
        "Tarjan": lambda i: tarjan_scc(h.out_adj),
        "Dijkstra": lambda i: dijkstra(adjw, fontes[i]),
        "Bellman-Ford": f_bf,
        "Kruskal": lambda i: kruskal(n, nd_w),
        "Prim": lambda i: prim(nd_adjw),
    }
    # consistência: Kruskal e Prim devem dar o mesmo peso total
    assert kruskal(n, nd_w)[0] == prim(nd_adjw)[0], "Kruskal e Prim divergem"
    return casos, rodadas_bf, arestas_w


def executar(g, repeticoes=30, tamanhos=None, fw_tamanhos=(100, 200, 300, 400, 600, 800, 1000, 1400),
             fw_repeticoes=10, semente=0, log=print):
    """Roda todos os algoritmos em subgrafos de tamanhos crescentes. Retorna lista de linhas."""
    if tamanhos is None:
        base = [500, 1000, 2000, 3000, 4000, 5000]
        tamanhos = [k for k in base if k < g.n] + [g.n]
    linhas = []
    for k in tamanhos:
        h = g if k == g.n else amostra_bfs(g, k, semente)
        casos, rodadas_bf, arestas_w = _casos(h, semente)
        log(f"  [benchmark] n={h.n} m={h.m}")
        for nome, fn in casos.items():
            tempos = medir(fn, repeticoes)
            ic = intervalo_confianca(tempos)
            linha = {"algoritmo": nome, "n": h.n, "m": h.m, "m_nao_dirigido": h.m_und,
                     "repeticoes": repeticoes, **ic}
            if nome == "Bellman-Ford":
                linha["rodadas_medias"] = float(np.mean(rodadas_bf))
            linhas.append(linha)
    for k in fw_tamanhos:
        if k > g.n:
            continue
        h = amostra_bfs(g, k, semente)
        arestas_w = [(u, v, 1) for u, v in h.arestas_dirigidas()]
        reps = fw_repeticoes if k <= 400 else max(3, fw_repeticoes // 2)
        log(f"  [benchmark] Floyd-Warshall n={h.n} ({reps} repetições)")
        tempos = medir(lambda i: floyd_warshall(h.n, arestas_w), reps)
        linhas.append({"algoritmo": "Floyd-Warshall", "n": h.n, "m": h.m, "m_nao_dirigido": h.m_und,
                       "repeticoes": reps, **intervalo_confianca(tempos)})
    return linhas


def ajustar_modelos(linhas):
    """Para cada algoritmo: constante c (mínimos quadrados, T = c·f(n,m)), R² e expoente
    empírico (inclinação de log T x log n)."""
    saida = {}
    for nome, (txt, f) in MODELOS.items():
        pts = sorted((l for l in linhas if l["algoritmo"] == nome), key=lambda l: l["n"])
        if len(pts) < 2:
            continue
        n = np.array([p["n"] for p in pts], float)
        T = np.array([p["media"] for p in pts], float)
        X = np.array([f(p["n"], p["m"], p["m_nao_dirigido"]) for p in pts], float)
        c = float((X * T).sum() / (X * X).sum())
        ss_res = float(((T - c * X) ** 2).sum())
        ss_tot = float(((T - T.mean()) ** 2).sum())
        saida[nome] = {
            "complexidade_teorica": txt,
            "constante_c": c,
            "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "expoente_empirico_em_n": float(np.polyfit(np.log(n), np.log(T), 1)[0]),
            "tamanhos": [int(v) for v in n],
        }
    # Bellman-Ford com parada antecipada: custo real ~ m · (nº de rodadas executadas)
    pts = sorted((l for l in linhas if l["algoritmo"] == "Bellman-Ford" and "rodadas_medias" in l),
                 key=lambda l: l["n"])
    if len(pts) >= 2:
        T = np.array([p["media"] for p in pts])
        X = np.array([p["m"] * p["rodadas_medias"] for p in pts], float)
        c = float((X * T).sum() / (X * X).sum())
        ss_tot = float(((T - T.mean()) ** 2).sum())
        saida["Bellman-Ford (m·rodadas)"] = {
            "complexidade_teorica": "O(m · rodadas), rodadas ≤ n-1",
            "constante_c": c,
            "R2": 1 - float(((T - c * X) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
            "rodadas_medias": [float(p["rodadas_medias"]) for p in pts],
            "tamanhos": [int(p["n"]) for p in pts],
        }
    return saida


def grafico(linhas, ajustes, caminho):
    nomes = [n for n in MODELOS if any(l["algoritmo"] == n for l in linhas)]
    cols = 3
    lin = math.ceil(len(nomes) / cols)
    fig, axs = plt.subplots(lin, cols, figsize=(14, 3.8 * lin))
    axs = np.atleast_1d(axs).ravel()
    for ax, nome in zip(axs, nomes):
        pts = sorted((l for l in linhas if l["algoritmo"] == nome), key=lambda l: l["n"])
        n = np.array([p["n"] for p in pts], float)
        T = np.array([p["media"] for p in pts])
        err = np.array([[p["media"] - p["ic_inf"] for p in pts], [p["ic_sup"] - p["media"] for p in pts]])
        ax.errorbar(n, T, yerr=err, fmt="o-", capsize=3, label="observado (média ± IC95%)")
        if nome in ajustes:
            f = MODELOS[nome][1]
            X = [f(p["n"], p["m"], p["m_nao_dirigido"]) for p in pts]
            ax.plot(n, ajustes[nome]["constante_c"] * np.array(X), "--", label=f"{MODELOS[nome][0]} ajustado")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(nome)
        ax.set_xlabel("n (vértices)")
        ax.set_ylabel("tempo (s)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axs[len(nomes):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
