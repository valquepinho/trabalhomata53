"""Parte III - small-world, lei de potência e robustez (sem bibliotecas de grafos)."""
import math
import random
from collections import deque

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import zeta
from scipy.stats import norm

from .algoritmos import bfs, componentes_conexas, tarjan_scc
from .metricas import triangulos_clustering


# ======================================================================
# 1. Small-world
# ======================================================================
def _maior_componente(und_adj):
    rot, tam = componentes_conexas(und_adj)
    c = int(np.argmax(tam))
    return [v for v in range(len(und_adj)) if rot[v] == c]


def _reindexar(und_adj, vertices):
    novo = {v: i for i, v in enumerate(vertices)}
    return [[novo[w] for w in und_adj[v] if w in novo] for v in vertices]


def caminho_medio(und_adj, fontes):
    """Comprimento médio dos caminhos mínimos (pares alcançáveis) a partir de `fontes`."""
    soma = 0
    cont = 0
    for s in fontes:
        d = np.asarray(bfs(und_adj, s))
        d = d[d > 0]
        soma += int(d.sum())
        cont += int(d.size)
    return soma / cont if cont else float("nan")


def grafo_aleatorio_gnm(n, m, rng):
    """Erdős-Rényi G(n, m): m arestas distintas sorteadas uniformemente."""
    arestas = set()
    while len(arestas) < m:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u != v:
            arestas.add((u, v) if u < v else (v, u))
    adj = [[] for _ in range(n)]
    for u, v in arestas:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def grafo_rewire(und_adj, rng, trocas_por_aresta=10):
    """Modelo nulo que preserva a sequência de graus (troca dupla de arestas)."""
    n = len(und_adj)
    arestas = [(u, v) for u in range(n) for v in und_adj[u] if u < v]
    existentes = set(arestas)
    m = len(arestas)
    for _ in range(trocas_por_aresta * m):
        i, j = rng.randrange(m), rng.randrange(m)
        a, b = arestas[i]
        c, d = arestas[j]
        if rng.random() < 0.5:
            c, d = d, c
        # (a,b),(c,d) -> (a,d),(c,b)
        if len({a, b, c, d}) < 4:
            continue
        n1 = (a, d) if a < d else (d, a)
        n2 = (c, b) if c < b else (b, c)
        if n1 in existentes or n2 in existentes:
            continue
        existentes.discard(arestas[i])
        existentes.discard(arestas[j])
        arestas[i], arestas[j] = n1, n2
        existentes.add(n1)
        existentes.add(n2)
    adj = [[] for _ in range(n)]
    for u, v in arestas:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _C_L(und_adj, n_fontes, rng):
    """C (clustering médio, todos os vértices) e L (caminho médio) na maior componente."""
    vs = _maior_componente(und_adj)
    sub = und_adj if len(vs) == len(und_adj) else _reindexar(und_adj, vs)
    C = triangulos_clustering(sub)["clustering_medio_todos_vertices"]
    fontes = list(range(len(sub)))
    if n_fontes and len(fontes) > n_fontes:
        fontes = rng.sample(fontes, n_fontes)
    return C, caminho_medio(sub, fontes), len(sub)


def pequeno_mundo(g, replicas=5, n_fontes=300, semente=0):
    """Compara (C, L) do grafo (versão não-dirigida, maior componente) com modelos nulos.

    sigma = (C / C_rand) / (L / L_rand)   (Humphries & Gurney, 2008); sigma > 1 sugere small-world.
    Também retorna as aproximações analíticas  C_rand ~ <k>/n  e  L_rand ~ ln n / ln <k>.
    """
    rng = random.Random(semente)
    vs = _maior_componente(g.und_adj)
    sub = _reindexar(g.und_adj, vs)
    n = len(sub)
    m = sum(len(a) for a in sub) // 2
    k_med = 2 * m / n
    C, L, _ = _C_L(sub, None, rng)   # valores reais, exatos

    def media_nulo(gerador):
        Cs, Ls = [], []
        for _ in range(replicas):
            c, l, _ = _C_L(gerador(), n_fontes, rng)
            Cs.append(c)
            Ls.append(l)
        return float(np.mean(Cs)), float(np.mean(Ls)), float(np.std(Cs, ddof=1)), float(np.std(Ls, ddof=1))

    C_er, L_er, sC_er, sL_er = media_nulo(lambda: grafo_aleatorio_gnm(n, m, rng))
    C_cf, L_cf, sC_cf, sL_cf = media_nulo(lambda: grafo_rewire(sub, rng))
    C_an = k_med / n
    L_an = math.log(n) / math.log(k_med)
    return {
        "vertices_maior_componente": n,
        "arestas_maior_componente": m,
        "grau_medio_nao_dirigido": k_med,
        "C_real": C,
        "L_real": L,
        "ER": {"C": C_er, "L": L_er, "desvio_C": sC_er, "desvio_L": sL_er,
               "sigma": (C / C_er) / (L / L_er)},
        "preserva_graus": {"C": C_cf, "L": L_cf, "desvio_C": sC_cf, "desvio_L": sL_cf,
                           "sigma": (C / C_cf) / (L / L_cf)},
        "analitico": {"C": C_an, "L": L_an, "sigma": (C / C_an) / (L / L_an)},
        "replicas": replicas,
        "fontes_por_replica": n_fontes,
    }


# ======================================================================
# 2. Lei de potência (MLE discreto, Clauset-Shalizi-Newman 2009)
# ======================================================================
def _ll_pot(alpha, x, xmin):
    return -len(x) * np.log(zeta(alpha, xmin)) - alpha * np.log(x).sum()


def _ajuste_alpha(cauda, xmin):
    r = minimize_scalar(lambda a: -_ll_pot(a, cauda, xmin), bounds=(1.01, 12.0), method="bounded")
    return float(r.x)


def _ks(cauda, alpha, xmin):
    vals, cont = np.unique(cauda, return_counts=True)
    emp = np.cumsum(cont) / cont.sum()
    mod = 1.0 - zeta(alpha, vals + 1) / zeta(alpha, xmin)
    return float(np.max(np.abs(emp - mod)))


def _loglik_alternativas(cauda, xmin):
    """Log-verossimilhanças ponto a ponto de exponencial e log-normal discretas (truncadas em xmin)."""
    media = cauda.mean()
    out = {}
    # exponencial discreta: p(x) = (1 - e^-l) e^{l xmin} e^{-l x}
    lam = math.log(1 + 1 / (media - xmin)) if media > xmin else 10.0
    out["exponencial"] = (np.log(1 - math.exp(-lam)) + lam * xmin - lam * cauda, {"lambda": lam})
    # log-normal discreta, normalizada no suporte xmin..X
    X = int(10 * cauda.max())
    suporte = np.arange(xmin, X + 1, dtype=float)
    ls = np.log(suporte)

    def nll(p):
        mu, s = p[0], abs(p[1]) + 1e-6
        logw = -ls - (ls - mu) ** 2 / (2 * s * s)
        lognorm = np.log(np.exp(logw - logw.max()).sum()) + logw.max()
        lx = np.log(cauda)
        return -(np.sum(-lx - (lx - mu) ** 2 / (2 * s * s)) - len(cauda) * lognorm)

    # Parâmetros limitados (mu in [-12, 12], sigma in [0.2, 4]): sem limites a log-normal degenera
    # (sigma -> infinito) e imita uma lei de potência, o que invalida a comparação.
    melhor_r = None
    for mu0, s0 in ((np.log(media), 1.0), (1.0, 0.5), (3.0, 2.0)):
        r = minimize(nll, x0=[mu0, s0], method="L-BFGS-B", bounds=[(-12, 12), (0.2, 4.0)])
        if melhor_r is None or r.fun < melhor_r.fun:
            melhor_r = r
    r = melhor_r
    mu, s = float(r.x[0]), abs(float(r.x[1])) + 1e-6
    logw = -ls - (ls - mu) ** 2 / (2 * s * s)
    lognorm = np.log(np.exp(logw - logw.max()).sum()) + logw.max()
    lx = np.log(cauda)
    out["lognormal"] = (-lx - (lx - mu) ** 2 / (2 * s * s) - lognorm, {"mu": mu, "sigma": s})
    return out


def lei_de_potencia(vetor_graus, n_min_cauda=50):
    """Ajusta lei de potência discreta à cauda do vetor de graus e compara com alternativas.

    Procedimento: para cada xmin candidato estima alpha por MLE e calcula a distância KS;
    escolhe o xmin de menor KS. Depois compara a verossimilhança com exponencial e log-normal
    (razão de verossimilhança normalizada de Vuong; R > 0 favorece a lei de potência).
    """
    x = np.asarray(vetor_graus)
    zeros = int((x < 1).sum())
    x = x[x >= 1].astype(float)
    candidatos = [v for v in np.unique(x) if (x >= v).sum() >= n_min_cauda]
    melhor = None
    for xmin in candidatos:
        cauda = x[x >= xmin]
        alpha = _ajuste_alpha(cauda, xmin)
        ks = _ks(cauda, alpha, xmin)
        if melhor is None or ks < melhor[0]:
            melhor = (ks, xmin, alpha)
    if melhor is None:
        return {"erro": "cauda curta demais para ajustar", "zeros_ignorados": zeros}
    ks, xmin, alpha = melhor
    cauda = x[x >= xmin]
    ll_pot = -np.log(zeta(alpha, xmin)) - alpha * np.log(cauda)
    comparacoes = {}
    for nome, (ll_alt, par) in _loglik_alternativas(cauda, int(xmin)).items():
        d = ll_pot - ll_alt
        R = float(d.sum())
        sd = float(d.std(ddof=1))
        z = R / (math.sqrt(len(d)) * sd) if sd > 0 else 0.0
        comparacoes[nome] = {"R": R, "z_vuong": z, "p_valor": float(2 * (1 - norm.cdf(abs(z)))),
                             "parametros": par,
                             "favorece": "lei de potência" if R > 0 else nome}
    return {
        "xmin": int(xmin), "alpha": alpha, "ks": ks,
        "n_cauda": int(len(cauda)), "n_total": int(len(x)), "zeros_ignorados": zeros,
        "comparacoes": comparacoes,
    }


def ccdf(vetor_graus):
    """(valores, P(X >= valor)) para o gráfico log-log."""
    x = np.asarray(vetor_graus)
    x = x[x >= 1]
    vals, cont = np.unique(x, return_counts=True)
    p = cont[::-1].cumsum()[::-1] / cont.sum()
    return vals, p


# ======================================================================
# 3. Robustez
# ======================================================================
def intermediacao(und_adj, fontes):
    """Centralidade de intermediação (Brandes, não ponderado) usando as `fontes` dadas.
    Com todas as fontes é exata; com amostra, é um estimador (ranking aproximado)."""
    n = len(und_adj)
    bc = [0.0] * n
    for s in fontes:
        S = []
        P = [[] for _ in range(n)]
        sigma = [0] * n
        sigma[s] = 1
        dist = [-1] * n
        dist[s] = 0
        Q = deque([s])
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in und_adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = [0.0] * n
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    escala = n / len(fontes) / 2.0
    return [b * escala for b in bc]


def _metricas_robustez(h, n_orig, n_fontes, rng):
    rot, tam = componentes_conexas(h.und_adj)
    maior = int(np.argmax(tam))
    vs = [v for v in range(h.n) if rot[v] == maior]
    fontes = vs if len(vs) <= n_fontes else rng.sample(vs, n_fontes)
    soma = cont = 0
    efic = 0.0
    for s in fontes:
        d = np.asarray(bfs(h.und_adj, s))
        d = d[d > 0]
        soma += int(d.sum())
        cont += int(d.size)
        efic += float((1.0 / d).sum()) / (h.n - 1)
    rot_s, k_s = tarjan_scc(h.out_adj)
    comp_s = np.bincount(rot_s, minlength=k_s)
    n_isolados = sum(1 for a in h.und_adj if len(a) == 0)
    return {
        "frac_maior_WCC": len(vs) / n_orig,   # relativo ao n ORIGINAL (máx. = 1 - fração removida)
        "frac_maior_WCC_entre_restantes": len(vs) / h.n,
        "n_componentes_fracas": len(tam),
        "L_maior_WCC": soma / cont if cont else float("nan"),
        "eficiencia_global": efic / len(fontes),
        "frac_maior_SCC": int(comp_s.max()) / n_orig,
        "n_nos_isolados": n_isolados,
        "frac_nos_isolados": n_isolados / h.n if h.n else 0.0,
    }


def robustez(g, fracao=0.05, tentativas=30, n_fontes=200, k_intermediacao=500, semente=0):
    """Remove `fracao` dos vértices (aleatório x mais centrais) e mede o impacto.

    Ataques direcionados são estáticos: o ranking é calculado uma vez no grafo original.
    Centralidades: grau (não-dirigido) e intermediação (amostrada com `k_intermediacao` fontes).
    """
    rng = random.Random(semente)
    n = g.n
    q = max(1, int(round(fracao * n)))
    base = _metricas_robustez(g, n, n_fontes, rng)

    chaves = ["frac_maior_WCC", "frac_maior_WCC_entre_restantes", "n_componentes_fracas", "L_maior_WCC", "eficiencia_global", "frac_maior_SCC", "n_nos_isolados", "frac_nos_isolados"]
    acum = {c: [] for c in chaves}
    for t in range(tentativas):
        rem = random.Random(semente + 1000 + t).sample(range(n), q)
        r = _metricas_robustez(g.sem_vertices(rem), n, n_fontes, rng)
        for c in chaves:
            acum[c].append(r[c])
    aleatorio = {
        c: {
            "media": float(np.mean(v)),
            "desvio": float(np.std(v, ddof=1)),
            "valores": [float(x) for x in v],
        }
        for c, v in acum.items()
    }

    grau = [len(a) for a in g.und_adj]
    por_grau = sorted(range(n), key=lambda v: (-grau[v], v))[:q]
    fontes_b = list(range(n)) if k_intermediacao >= n else rng.sample(range(n), k_intermediacao)
    bc = intermediacao(g.und_adj, fontes_b)
    por_bc = sorted(range(n), key=lambda v: (-bc[v], v))[:q]
    return {
        "vertices_removidos": q,
        "fracao": fracao,
        "tentativas_aleatorio": tentativas,
        "base": base,
        "aleatorio": aleatorio,
        "alvo_grau": _metricas_robustez(g.sem_vertices(por_grau), n, n_fontes, rng),
        "alvo_intermediacao": _metricas_robustez(g.sem_vertices(por_bc), n, n_fontes, rng),
        "intermediacao_fontes": len(fontes_b),
        "sobreposicao_grau_x_intermediacao": len(set(por_grau) & set(por_bc)),
    }


# ======================================================================
# 4. Modelos clássicos (pergunta extra): ER, Barabási-Albert, Watts-Strogatz
# ======================================================================
def grafo_ba(n, m_medio, rng):
    """Barabási-Albert: cada novo vértice liga-se a 3 ou 4 existentes (prob. proporcional ao grau),
    com média `m_medio`, para igualar o nº de arestas do grafo real."""
    p4 = m_medio - 3.0
    adj = [set() for _ in range(n)]
    rep = []  # vértice v aparece grau(v) vezes => sorteio preferencial
    for u in range(4):
        for v in range(u + 1, 4):
            adj[u].add(v)
            adj[v].add(u)
            rep += [u, v]
    for v in range(4, n):
        k = 4 if rng.random() < p4 else 3
        alvos = set()
        while len(alvos) < k:
            alvos.add(rng.choice(rep))
        for u in alvos:
            adj[v].add(u)
            adj[u].add(v)
            rep += [u, v]
    return [sorted(s) for s in adj]


def grafo_ws(n, k, p, rng):
    """Watts-Strogatz: anel regular com k vizinhos (k/2 de cada lado); cada aresta é religada com prob. p."""
    adj = [set() for _ in range(n)]
    for u in range(n):
        for j in range(1, k // 2 + 1):
            v = (u + j) % n
            adj[u].add(v)
            adj[v].add(u)
    for u in range(n):
        for j in range(1, k // 2 + 1):
            v = (u + j) % n
            if v in adj[u] and rng.random() < p:
                w = rng.randrange(n)
                if w != u and w not in adj[u]:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[u].add(w)
                    adj[w].add(u)
    return [sorted(s) for s in adj]


def modelos_classicos(g, replicas=5, n_fontes=300, semente=0):
    """Compara o grafo real (versão não-dirigida) com ER, BA e WS de mesmo n e ~mesmo m."""
    rng = random.Random(semente)
    n, m = g.n, g.m_und
    k_med = 2 * m / n

    def resumo(adj, exato=False):
        c, l, nn = _C_L(adj, None if exato else n_fontes, rng)
        graus = np.array([len(a) for a in adj])
        return {"C": c, "L": l, "grau_max": int(graus.max()), "grau_medio": float(graus.mean()),
                "desvio_grau": float(graus.std()), "arestas": int(graus.sum() // 2)}

    def media(gerador):
        rs = [resumo(gerador()) for _ in range(replicas)]
        return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} | {
            "desvio_C": float(np.std([r["C"] for r in rs], ddof=1)),
            "desvio_L": float(np.std([r["L"] for r in rs], ddof=1))}

    k_ws = 2 * int(round(k_med / 2))
    return {
        "real": resumo(g.und_adj, exato=True),
        "ER": media(lambda: grafo_aleatorio_gnm(n, m, rng)),
        "BA": media(lambda: grafo_ba(n, k_med / 2, rng)),
        "WS": media(lambda: grafo_ws(n, k_ws, 0.1, rng)),
        "parametros": {"BA_m_medio": k_med / 2, "WS_k": k_ws, "WS_p": 0.1, "replicas": replicas},
    }
