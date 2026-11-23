#!/usr/bin/env python3
"""Executa todo o trabalho: Parte I, Parte II (benchmark) e Parte III.

Uso (da raiz do repositório):
    python run_all.py --entrada data/p2p-Gnutella08.txt.gz --saida results

Opções úteis:
    --repeticoes 30        nº de repetições de cada medição de tempo (>= 30 => IC com Normal)
    --max-fontes N         usa N fontes de BFS (amostra) nas distâncias da Parte I (mais rápido)
    --pular-benchmark      não roda a Parte II
    --pular-parte3         não roda a Parte III
    --fw-completo          roda também 1 vez o Floyd-Warshall no grafo inteiro (minutos)
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from src import benchmark as B
from src import grafo as G
from src import metricas as M
from src import parte3 as P3
from src import saidas as S
from src.algoritmos import eulerianidade, floyd_warshall


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def salvar_json(obj, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, help="lista de arestas (.txt ou .txt.gz, formato SNAP)")
    ap.add_argument("--saida", default="results")
    ap.add_argument("--repeticoes", type=int, default=30)
    ap.add_argument("--max-fontes", type=int, default=None)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--replicas-nulo", type=int, default=5)
    ap.add_argument("--tentativas-robustez", type=int, default=30)
    ap.add_argument("--k-intermediacao", type=int, default=500)
    ap.add_argument("--pular-benchmark", action="store_true")
    ap.add_argument("--pular-parte3", action="store_true")
    ap.add_argument("--fw-completo", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.saida, exist_ok=True)
    out = lambda nome: os.path.join(a.saida, nome)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # ------------------------------------------------------------ leitura
    log("Lendo e tratando o grafo...")
    g, info = G.carregar(a.entrada)
    salvar_json(info, out("tratamento.json"))
    log(f"  n={g.n}  m={g.m} (dirigidas)  m_und={g.m_und}  info={info}")

    # ------------------------------------------------------------ Parte I
    log("Parte I: graus, densidade, componentes...")
    vetores, resumo_graus = M.graus(g)
    dens = M.densidade(g)
    comp = M.componentes(g)
    with open(out("graus.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grau", "n_entrada", "n_saida", "n_nao_dirigido", "n_total_ent_mais_sai"])
        dist = {k: M.distribuicao(v) for k, v in vetores.items()}
        todos = sorted(set().union(*[set(d) for d in dist.values()]))
        for k in todos:
            w.writerow([k] + [dist[c].get(k, 0) for c in ("entrada", "saida", "nao_dirigido", "total")])
    with open(out("componentes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "posicao_por_tamanho", "tamanho"])
        for tipo, lista in (("fraca", comp["tamanhos_fracas"]), ("forte", comp["tamanhos_fortes"])):
            for i, t in enumerate(lista, 1):
                w.writerow([tipo, i, t])
    log(f"  componentes fracas={comp['n_fracas']} fortes={comp['n_fortes']}")

    log("Parte I: triângulos e clustering...")
    tri = M.triangulos_clustering(g.und_adj)
    log(f"  triângulos={tri['triangulos']}  C={tri['clustering_medio_todos_vertices']:.5f}")

    log("Parte I: distâncias (BFS a partir de cada vértice; pode levar alguns minutos)...")
    cam = M.caminhos(g, comp, max_fontes=a.max_fontes)
    for k, v in cam.items():
        log(f"  {k}: diam={v['diametro']} raio={v['raio']} L={v['comprimento_medio']:.4f} ef90={v['diametro_efetivo_90']:.2f}")

    eul = eulerianidade(g)
    tabela1 = S.tabela_parte1(g, info, resumo_graus, dens, comp, cam, tri, eul)
    S.salvar_tabela(tabela1, ["Medida", "Computada?", "Valor", "Observação"], out("tabela_medidas"),
                    "Medidas da Parte I e se foram computadas.", "tab:medidas",
                    ["p{3.6cm}", "c", "p{5cm}", "p{4cm}"])
    parte1 = {
        "tratamento": info, "graus": resumo_graus, "densidade": dens,
        "componentes": {k: v for k, v in comp.items() if not k.startswith("rot_") and not k.startswith("vertices_")},
        "caminhos": cam,
        "triangulos_clustering": {k: v for k, v in tri.items() if k != "clustering_local"},
        "eulerianidade": eul,
    }
    salvar_json(parte1, out("parte1.json"))

    # ------------------------------------------------------------ Parte III
    pot = {}
    if not a.pular_parte3:
        log("Parte III: lei de potência...")
        for k in ("nao_dirigido", "entrada", "saida"):
            pot[k] = P3.lei_de_potencia(vetores[k])
            log(f"  {k}: " + (f"alpha={pot[k]['alpha']:.3f} xmin={pot[k]['xmin']} KS={pot[k]['ks']:.3f}"
                              if "alpha" in pot[k] else pot[k].get("erro", "?")))
        log("Parte III: small-world (modelos nulos)...")
        sw = P3.pequeno_mundo(g, replicas=a.replicas_nulo, semente=a.semente)
        log(f"  C={sw['C_real']:.4f} L={sw['L_real']:.3f} | ER: C={sw['ER']['C']:.4f} L={sw['ER']['L']:.3f} "
            f"sigma={sw['ER']['sigma']:.2f} | preserva graus sigma={sw['preserva_graus']['sigma']:.2f}")
        log("Parte III: robustez (aleatório x direcionado)...")
        rb = P3.robustez(g, tentativas=a.tentativas_robustez, k_intermediacao=a.k_intermediacao, semente=a.semente)
        log("Parte III (extra): modelos clássicos ER/BA/WS...")
        mod = P3.modelos_classicos(g, replicas=a.replicas_nulo, semente=a.semente)
        for k in ("real", "ER", "BA", "WS"):
            log(f"  {k}: C={mod[k]['C']:.4f} L={mod[k]['L']:.3f} kmax={mod[k]['grau_max']:.0f}")
        salvar_json({"lei_de_potencia": pot, "small_world": sw, "robustez": rb, "modelos_classicos": mod},
                    out("parte3.json"))
        S.fig_graus(vetores, pot, out("distribuicao_graus.png"))
        S.fig_smallworld(sw, out("smallworld.png"))
        S.fig_robustez(rb, out("robustez.png"))
    else:
        S.fig_graus(vetores, {}, out("distribuicao_graus.png"))

    log("Figuras da Parte I...")
    S.fig_caminhos(cam, out("distancias.png"))
    S.fig_visualizacao(g, out("visualizacao.png"))

    # ------------------------------------------------------------ Parte II
    if not a.pular_benchmark:
        log("Parte II: benchmark (tempo real com IC)...")
        linhas = B.executar(g, repeticoes=a.repeticoes, semente=a.semente, log=log)
        ajustes = B.ajustar_modelos(linhas)
        with open(out("benchmark_tempos.csv"), "w", newline="") as f:
            cols = ["algoritmo", "n", "m", "m_nao_dirigido", "repeticoes", "media", "desvio", "ic_inf", "ic_sup",
                    "distribuicao", "valor_critico", "rodadas_medias"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(linhas)
        salvar_json(ajustes, out("benchmark_ajustes.json"))
        B.grafico(linhas, ajustes, out("benchmark.png"))
        # tabela para o relatório: algoritmo no maior tamanho medido
        tab2 = []
        for nome, (txt, _) in B.MODELOS.items():
            pts = [l for l in linhas if l["algoritmo"] == nome]
            if not pts:
                continue
            p = max(pts, key=lambda l: l["n"])
            aj = ajustes.get(nome, {})
            tab2.append([nome, txt, p["n"], p["repeticoes"], f"{p['media'] * 1e3:.3f}", f"{p['desvio'] * 1e3:.3f}",
                         f"[{p['ic_inf'] * 1e3:.3f}; {p['ic_sup'] * 1e3:.3f}]", p["distribuicao"],
                         f"{aj.get('expoente_empirico_em_n', float('nan')):.2f}"])
        S.salvar_tabela(tab2, ["Algoritmo", "Teórica", "n", "Rep.", "Média (ms)", "Desvio (ms)", "IC 95% (ms)",
                               "Distribuição", "Exp. emp."], out("tabela_tempos"),
                        "Tempo real por algoritmo (maior tamanho medido).", "tab:tempos",
                        ["l", "l", "r", "r", "r", "r", "l", "l", "r"])
        if a.fw_completo:
            log("Floyd-Warshall no grafo inteiro (1 execução)...")
            arestas_w = [(u, v, 1) for u, v in g.arestas_dirigidas()]
            t1 = time.perf_counter()
            floyd_warshall(g.n, arestas_w)
            salvar_json({"n": g.n, "tempo_s": time.perf_counter() - t1}, out("floyd_warshall_completo.json"))

    log("Concluído. Resultados em: " + os.path.abspath(a.saida))


if __name__ == "__main__":
    main()
