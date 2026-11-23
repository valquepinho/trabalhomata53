"""Geração de tabelas (CSV e LaTeX) e figuras a partir dos resultados."""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import zeta

from .benchmark import amostra_bfs
from .parte3 import ccdf


def fmt(x):
    if isinstance(x, (bool, np.bool_)):
        return "sim" if x else "não"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", ".")
    if isinstance(x, (float, np.floating)):
        return f"{x:.4g}"
    return str(x)


def tex(s):
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#"),
                 ("^", "\\^{}")):
        s = s.replace(a, b)
    for a, b in (("→", "$\\to$"), ("≥", "$\\ge$"), ("≤", "$\\le$"), ("·", "$\\cdot$"), ("±", "$\\pm$"),
                 ("≈", "$\\approx$"), ("σ", "$\\sigma$"), ("α", "$\\alpha$"), ("≠", "$\\ne$")):
        s = s.replace(a, b)
    return s


def salvar_tabela(linhas, cabecalho, base, legenda, rotulo, larguras=None):
    """Salva `base`.csv e `base`.tex (tabular com última coluna em parbox)."""
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cabecalho)
        w.writerows(linhas)
    larguras = larguras or ["l"] * (len(cabecalho) - 1) + ["p{6cm}"]
    with open(base + ".tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\\small\n")
        f.write("\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{" + "".join(larguras) + "}\n\\hline\n")
        f.write(" & ".join("\\textbf{" + tex(c) + "}" for c in cabecalho) + " \\\\\n\\hline\n")
        for l in linhas:
            f.write(" & ".join(tex(str(c)) for c in l) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}}\n")
        f.write(f"\\caption{{{tex(legenda)}}}\n\\label{{{rotulo}}}\n\\end{{table}}\n")


# ----------------------------------------------------------------------
def tabela_parte1(g, info, gr, dens, comp, cam, tri, eul):
    d, u, w = cam["dirigido_alcancaveis"], cam["nao_dirigido_maior_WCC"], cam["dirigido_maior_SCC"]
    sim, nao = "Sim", "Não"

    def r(v):
        return fmt(v) if v is not None else "não exato (amostra)"

    def gr3(k):
        s = gr[k]
        return f"min {s['min']} / máx {s['max']} / média {s['media']:.3f}"

    tam_f = comp["tamanhos_fracas"]
    tam_s = comp["tamanhos_fortes"]
    singletons = sum(1 for t in tam_s if t == 1)
    linhas = [
        ["Número de vértices", sim, fmt(g.n), "Após remoção de laços/duplicatas"],
        ["Número de arestas", sim,
         f"{fmt(g.m)} dirigidas; {fmt(g.m_und)} não-dirigidas",
         f"{fmt(g.m - g.m_und)} pares recíprocos (u→v e v→u)"],
        ["Grau mínimo/máximo/médio (entrada)", sim, gr3("entrada"), "Grau de entrada"],
        ["Grau mínimo/máximo/médio (saída)", sim, gr3("saida"), "Grau de saída"],
        ["Grau mínimo/máximo/médio (não-dirigido)", sim, gr3("nao_dirigido"), "Vizinhos distintos"],
        ["Distribuição de graus", sim, "ver figura e graus.csv", "Histograma e CCDF log-log"],
        ["Densidade", sim, f"dirigida {dens['dirigida']:.4g}; não-dirigida {dens['nao_dirigida']:.4g}",
         "m/(n(n-1)) e 2m/(n(n-1))"],
        ["Nº de componentes conexas", sim,
         f"{comp['n_fracas']} fracas; {fmt(comp['n_fortes'])} fortes",
         "Fracas: BFS na versão não-dirigida; fortes: Tarjan"],
        ["Tamanho de cada componente", sim,
         f"fracas: {tam_f[:6]}; fortes: maior {tam_s[0]}, {fmt(singletons)} unitárias",
         "Lista completa em componentes.csv"],
        ["Diâmetro", sim,
         f"dirigido (alcançáveis) {d['diametro']}; não-dir. (maior WCC) {u['diametro']}; maior SCC {w['diametro']}",
         "Grafo dirigido não é fortemente conexo: distância infinita entre muitos pares"],
        ["Raio", sim,
         f"não-dir. (maior WCC) {r(u['raio'])}; maior SCC {r(w['raio'])}",
         "Não definido no grafo dirigido completo (excentricidade infinita); calculado nos dois subgrafos conexos"],
        ["Comprimento médio dos caminhos", sim,
         f"dirigido {d['comprimento_medio']:.4f}; não-dir. {u['comprimento_medio']:.4f}; SCC {w['comprimento_medio']:.4f}",
         "Média sobre pares alcançáveis (s≠t)"],
        ["Diâmetro efetivo (percentil 90)", sim,
         f"dirigido {d['diametro_efetivo_90']:.3f}; não-dir. {u['diametro_efetivo_90']:.3f}",
         "Medida extra, comparável ao valor do SNAP"],
        ["Coeficiente de clustering médio", sim,
         f"{tri['clustering_medio_todos_vertices']:.5f} (todos os vértices); "
         f"{tri['clustering_medio_grau_ge_2']:.5f} (grau ≥ 2); transitividade {tri['transitividade_global']:.5f}",
         "Versão não-dirigida; vértices com grau < 2 têm C = 0"],
        ["Número de triângulos", sim, fmt(tri["triangulos"]), "Viável (grafo esparso); versão não-dirigida"],
        ["Visualização do grafo", sim, "visualizacao.png", "Versão reduzida: subgrafo de 300 vértices por BFS a partir do hub"],
        ["Eulerianidade", sim,
         f"circuito (dir.): {fmt(eul['dirigido_tem_circuito_euleriano'])}; "
         f"{fmt(eul['vertices_desbalanceados_dirigido'])} vértices desbalanceados",
         "Parte II"],
    ]
    return linhas


# ----------------------------------------------------------------------
def fig_graus(vetores, pot, caminho):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    rot = {"entrada": "entrada", "saida": "saída", "nao_dirigido": "não-dirigido"}
    for k in ("entrada", "saida", "nao_dirigido"):
        x = np.asarray(vetores[k])
        vals, cont = np.unique(x[x >= 1], return_counts=True)
        axs[0].loglog(vals, cont / len(x), "o", ms=4, alpha=0.7, label=rot[k])
        v2, p2 = ccdf(x)
        axs[1].loglog(v2, p2, "-o", ms=3, alpha=0.7, label=rot[k])
    axs[0].set_xlabel("grau k")
    axs[0].set_ylabel("P(k)")
    axs[0].set_title("Distribuição de graus (log-log)")
    axs[1].set_xlabel("grau k")
    axs[1].set_ylabel("P(X ≥ k)")
    axs[1].set_title("CCDF e ajuste de lei de potência (não-dirigido)")
    p = pot.get("nao_dirigido", {})
    if "alpha" in p:
        xs = np.arange(p["xmin"], max(vetores["nao_dirigido"]) + 1)
        frac = p["n_cauda"] / len(vetores["nao_dirigido"])
        axs[1].loglog(xs, frac * zeta(p["alpha"], xs) / zeta(p["alpha"], p["xmin"]), "k--",
                      label=f"lei de potência α={p['alpha']:.2f}, xmin={p['xmin']}")
    for ax in axs:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_caminhos(cam, caminho):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    nomes = {"dirigido_alcancaveis": "Dirigido (pares alcançáveis)",
             "nao_dirigido_maior_WCC": "Não-dirigido (maior WCC)",
             "dirigido_maior_SCC": "Dirigido (maior SCC)"}
    for ax, (k, t) in zip(axs, nomes.items()):
        h = cam[k]["histograma_distancias"]
        ds = sorted(h)
        tot = sum(h.values())
        ax.bar(ds, [h[d] / tot for d in ds])
        ax.set_title(f"{t}\nL = {cam[k]['comprimento_medio']:.2f}, diâmetro = {cam[k]['diametro']}", fontsize=9)
        ax.set_xlabel("distância")
        ax.set_ylabel("fração de pares")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_smallworld(sw, caminho):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    rotulos = ["real", "ER", "preserva graus", "analítico"]
    C = [sw["C_real"], sw["ER"]["C"], sw["preserva_graus"]["C"], sw["analitico"]["C"]]
    L = [sw["L_real"], sw["ER"]["L"], sw["preserva_graus"]["L"], sw["analitico"]["L"]]
    axs[0].bar(rotulos, C)
    axs[0].set_yscale("log")
    axs[0].set_title("Clustering médio C")
    axs[1].bar(rotulos, L)
    axs[1].set_title("Comprimento médio L")
    for ax in axs:
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_robustez(rb, caminho):
    """Gera boxplots das 30 repetições aleatórias (com média ± desvio padrão)
    para as quatro métricas do Apêndice C, comparando com base e ataques direcionados."""
    metricas = [
        ("frac_maior_WCC_entre_restantes", "Métrica A: Maior WCC / restantes"),
        ("n_componentes_fracas", "Métrica B: Nº de componentes (WCC)"),
        ("L_maior_WCC", "Métrica C: Distância média L (WCC)"),
        ("frac_nos_isolados", "Métrica D: Fração de nós isolados"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (chave, titulo) in zip(axs.flat, metricas):
        al = rb["aleatorio"][chave]
        vals = al.get("valores", [al["media"]])
        media = al["media"]
        desvio = al["desvio"]

        # Boxplot das repetições aleatórias na posição 2
        bp = ax.boxplot(
            [vals], positions=[2], widths=0.42, patch_artist=True,
            boxprops=dict(facecolor="#a6cee3", edgecolor="#1f78b4", linewidth=1.2),
            medianprops=dict(color="#08306b", linewidth=1.6),
            whiskerprops=dict(color="#1f78b4", linewidth=1.2),
            capprops=dict(color="#1f78b4", linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, markerfacecolor="#1f78b4", alpha=0.6),
        )
        # Média ± desvio padrão sobreposto ao boxplot
        ax.errorbar(
            [2], [media], yerr=[desvio], fmt="D", color="#d95f02",
            ecolor="#d95f02", elinewidth=1.8, capsize=6, markersize=6,
            zorder=5, label="Aleatório: média ± DP"
        )

        # Pontos dos demais cenários (base, alvo grau, alvo intermediação)
        ax.scatter([1], [rb["base"][chave]], color="#33a02c", marker="s", s=65, zorder=5, label="Base (sem remoção)")
        ax.scatter([3], [rb["alvo_grau"][chave]], color="#e31a1c", marker="^", s=75, zorder=5, label="Alvo: maior grau")
        ax.scatter([4], [rb["alvo_intermediacao"][chave]], color="#6a3d9a", marker="v", s=75, zorder=5, label="Alvo: intermediação")

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Base", "Aleatório 5%\n(30 rep.)", "Alvo:\ngrau", "Alvo:\nintermediação"], fontsize=9)
        ax.set_title(titulo, fontsize=10.5, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


def fig_visualizacao(g, caminho, k=300):
    import networkx as nx  # usado apenas para calcular o layout e desenhar
    h = amostra_bfs(g, k)
    G = nx.Graph()
    G.add_nodes_from(range(h.n))
    G.add_edges_from(h.arestas_nao_dirigidas())
    pos = nx.spring_layout(G, seed=0, k=0.25, iterations=200)
    grau = np.array([G.degree(v) for v in G])
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.5, ax=ax)
    nc = nx.draw_networkx_nodes(G, pos, node_size=10 + 12 * grau, node_color=grau, cmap="viridis", ax=ax)
    fig.colorbar(nc, ax=ax, shrink=0.7, label="grau")
    ax.set_title(f"Subgrafo reduzido: {h.n} vértices (BFS a partir do vértice de maior grau)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
