# Análise estrutural do grafo p2p-Gnutella08

Trabalho de grafos: Parte I (análise estrutural), Parte II (algoritmos da disciplina com tempo real e IC)
e Parte III (small-world, lei de potência, robustez). Dataset: https://snap.stanford.edu/data/p2p-Gnutella08.html
(grafo dirigido, 6.301 vértices, 20.777 arestas).

## Como executar
```bash
pip install -r requirements.txt            # numpy, scipy, matplotlib, networkx
# 1) o dado já está em data/p2p-Gnutella08.txt.gz (se precisar baixar de novo:)
#    curl -O https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz && mv p2p-Gnutella08.txt.gz data/
# 2) rode tudo (alguns minutos)
python run_all.py --entrada data/p2p-Gnutella08.txt.gz --saida results
# 3) testes (compara as implementações com o networkx)
python -m unittest tests.test_algoritmos -v
```
Opções: `--repeticoes 30`, `--max-fontes N` (amostra de fontes de BFS, mais rápido), `--pular-benchmark`,
`--pular-parte3`, `--fw-completo` (Floyd-Warshall 1x no grafo inteiro, alguns minutos).
## Origem e verificação do dado
`data/p2p-Gnutella08.txt.gz` (SNAP, grafo dirigido, 6.301 vértices, 20.777 arestas). Obtido por download direto no site oficial do SNAP
(`https://snap.stanford.edu/data/p2p-Gnutella08.html`); as contagens e o cabeçalho conferem
com a página oficial. SHA-256: `cb3019062be21e55b03f19a73e88edf34693bafb68ae581621cc7a037fa604a9`.
Tratamento: 0 laços, 0 arestas duplicadas, 0 pares recíprocos (o cabeçalho do arquivo diz que cada par não ordenado é salvo uma vez).

## Relatório
`relatorio/relatorio.tex` (lê as figuras e tabelas de `results/`). Compile com `pdflatex relatorio.tex` (2x) dentro de `relatorio/`.
Os campos em vermelho no PDF (nome, matrícula, cálculo de determinação do grafo, link do GitHub) devem ser preenchidos.

`data/sintetico.txt` é um grafo FALSO só para testar o pipeline (`scripts/gerar_sintetico.py`); não use seus números.

## Estrutura
- `src/grafo.py` leitura e tratamento (remove laços e duplicatas; reindexa).
- `src/algoritmos.py` BFS, DFS, Euler, Dijkstra, Bellman-Ford, Floyd-Warshall, Tarjan, Kruskal, Prim (feitos à mão).
- `src/metricas.py` Parte I. `src/parte3.py` Parte III. `src/benchmark.py` tempos, IC e ajuste à complexidade.
- `src/saidas.py` tabelas (CSV/LaTeX) e figuras. `relatorio/relatorio.tex` esqueleto do relatório.
- networkx é usado só nos testes e para desenhar o layout da visualização.

## Decisões de modelagem
- Grau, componentes fortes (Tarjan) e BFS/DFS usam o grafo dirigido; clustering, triângulos, MST e small-world usam a versão não-dirigida.
- Distâncias em 3 contextos: dirigido (pares alcançáveis), não-dirigido na maior componente fraca, dirigido na maior componente forte. O raio só é reportado nos dois últimos (no dirigido completo é indefinido).
- Pesos: o grafo é sem pesos; Dijkstra, Bellman-Ford e Floyd-Warshall usam peso 1. Kruskal/Prim usam pesos inteiros aleatórios (1..10, semente fixa) para a MST não ser trivial; o grafo é desconexo, então o resultado é uma floresta.
- Tempo real: n repetições (padrão 30). n ≥ 30 usa IC Normal (z), n < 30 usa t-Student; desvio amostral (ddof=1). Medições em subgrafos de tamanho crescente (BFS a partir do hub). Bellman-Ford usa parada antecipada, então o observado fica abaixo do pior caso O(nm); Floyd-Warshall tem o laço em k explícito e o (i,j) vetorizado com numpy.
- Small-world: σ = (C/C_rand)/(L/L_rand) contra G(n,m), contra modelo que preserva graus e contra as aproximações analíticas.
- Lei de potência: MLE discreto com xmin por KS (Clauset et al.), comparada a exponencial e log-normal (Vuong).
- Robustez: remoção de 5% aleatória (30 tentativas) e dos 5% de maior grau e de maior intermediação (Brandes amostrado, 500 fontes). Ataque estático.

## Saídas em `results/`
`tabela_medidas.{csv,tex}`, `tabela_tempos.{csv,tex}`, `parte1.json`, `parte3.json`, `benchmark_*.{csv,json,png}`,
`graus.csv`, `componentes.csv`, `distribuicao_graus.png`, `distancias.png`, `smallworld.png`, `robustez.png`, `visualizacao.png`.
