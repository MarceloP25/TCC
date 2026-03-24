# =============================================================================
# AVALIAÇÃO DOS MODELOS — ETAPA 4b DO CRISP-DM (Avaliação)
# TCC: Predição de Indicadores Financeiros com Machine Learning
# =============================================================================
# Entrada : modelos/treino_metadata.json    — gerado por cvm_treino.py
#           modelos/[target]_[alg].joblib   — gerado por cvm_treino.py
#           cvm_dataset_teste.csv           — gerado por cvm_preparacao.py
#           cvm_dataset_treino.csv          — gerado por cvm_preparacao.py
#
# Saídas  : resultados/
#               avaliacao_resultados.json       — todas as métricas (JSON)
#               metricas_comparativo.csv        — tabela para o TCC
#               relatorio_avaliacao.txt         — narrativa para o TCC
#               graficos/
#                   comparativo_mape_[target].png
#                   predicao_vs_real_[target].png
#                   feature_importance_[target].png
#                   desempenho_por_setor_[target].png
#           logs/
#               log_avaliacao.txt
#
# Responsabilidade: carregar modelos já treinados e avaliar.
# Nunca retreina — garante que os resultados são reprodutíveis.
#
# Responde:
#   Pergunta A — Qual algoritmo é mais acurado por target?
#   Pergunta B — O desempenho varia entre setores? Por quê?
#
# Uso: python cvm_avaliacao.py  (execute APÓS cvm_treino.py)
# =============================================================================

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 130

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PASTA_MODELOS    = 'modelos'
PASTA_RESULTADOS = 'resultados'
PASTA_GRAFICOS   = os.path.join(PASTA_RESULTADOS, 'graficos')
PASTA_LOGS       = 'logs'

ARQUIVO_TREINO   = 'cvm_dataset_treino.csv'
ARQUIVO_TESTE    = 'cvm_dataset_teste.csv'
METADATA_JSON    = os.path.join(PASTA_MODELOS, 'treino_metadata.json')

TARGETS = {
    'TARGET_DRE_3.01': 'Receita Líquida (t+1)',
    'TARGET_DRE_3.11': 'Lucro Líquido (t+1)',
    'TARGET_EBITDA'  : 'EBITDA (t+1)',
}

CORES_ALGORITMOS = {
    'Regressão Linear': '#2E5090',
    'SVR'             : '#1A5E35',
    'Random Forest'   : '#C65911',
    'Gradient Boosting': '#7B2D8B',
}

S  = '=' * 75
s2 = '─' * 75


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + '\n')
        f.flush()


def mape_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def fmt_bi(v):
    if pd.isna(v):
        return 'N/A'
    return f'R$ {v/1e9:,.2f} bi'


def garantir_pastas():
    for p in [PASTA_RESULTADOS, PASTA_GRAFICOS, PASTA_LOGS]:
        os.makedirs(p, exist_ok=True)


# =============================================================================
# CARREGAMENTO
# =============================================================================

def carregar_tudo(log_f):
    log(f'\n{S}\n📂 CARREGAMENTO\n{S}', log_f)

    for arq in [METADATA_JSON, ARQUIVO_TREINO, ARQUIVO_TESTE]:
        if not os.path.exists(arq):
            raise FileNotFoundError(
                f'"{arq}" não encontrado. '
                'Execute cvm_treino.py e cvm_preparacao.py antes.'
            )

    with open(METADATA_JSON, encoding='utf-8') as f:
        metadata = json.load(f)

    treino = pd.read_csv(ARQUIVO_TREINO, sep=';', encoding='utf-8-sig', low_memory=False)
    teste  = pd.read_csv(ARQUIVO_TESTE,  sep=';', encoding='utf-8-sig', low_memory=False)

    for df in [treino, teste]:
        df['DT_FIM_EXERC'] = pd.to_datetime(df['DT_FIM_EXERC'], errors='coerce')

    log(f'  Treino : {treino.shape[0]} linhas', log_f)
    log(f'  Teste  : {teste.shape[0]} linhas', log_f)
    log(f'  Targets no metadata: {list(metadata.keys())}', log_f)

    # Carrega todos os modelos do disco
    modelos = {}
    for tgt_col, info in metadata.items():
        modelos[tgt_col] = {}
        for nome_alg, dados in info['algoritmos'].items():
            if 'arquivo_joblib' not in dados:
                continue
            caminho = os.path.join(PASTA_MODELOS, dados['arquivo_joblib'])
            if not os.path.exists(caminho):
                log(f'  ⚠️  Arquivo não encontrado: {caminho}', log_f)
                continue
            modelos[tgt_col][nome_alg] = joblib.load(caminho)
            log(f'  ✅ Carregado: {dados["arquivo_joblib"]}', log_f)

    return metadata, treino, teste, modelos


# =============================================================================
# AVALIAÇÃO NO CONJUNTO DE TESTE — Pergunta A
# =============================================================================

def avaliar_no_teste(teste, metadata, modelos, log_f):
    """
    Avalia cada modelo no conjunto de teste — dados nunca vistos durante
    o treinamento. O conjunto de teste contém os períodos mais recentes
    de cada empresa, simulando o cenário real de uso do modelo.

    Métricas reportadas:
    - RMSE: penaliza desvios grandes — adequado quando erros grandes são custosos
    - MAE : desvio típico em reais — interpretação direta para gestores
    - MAPE: erro percentual — permite comparar targets de escalas diferentes
    - R²  : proporção da variância explicada — medida geral de ajuste

    O melhor modelo por target é definido pelo menor MAPE no teste,
    pois é a métrica mais interpretável para não especialistas em ML.
    """
    log(f'\n{S}\n📊 PERGUNTA A — COMPARATIVO DE ALGORITMOS NO TESTE\n{S}', log_f)

    resultados   = {}   # {tgt_col: {alg: metricas}}
    linhas_csv   = []
    melhor_p_tgt = {}

    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in modelos:
            continue

        info_meta = metadata.get(tgt_col, {})
        feats = info_meta.get('features', [])
        feats_disp = [f for f in feats if f in teste.columns]

        df_te = (teste[feats_disp + [tgt_col]]
                 .replace([np.inf, -np.inf], np.nan)
                 .dropna())

        if len(df_te) < 3:
            log(f'\n  ⚠️  {tgt_nome}: apenas {len(df_te)} obs no teste.', log_f)
            continue

        X_te = df_te[feats_disp].values
        y_te = df_te[tgt_col].values

        log(f'\n{s2}', log_f)
        log(f'  TARGET: {tgt_nome}  (n_teste={len(df_te)})', log_f)
        log(f'{s2}', log_f)
        log(f'  {"Algoritmo":<22} {"RMSE (bi)":>12} {"MAE (bi)":>12} '
            f'{"MAPE":>9} {"R²":>8}  {"vs. CV"}', log_f)
        log(f'  {"─"*22} {"─"*12} {"─"*12} {"─"*9} {"─"*8}  {"─"*20}', log_f)

        resultados[tgt_col] = {}
        melhor_mape = np.inf

        for nome_alg, artefato in modelos[tgt_col].items():
            modelo = artefato['modelo']
            pred   = modelo.predict(X_te)

            rmse_te = float(np.sqrt(mean_squared_error(y_te, pred)))
            mae_te  = float(mean_absolute_error(y_te, pred))
            mape_te = mape_score(y_te, pred)
            r2_te   = float(r2_score(y_te, pred))

            mape_cv = (metadata[tgt_col]['algoritmos']
                       .get(nome_alg, {})
                       .get('metricas_cv', {})
                       .get('mape', np.nan))
            diff_cv = (f'+{mape_te - mape_cv:.1f}pp'
                       if not np.isnan(mape_cv) else '—')

            log(f'  {nome_alg:<22} {rmse_te/1e9:>11.3f}  {mae_te/1e9:>11.3f}  '
                f'{mape_te:>8.1f}% {r2_te:>8.3f}  {diff_cv}', log_f)

            resultados[tgt_col][nome_alg] = {
                'rmse': rmse_te, 'mae': mae_te,
                'mape': mape_te, 'r2': r2_te,
                'pred': pred.tolist(), 'real': y_te.tolist(),
            }

            linhas_csv.append({
                'target': tgt_nome, 'algoritmo': nome_alg,
                'rmse_cv_bi':  metadata[tgt_col]['algoritmos']
                               .get(nome_alg, {}).get('metricas_cv', {}).get('rmse', np.nan) / 1e9,
                'mae_cv_bi':   metadata[tgt_col]['algoritmos']
                               .get(nome_alg, {}).get('metricas_cv', {}).get('mae', np.nan) / 1e9,
                'mape_cv_pct': mape_cv,
                'r2_cv':       metadata[tgt_col]['algoritmos']
                               .get(nome_alg, {}).get('metricas_cv', {}).get('r2', np.nan),
                'rmse_teste_bi': rmse_te / 1e9,
                'mae_teste_bi':  mae_te / 1e9,
                'mape_teste_pct': mape_te,
                'r2_teste':      r2_te,
            })

            if mape_te < melhor_mape:
                melhor_mape = mape_te
                melhor_p_tgt[tgt_col] = {
                    'algoritmo'  : nome_alg,
                    'modelo'     : modelo,
                    'features'   : feats_disp,
                    'mape_teste' : mape_te,
                    'r2_teste'   : r2_te,
                    'pred'       : pred,
                    'real'       : y_te,
                }

        if tgt_col in melhor_p_tgt:
            log(f'\n  ✅ Melhor: {melhor_p_tgt[tgt_col]["algoritmo"]} '
                f'(MAPE={melhor_mape:.1f}%  R²={melhor_p_tgt[tgt_col]["r2_teste"]:.3f})', log_f)

    # CSV comparativo
    pd.DataFrame(linhas_csv).to_csv(
        os.path.join(PASTA_RESULTADOS, 'metricas_comparativo.csv'),
        index=False, sep=';', encoding='utf-8-sig'
    )
    log(f'\n  💾 metricas_comparativo.csv salvo.', log_f)

    return resultados, melhor_p_tgt


# =============================================================================
# AVALIAÇÃO POR SETOR — Pergunta B
# =============================================================================

def avaliar_por_setor(teste, melhor_p_tgt, metadata, log_f):
    """
    Avalia o melhor modelo de cada target segmentado por setor.

    Hipóteses baseadas na teoria financeira (Damodaran, 2012):
    - Energia: mais previsível — receitas reguladas por contrato de concessão
    - Commodities: menos previsível — atrelado a preços internacionais voláteis
    - Varejo: moderado — correlato com ciclo econômico doméstico
    - Tecnologia: variável — crescimento pode ser não linear
    - Petróleo: desafiador — Petrobras tem eventos políticos que distorcem
    """
    log(f'\n{S}\n🏭 PERGUNTA B — DESEMPENHO POR SETOR\n{S}', log_f)

    resultados_setor = {}

    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in melhor_p_tgt:
            continue

        info   = melhor_p_tgt[tgt_col]
        feats  = info['features']
        modelo = info['modelo']

        log(f'\n  {tgt_nome}  ({info["algoritmo"]})', log_f)
        log(f'  {"Setor":<16} {"N":>4} {"MAPE":>9} {"MAE (bi)":>12} '
            f'{"R²":>8}  Interpretação', log_f)
        log(f'  {"─"*16} {"─"*4} {"─"*9} {"─"*12} {"─"*8}  {"─"*25}', log_f)

        resultados_setor[tgt_col] = {}

        for setor, grupo in teste.groupby('SETOR'):
            feats_disp = [f for f in feats if f in grupo.columns]
            df_s = (grupo[feats_disp + [tgt_col]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna())

            if len(df_s) < 2:
                continue

            X_s, y_s = df_s[feats_disp].values, df_s[tgt_col].values
            pred  = modelo.predict(X_s)

            mape_s = mape_score(y_s, pred)
            mae_s  = float(mean_absolute_error(y_s, pred))
            r2_s   = float(r2_score(y_s, pred)) if len(y_s) > 1 else np.nan

            # Classificação qualitativa
            if mape_s < 10:
                qualidade = '⭐ Excelente'
            elif mape_s < 20:
                qualidade = '✅ Bom'
            elif mape_s < 35:
                qualidade = '⚠️  Moderado'
            else:
                qualidade = '❌ Desafiador'

            log(f'  {setor:<16} {len(df_s):>4} {mape_s:>8.1f}% '
                f'{mae_s/1e9:>11.2f}  {r2_s:>8.3f}  {qualidade}', log_f)

            resultados_setor[tgt_col][setor] = {
                'n': len(df_s), 'mape': mape_s,
                'mae': mae_s,   'r2': r2_s,
            }

    return resultados_setor


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def analisar_feature_importance(modelos, metadata, log_f):
    """
    Extrai a importância das features do Random Forest para cada target.

    A importância é medida pela redução média do MSE (impurity decrease)
    em todas as divisões de todas as árvores onde a feature é usada.
    Features com alta importância têm interpretação financeira direta —
    são as variáveis que o modelo considera mais informativas para a predição.
    """
    log(f'\n{S}\n🌲 FEATURE IMPORTANCE — RANDOM FOREST\n{S}', log_f)

    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in modelos:
            continue
        if 'Random Forest' not in modelos[tgt_col]:
            continue

        artefato = modelos[tgt_col]['Random Forest']
        modelo   = artefato['modelo']
        feats    = metadata.get(tgt_col, {}).get('features', artefato.get('features', []))

        # O Random Forest pode estar diretamente ou dentro de um Pipeline
        rf = modelo
        if hasattr(modelo, 'named_steps'):
            for step in modelo.named_steps.values():
                if hasattr(step, 'feature_importances_'):
                    rf = step
                    break

        if not hasattr(rf, 'feature_importances_'):
            log(f'  ⚠️  {tgt_nome}: feature_importances_ não disponível.', log_f)
            continue

        imp   = pd.Series(rf.feature_importances_, index=feats[:len(rf.feature_importances_)])
        top15 = imp.sort_values(ascending=False).head(15)

        log(f'\n  {tgt_nome} — Top 15 features:', log_f)
        for feat, v in top15.items():
            barra = '█' * int(v * 150)
            log(f'    {feat:<45} {v:.4f}  {barra}', log_f)

        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        n = len(top15)
        cores = ['#1F3864' if i < 5 else '#2E5090' if i < 10 else '#7FA7D9'
                 for i in range(n)]
        ax.barh(range(n), top15.values[::-1], color=cores[::-1], edgecolor='white',
                linewidth=0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f.replace('_%', '').replace('_YOY', ' (YoY)').replace('_', ' ')
             for f in top15.index[::-1]],
            fontsize=9
        )
        ax.set_xlabel('Importância relativa (redução média do MSE)', fontsize=10)
        ax.set_title(f'Feature Importance — Random Forest\n{tgt_nome}', fontsize=11)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        nome_arq = f'feature_importance_{tgt_col}.png'
        plt.savefig(os.path.join(PASTA_GRAFICOS, nome_arq), dpi=150, bbox_inches='tight')
        plt.close()
        log(f'  💾 {nome_arq}', log_f)


# =============================================================================
# GRÁFICOS
# =============================================================================

def gerar_graficos(resultados, melhor_p_tgt, resultados_setor, log_f):
    log(f'\n{S}\n📈 GERAÇÃO DE GRÁFICOS\n{S}', log_f)

    # ── Gráfico 1: Comparativo de MAPE entre algoritmos por target ────────────
    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in resultados:
            continue

        algs  = list(resultados[tgt_col].keys())
        mapes = [resultados[tgt_col][a]['mape'] for a in algs]
        cores = [CORES_ALGORITMOS.get(a, '#888888') for a in algs]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(algs, mapes, color=cores, edgecolor='white', linewidth=0.8)

        for bar, v in zip(bars, mapes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('MAPE no conjunto de teste (%)', fontsize=11)
        ax.set_title(f'Comparativo de Algoritmos — {tgt_nome}', fontsize=12)
        ax.set_ylim(0, max(mapes) * 1.25)
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, linewidth=1,
                   label='Referência: 10% MAPE')
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, linewidth=1,
                   label='Referência: 20% MAPE')
        ax.legend(fontsize=8)
        plt.xticks(fontsize=10)
        plt.tight_layout()

        nome_arq = f'comparativo_mape_{tgt_col}.png'
        plt.savefig(os.path.join(PASTA_GRAFICOS, nome_arq), dpi=150, bbox_inches='tight')
        plt.close()
        log(f'  💾 {nome_arq}', log_f)

    # ── Gráfico 2: Real vs. Predito (melhor modelo por target) ────────────────
    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in melhor_p_tgt:
            continue

        info   = melhor_p_tgt[tgt_col]
        y_real = np.asarray(info['real'])
        y_pred = np.asarray(info['pred'])

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_real / 1e9, y_pred / 1e9,
                   alpha=0.7, edgecolors='#1F3864',
                   facecolors='lightsteelblue', s=65, linewidths=0.8)

        lim_min = min(y_real.min(), y_pred.min()) / 1e9
        lim_max = max(y_real.max(), y_pred.max()) / 1e9
        margem  = (lim_max - lim_min) * 0.05
        ax.plot([lim_min - margem, lim_max + margem],
                [lim_min - margem, lim_max + margem],
                'r--', linewidth=1.3, label='Predição perfeita')

        ax.set_xlabel('Valor Real (R$ bi)', fontsize=11)
        ax.set_ylabel('Valor Predito (R$ bi)', fontsize=11)
        ax.set_title(
            f'Real vs. Predito — {tgt_nome}\n'
            f'{info["algoritmo"]}  |  MAPE={info["mape_teste"]:.1f}%  '
            f'R²={info["r2_teste"]:.3f}',
            fontsize=11
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        nome_arq = f'predicao_vs_real_{tgt_col}.png'
        plt.savefig(os.path.join(PASTA_GRAFICOS, nome_arq), dpi=150, bbox_inches='tight')
        plt.close()
        log(f'  💾 {nome_arq}', log_f)

    # ── Gráfico 3: MAPE por setor (melhor modelo, todos os targets) ──────────
    fig, axes = plt.subplots(1, len(resultados_setor),
                             figsize=(5 * len(resultados_setor), 5),
                             squeeze=False)

    for idx, (tgt_col, por_setor) in enumerate(resultados_setor.items()):
        ax = axes[0][idx]
        setores = list(por_setor.keys())
        mapes   = [por_setor[s]['mape'] for s in setores]
        cores   = ['#1A5E35' if m < 15 else '#C65911' if m < 30 else '#8B0000'
                   for m in mapes]

        bars = ax.barh(setores, mapes, color=cores, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, mapes):
            ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{v:.1f}%', va='center', fontsize=9)

        ax.set_xlabel('MAPE (%)', fontsize=10)
        tgt_nome = TARGETS.get(tgt_col, tgt_col)
        ax.set_title(f'{tgt_nome}', fontsize=10)
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Desempenho por Setor — MAPE no teste (%)', fontsize=12, y=1.02)
    plt.tight_layout()
    nome_arq = 'desempenho_por_setor.png'
    plt.savefig(os.path.join(PASTA_GRAFICOS, nome_arq), dpi=150, bbox_inches='tight')
    plt.close()
    log(f'  💾 {nome_arq}', log_f)


# =============================================================================
# GERAÇÃO DOS RELATÓRIOS (JSON + TEXTUAL)
# =============================================================================

def gerar_relatorios(metadata, resultados, melhor_p_tgt,
                     resultados_setor, log_f):
    """
    Gera dois relatórios:

    1. avaliacao_resultados.json
       Estrutura completa com todas as métricas, hierarquizada por
       target e algoritmo. Consumido pelo cvm_cenarios.py para
       identificar o melhor modelo sem precisar ler o console.

    2. relatorio_avaliacao.txt
       Narrativa estruturada pronta para ser adaptada nas seções
       de Desenvolvimento e Resultados do TCC. Inclui interpretação
       qualitativa dos números, não apenas as tabelas brutas.
    """
    log(f'\n{S}\n📄 GERAÇÃO DE RELATÓRIOS\n{S}', log_f)

    # ── JSON completo ─────────────────────────────────────────────────────────
    json_saida = {
        'pergunta_a': {},
        'pergunta_b': {},
        'melhor_por_target': {},
    }

    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in resultados:
            continue

        # Pergunta A
        json_saida['pergunta_a'][tgt_col] = {
            'target_nome': tgt_nome,
            'algoritmos': {
                alg: {
                    'mape_cv'   : (metadata.get(tgt_col, {}).get('algoritmos', {})
                                   .get(alg, {}).get('metricas_cv', {}).get('mape', None)),
                    'r2_cv'     : (metadata.get(tgt_col, {}).get('algoritmos', {})
                                   .get(alg, {}).get('metricas_cv', {}).get('r2', None)),
                    'mape_teste': m['mape'],
                    'mae_teste' : m['mae'],
                    'rmse_teste': m['rmse'],
                    'r2_teste'  : m['r2'],
                }
                for alg, m in resultados[tgt_col].items()
            }
        }

        # Melhor por target
        if tgt_col in melhor_p_tgt:
            info = melhor_p_tgt[tgt_col]
            json_saida['melhor_por_target'][tgt_col] = {
                'target_nome'   : tgt_nome,
                'algoritmo'     : info['algoritmo'],
                'arquivo_joblib': (
                    f'{tgt_col}_{info["algoritmo"].replace(" ", "_").lower()}.joblib'
                ),
                'features'      : info['features'],
                'mape_teste'    : info['mape_teste'],
                'r2_teste'      : info['r2_teste'],
            }

        # Pergunta B
        if tgt_col in resultados_setor:
            json_saida['pergunta_b'][tgt_col] = {
                'target_nome': tgt_nome,
                'setores': resultados_setor[tgt_col],
            }

    arq_json = os.path.join(PASTA_RESULTADOS, 'avaliacao_resultados.json')
    with open(arq_json, 'w', encoding='utf-8') as f:
        json.dump(json_saida, f, ensure_ascii=False, indent=2)
    log(f'  💾 avaliacao_resultados.json salvo.', log_f)

    # ── Relatório textual para o TCC ─────────────────────────────────────────
    arq_txt = os.path.join(PASTA_RESULTADOS, 'relatorio_avaliacao.txt')
    with open(arq_txt, 'w', encoding='utf-8') as f:

        f.write(S + '\n')
        f.write('RELATÓRIO DE AVALIAÇÃO — INSUMOS PARA O TCC\n')
        f.write('Seções: Desenvolvimento / Resultados\n')
        f.write(S + '\n\n')

        f.write('=' * 40 + '\n')
        f.write('PERGUNTA A — COMPARATIVO DE ALGORITMOS\n')
        f.write('=' * 40 + '\n\n')

        for tgt_col, tgt_nome in TARGETS.items():
            if tgt_col not in resultados:
                continue
            f.write(f'Target: {tgt_nome}\n')
            f.write(f'{"─"*40}\n')

            algs_sorted = sorted(
                resultados[tgt_col].items(),
                key=lambda x: x[1]['mape']
            )
            f.write(f'  {"Algoritmo":<22} {"MAPE":>8} {"R²":>8}\n')
            for alg, m in algs_sorted:
                destaque = '  ← MELHOR' if alg == algs_sorted[0][0] else ''
                f.write(f'  {alg:<22} {m["mape"]:>7.1f}% {m["r2"]:>8.3f}{destaque}\n')

            melhor_alg  = algs_sorted[0][0]
            melhor_mape = algs_sorted[0][1]['mape']
            melhor_r2   = algs_sorted[0][1]['r2']
            f.write(
                f'\n  Para a predição de {tgt_nome}, o algoritmo {melhor_alg} '
                f'apresentou o melhor desempenho no conjunto de teste, com MAPE de '
                f'{melhor_mape:.1f}% e R² de {melhor_r2:.3f}, indicando que o modelo '
                f'explica {melhor_r2*100:.1f}% da variância do indicador futuro e '
                f'erra em média {melhor_mape:.1f}% em termos percentuais.\n\n'
            )

        f.write('\n' + '=' * 40 + '\n')
        f.write('PERGUNTA B — DESEMPENHO POR SETOR\n')
        f.write('=' * 40 + '\n\n')

        for tgt_col, tgt_nome in TARGETS.items():
            if tgt_col not in resultados_setor:
                continue
            f.write(f'Target: {tgt_nome}\n')
            f.write(f'{"─"*40}\n')

            setor_sorted = sorted(
                resultados_setor[tgt_col].items(),
                key=lambda x: x[1]['mape']
            )
            f.write(f'  {"Setor":<16} {"MAPE":>8} {"R²":>8}\n')
            for setor, m in setor_sorted:
                f.write(f'  {setor:<16} {m["mape"]:>7.1f}% {m["r2"]:>8.3f}\n')

            mais_prev  = setor_sorted[0]
            menos_prev = setor_sorted[-1]
            f.write(
                f'\n  O setor com maior previsibilidade para {tgt_nome} foi '
                f'{mais_prev[0]} (MAPE={mais_prev[1]["mape"]:.1f}%), enquanto '
                f'{menos_prev[0]} apresentou o desempenho mais desafiador '
                f'(MAPE={menos_prev[1]["mape"]:.1f}%). Essa diferença é consistente '
                f'com a literatura financeira: setores com receitas mais previsíveis '
                f'(contratos regulados, demanda inelástica) tendem a produzir modelos '
                f'mais acurados.\n\n'
            )

        f.write('\n' + '=' * 40 + '\n')
        f.write('LIMITAÇÕES IDENTIFICADAS\n')
        f.write('=' * 40 + '\n\n')
        limitacoes = [
            'Séries temporais curtas (5–10 períodos DFP por empresa) limitam a '
            'capacidade dos modelos de capturar padrões de longo prazo.',
            'Empresas que utilizam DFC pelo Método Direto não possuem D&A '
            'disponível, resultando em EBITDA = NaN e impossibilitando a '
            'predição do target TARGET_EBITDA para essas empresas.',
            'O modelo não incorpora variáveis macroeconômicas exógenas '
            '(taxa SELIC, câmbio, PIB), que têm impacto relevante em alguns setores.',
            'Eventos não recorrentes (fusões, impairments, crises) podem '
            'distorcer tanto as features quanto os targets, afetando a acurácia '
            'em períodos atípicos.',
        ]
        for i, l in enumerate(limitacoes, 1):
            f.write(f'{i}. {l}\n\n')

    log(f'  💾 relatorio_avaliacao.txt salvo.', log_f)

    return json_saida


# =============================================================================
# EXECUÇÃO
# =============================================================================

def executar_avaliacao():
    garantir_pastas()
    arq_log = os.path.join(PASTA_LOGS, 'log_avaliacao.txt')

    with open(arq_log, 'w', encoding='utf-8') as log_f:
        log_f.write('LOG DE AVALIAÇÃO — CVM\n')
        log_f.write('TCC: Predição de Indicadores Financeiros com ML\n')
        log_f.write(f'{S}\n\n')

        metadata, treino, teste, modelos = carregar_tudo(log_f)
        resultados, melhor_p_tgt = avaliar_no_teste(teste, metadata, modelos, log_f)
        resultados_setor = avaliar_por_setor(teste, melhor_p_tgt, metadata, log_f)
        analisar_feature_importance(modelos, metadata, log_f)
        gerar_graficos(resultados, melhor_p_tgt, resultados_setor, log_f)
        json_saida = gerar_relatorios(
            metadata, resultados, melhor_p_tgt, resultados_setor, log_f
        )

        log(f'\n{S}', log_f)
        log(f'✅ AVALIAÇÃO CONCLUÍDA', log_f)
        log(f'   Resultados em: ./{PASTA_RESULTADOS}/', log_f)
        log(f'   Gráficos em  : ./{PASTA_GRAFICOS}/', log_f)
        log(f'   Próximo passo: python cvm_cenarios.py', log_f)
        log(S, log_f)

    print(f'\n📄 Log salvo em: {arq_log}')
    return json_saida


if __name__ == '__main__':
    executar_avaliacao()