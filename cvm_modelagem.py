# =============================================================================
# MODELAGEM — ETAPA 4 DO CRISP-DM
# TCC: Predição de Indicadores Financeiros com Machine Learning
# =============================================================================
# Entrada : cvm_dataset_treino.csv
#           cvm_dataset_teste.csv
#           cvm_features_selecionadas.csv
#
# Saídas  : modelos/                     — modelos treinados (.joblib)
#           resultados/
#             metricas_comparativo.csv
#             feature_importance_[target].png
#             predicao_vs_real_[target].png
#             cenarios_resultados.csv
#             relatorio_modelagem.txt
#
# Blocos:
#   1 — Carregamento
#   2 — Treinamento com GridSearchCV + k-fold (Pergunta A)
#   3 — Avaliação no teste + comparativo    (Pergunta A)
#   4 — Avaliação por setor                 (Pergunta B)
#   5 — Feature importance (Random Forest)
#   6 — Análise de 3 cenários estratégicos  (Pergunta C)
#   7 — Relatório narrativo para o TCC
# =============================================================================

import os, json, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import Ridge
from sklearn.svm             import SVR
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline        import Pipeline

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

ARQUIVO_TREINO   = 'cvm_dataset_treino.csv'
ARQUIVO_TESTE    = 'cvm_dataset_teste.csv'
ARQUIVO_FEATURES = 'cvm_features_selecionadas.csv'
PASTA_MODELOS    = 'modelos'
PASTA_RESULTADOS = 'resultados'
N_FOLDS          = 5
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

TARGETS = {
    'TARGET_DRE_3.01': 'Receita Líquida (t+1)',
    'TARGET_DRE_3.11': 'Lucro Líquido (t+1)',
    'TARGET_EBITDA'  : 'EBITDA (t+1)',
}

EMPRESAS_ANCORA = {
    'Petróleo'   : '33.000.167/0001-01',  # Petrobras
    'Energia'    : '03.220.438/0001-73',  # Equatorial Energia
    'Varejo'     : '47.960.950/0001-21',  # Magazine Luiza
    'Commodities': '33.592.510/0001-54',  # Vale
    'Tecnologia' : '84.429.695/0001-11',  # WEG
}

# Cenário 1: Expansão com Alavancagem
# Empresa capta nova dívida equivalente a 0,5x o PL. Endividamento sobe,
# cobertura de juros cai, mas abre espaço para crescimento de receita.
# Cenário 2: Eficiência Operacional
# Corte estrutural de 10% nos custos. Margens melhoram sem afetar ativos.
# Cenário 3: Investimento em Capacidade (CAPEX)
# Investimento 30% maior em imobilizado. Pressiona liquidez no curto prazo
# mas impulsiona crescimento de receita em t+1.
CENARIOS = {
    'Expansão com Alavancagem': {
        'descricao': (
            'Captação de nova dívida LP equivalente a 0,5x o Patrimônio Líquido. '
            'Endividamento aumenta, cobertura de juros e liquidez são pressionadas, '
            'mas abre espaço para crescimento via expansão ou aquisições.'
        ),
        'ajustes': {
            'ALAVANCAGEM_DE'   : {'tipo': 'delta', 'valor':  0.50},
            'COBERTURA_JUROS'  : {'tipo': 'fator', 'valor':  0.75},
            'LIQUIDEZ_CORRENTE': {'tipo': 'fator', 'valor':  0.90},
            'ENDIVIDAMENTO_%'  : {'tipo': 'delta', 'valor':  8.00},
            'DRE_3.01_YOY'    : {'tipo': 'delta', 'valor':  5.00},
        }
    },
    'Eficiência Operacional': {
        'descricao': (
            'Programa de redução de 10% nos custos operacionais (CPV + despesas). '
            'Melhora direta nas margens sem impacto no endividamento ou ativos.'
        ),
        'ajustes': {
            'MARGEM_EBIT_%'    : {'tipo': 'delta', 'valor':  3.00},
            'MARGEM_EBITDA_%'  : {'tipo': 'delta', 'valor':  3.50},
            'MARGEM_LIQUIDA_%' : {'tipo': 'delta', 'valor':  2.00},
            'MARGEM_BRUTA_%'   : {'tipo': 'delta', 'valor':  2.50},
            'ROE_%'            : {'tipo': 'delta', 'valor':  2.00},
            'ROA_%'            : {'tipo': 'delta', 'valor':  1.00},
        }
    },
    'Investimento em Capacidade (CAPEX)': {
        'descricao': (
            'Aumento de 30% no investimento em imobilizado. Pressiona liquidez '
            'e FCO no curto prazo, mas impulsiona crescimento de receita em t+1.'
        ),
        'ajustes': {
            'GIRO_ATIVO'          : {'tipo': 'fator', 'valor':  0.88},
            'LIQUIDEZ_IMEDIATA'   : {'tipo': 'fator', 'valor':  0.85},
            'FCO_SOBRE_RECEITA_%' : {'tipo': 'fator', 'valor':  0.90},
            'DRE_3.01_YOY'       : {'tipo': 'delta', 'valor':  8.00},
            'CRESC_ATIVO_YOY_%'  : {'tipo': 'delta', 'valor': 12.00},
        }
    },
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

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def fmt_bi(v):
    if pd.isna(v): return 'N/A'
    return f'R$ {v/1e9:,.2f} bi'

def garantir_pastas():
    os.makedirs(PASTA_MODELOS,    exist_ok=True)
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)


# =============================================================================
# BLOCO 1 — CARREGAMENTO
# =============================================================================

def carregar_dados(log_f):
    log(f'\n{S}\n📂 BLOCO 1 — CARREGAMENTO\n{S}', log_f)

    for arq in [ARQUIVO_TREINO, ARQUIVO_TESTE, ARQUIVO_FEATURES]:
        if not os.path.exists(arq):
            raise FileNotFoundError(f'"{arq}" não encontrado — execute cvm_preparacao.py primeiro.')

    treino   = pd.read_csv(ARQUIVO_TREINO,   sep=';', encoding='utf-8-sig', low_memory=False)
    teste    = pd.read_csv(ARQUIVO_TESTE,    sep=';', encoding='utf-8-sig', low_memory=False)
    feats_df = pd.read_csv(ARQUIVO_FEATURES, sep=';', encoding='utf-8-sig')

    for df in [treino, teste]:
        df['DT_FIM_EXERC'] = pd.to_datetime(df['DT_FIM_EXERC'], errors='coerce')

    log(f'  Treino : {treino.shape[0]} linhas × {treino.shape[1]} colunas', log_f)
    log(f'  Teste  : {teste.shape[0]} linhas × {teste.shape[1]} colunas',   log_f)

    features_por_target = {}
    for tgt_col, tgt_nome in TARGETS.items():
        feats = (feats_df[feats_df['target'] == tgt_nome]
                 .sort_values('rank')['feature'].tolist())
        feats = [f for f in feats if f in treino.columns]
        features_por_target[tgt_col] = feats
        log(f'  {tgt_nome}: {len(feats)} features', log_f)

    return treino, teste, features_por_target


# =============================================================================
# BLOCO 2 — TREINAMENTO COM VALIDAÇÃO CRUZADA
# =============================================================================

def definir_modelos():
    return {
        'Regressão Linear': {
            'modelo': Pipeline([('sc', StandardScaler()), ('reg', Ridge())]),
            'grade' : {'reg__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        },
        'SVR': {
            'modelo': Pipeline([('sc', StandardScaler()),
                                ('svr', SVR(kernel='rbf', max_iter=5000))]),
            'grade' : {'svr__C': [0.1, 1.0, 10.0, 100.0],
                       'svr__epsilon': [0.01, 0.1, 0.5],
                       'svr__gamma': ['scale', 'auto']}
        },
        'Random Forest': {
            'modelo': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'grade' : {'max_depth': [None, 5, 10],
                       'min_samples_leaf': [1, 3, 5],
                       'max_features': ['sqrt', 'log2']}
        },
        'Gradient Boosting': {
            'modelo': GradientBoostingRegressor(random_state=42),
            'grade' : {'n_estimators': [100, 200],
                       'learning_rate': [0.05, 0.1, 0.2],
                       'max_depth': [3, 5],
                       'subsample': [0.8, 1.0]}
        },
    }


def treinar_modelos(treino, features_por_target, log_f):
    log(f'\n{S}\n🤖 BLOCO 2 — TREINAMENTO (k-fold={N_FOLDS} + GridSearchCV)\n{S}', log_f)

    modelos_def  = definir_modelos()
    kf           = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    resultados_cv = {}

    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in treino.columns:
            continue
        feats = features_por_target.get(tgt_col, [])
        if not feats:
            continue

        df_t = (treino[feats + [tgt_col]]
                .replace([np.inf, -np.inf], np.nan).dropna())
        if len(df_t) < N_FOLDS * 2:
            log(f'\n  ⚠️  {tgt_nome}: {len(df_t)} obs — insuficiente.', log_f)
            continue

        X_tr = df_t[feats].values
        y_tr = df_t[tgt_col].values

        log(f'\n{s2}\n  TARGET: {tgt_nome}  ({len(df_t)} obs)\n{s2}', log_f)
        log(f'  {"Algoritmo":<22} {"RMSE_CV":>14} {"MAE_CV":>14} {"MAPE_CV":>10} {"R²_CV":>8}', log_f)
        log(f'  {"─"*22} {"─"*14} {"─"*14} {"─"*10} {"─"*8}', log_f)

        resultados_cv[tgt_col] = {}

        for nome_alg, cfg in modelos_def.items():
            try:
                gs = GridSearchCV(cfg['modelo'], cfg['grade'],
                                  cv=kf, scoring='neg_mean_squared_error',
                                  n_jobs=-1, refit=True)
                gs.fit(X_tr, y_tr)
                melhor = gs.best_estimator_

                cv_res = cross_validate(melhor, X_tr, y_tr, cv=kf,
                                        scoring=['neg_mean_squared_error',
                                                 'neg_mean_absolute_error', 'r2'])
                rmse_cv = np.sqrt(-cv_res['test_neg_mean_squared_error'].mean())
                mae_cv  = -cv_res['test_neg_mean_absolute_error'].mean()
                r2_cv   = cv_res['test_r2'].mean()

                mape_folds = []
                for tri, vli in kf.split(X_tr):
                    melhor.fit(X_tr[tri], y_tr[tri])
                    mape_folds.append(mape(y_tr[vli], melhor.predict(X_tr[vli])))
                mape_cv = np.nanmean(mape_folds)

                melhor.fit(X_tr, y_tr)  # retreina em tudo

                joblib.dump({'modelo': melhor, 'features': feats,
                             'target': tgt_col, 'algoritmo': nome_alg,
                             'params': gs.best_params_},
                            os.path.join(PASTA_MODELOS,
                                         f'{tgt_col}_{nome_alg.replace(" ","_").lower()}.joblib'))

                resultados_cv[tgt_col][nome_alg] = {
                    'modelo': melhor, 'features': feats,
                    'rmse_cv': rmse_cv, 'mae_cv': mae_cv,
                    'mape_cv': mape_cv, 'r2_cv': r2_cv,
                }
                log(f'  {nome_alg:<22} {rmse_cv/1e9:>13.2f}bi {mae_cv/1e9:>13.2f}bi '
                    f'{mape_cv:>9.1f}% {r2_cv:>8.3f}', log_f)

            except Exception as e:
                log(f'  {nome_alg:<22} ERRO: {e}', log_f)

    return resultados_cv


# =============================================================================
# BLOCO 3 — AVALIAÇÃO NO TESTE
# =============================================================================

def avaliar_no_teste(teste, resultados_cv, features_por_target, log_f):
    log(f'\n{S}\n📊 BLOCO 3 — AVALIAÇÃO NO TESTE (Pergunta A)\n{S}', log_f)

    linhas_csv, melhor_por_target = [], {}

    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in resultados_cv:
            continue
        feats = features_por_target.get(tgt_col, [])
        df_te = (teste[feats + [tgt_col]]
                 .replace([np.inf, -np.inf], np.nan).dropna())
        if len(df_te) < 3:
            continue

        X_te, y_te = df_te[feats].values, df_te[tgt_col].values

        log(f'\n{s2}\n  TARGET: {tgt_nome}  (n_teste={len(df_te)})\n{s2}', log_f)
        log(f'  {"Algoritmo":<22} {"RMSE":>14} {"MAE":>14} {"MAPE":>10} {"R²":>8}', log_f)

        melhor_mape = np.inf

        for nome_alg, info in resultados_cv[tgt_col].items():
            pred    = info['modelo'].predict(X_te)
            rmse_te = np.sqrt(mean_squared_error(y_te, pred))
            mae_te  = mean_absolute_error(y_te, pred)
            mape_te = mape(y_te, pred)
            r2_te   = r2_score(y_te, pred)

            log(f'  {nome_alg:<22} {rmse_te/1e9:>13.2f}bi {mae_te/1e9:>13.2f}bi '
                f'{mape_te:>9.1f}% {r2_te:>8.3f}', log_f)

            linhas_csv.append({'target': tgt_nome, 'algoritmo': nome_alg,
                                'rmse_cv': info['rmse_cv']/1e9, 'mae_cv': info['mae_cv']/1e9,
                                'mape_cv': info['mape_cv'], 'r2_cv': info['r2_cv'],
                                'rmse_teste': rmse_te/1e9, 'mae_teste': mae_te/1e9,
                                'mape_teste': mape_te, 'r2_teste': r2_te})

            if mape_te < melhor_mape:
                melhor_mape = mape_te
                melhor_por_target[tgt_col] = {
                    'algoritmo': nome_alg, 'modelo': info['modelo'],
                    'features': feats, 'mape_teste': mape_te,
                    'r2_teste': r2_te, 'pred': pred, 'real': y_te,
                }

        if tgt_col in melhor_por_target:
            log(f'\n  ✅ Melhor: {melhor_por_target[tgt_col]["algoritmo"]} '
                f'(MAPE={melhor_mape:.1f}%)', log_f)
            _plot_real_vs_pred(y_te, melhor_por_target[tgt_col]['pred'],
                               tgt_nome, melhor_por_target[tgt_col]['algoritmo'])

    pd.DataFrame(linhas_csv).to_csv(
        os.path.join(PASTA_RESULTADOS, 'metricas_comparativo.csv'),
        index=False, sep=';', encoding='utf-8-sig')
    log(f'\n  💾 metricas_comparativo.csv salvo.', log_f)
    return melhor_por_target


def _plot_real_vs_pred(y_real, y_pred, tgt_nome, alg_nome):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_real/1e9, y_pred/1e9, alpha=0.7, edgecolors='steelblue',
               facecolors='lightblue', s=60, linewidths=0.8)
    lmin = min(y_real.min(), y_pred.min()) / 1e9
    lmax = max(y_real.max(), y_pred.max()) / 1e9
    ax.plot([lmin, lmax], [lmin, lmax], 'r--', lw=1.2, label='Predição perfeita')
    ax.set(xlabel='Valor Real (R$ bi)', ylabel='Valor Predito (R$ bi)',
           title=f'{tgt_nome}\n{alg_nome}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    nome = f'predicao_vs_real_{tgt_nome.replace(" ","_").replace("/","")[:40]}.png'
    plt.savefig(os.path.join(PASTA_RESULTADOS, nome), dpi=150)
    plt.close()


# =============================================================================
# BLOCO 4 — AVALIAÇÃO POR SETOR (Pergunta B)
# =============================================================================

def avaliar_por_setor(teste, melhor_por_target, log_f):
    log(f'\n{S}\n🏭 BLOCO 4 — ANÁLISE POR SETOR (Pergunta B)\n{S}', log_f)
    resultados_setor = {}

    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in melhor_por_target:
            continue
        info   = melhor_por_target[tgt_col]
        feats  = info['features']
        modelo = info['modelo']

        log(f'\n  {tgt_nome} — {info["algoritmo"]}', log_f)
        log(f'  {"Setor":<16} {"N":>4} {"MAPE":>8} {"MAE (bi)":>12} {"R²":>8}', log_f)

        resultados_setor[tgt_col] = {}
        for setor, grupo in teste.groupby('SETOR'):
            df_s = (grupo[feats + [tgt_col]]
                    .replace([np.inf, -np.inf], np.nan).dropna())
            if len(df_s) < 2:
                continue
            X_s, y_s = df_s[feats].values, df_s[tgt_col].values
            pred = modelo.predict(X_s)
            mape_s = mape(y_s, pred)
            mae_s  = mean_absolute_error(y_s, pred)
            r2_s   = r2_score(y_s, pred) if len(y_s) > 1 else np.nan
            log(f'  {setor:<16} {len(df_s):>4} {mape_s:>7.1f}% {mae_s/1e9:>11.2f}bi {r2_s:>8.3f}', log_f)
            resultados_setor[tgt_col][setor] = {'n': len(df_s), 'mape': mape_s,
                                                 'mae': mae_s, 'r2': r2_s}
    return resultados_setor


# =============================================================================
# BLOCO 5 — FEATURE IMPORTANCE
# =============================================================================

def analisar_feature_importance(resultados_cv, log_f):
    log(f'\n{S}\n🌲 BLOCO 5 — FEATURE IMPORTANCE (Random Forest)\n{S}', log_f)

    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in resultados_cv:
            continue
        if 'Random Forest' not in resultados_cv[tgt_col]:
            continue

        info   = resultados_cv[tgt_col]['Random Forest']
        modelo = info['modelo']
        feats  = info['features']

        rf = modelo
        if not hasattr(rf, 'feature_importances_'):
            log(f'  ⚠️  {tgt_nome}: feature_importances_ não disponível.', log_f)
            continue

        imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
        top15 = imp.head(15)

        log(f'\n  {tgt_nome} — Top 15:', log_f)
        for feat, v in top15.items():
            log(f'    {feat:<45} {v:.4f}  {"█"*int(v*100)}', log_f)

        fig, ax = plt.subplots(figsize=(9, 6))
        cores = ['#1F3864' if i < 5 else '#2E5090' if i < 10 else '#7FA7D9'
                 for i in range(len(top15))]
        ax.barh(range(len(top15)), top15.values[::-1], color=cores[::-1])
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15.index[::-1], fontsize=9)
        ax.set(xlabel='Importância', title=f'Feature Importance — RF\n{tgt_nome}')
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        nome = f'feature_importance_{tgt_col}.png'
        plt.savefig(os.path.join(PASTA_RESULTADOS, nome), dpi=150)
        plt.close()
        log(f'  💾 {nome}', log_f)


# =============================================================================
# BLOCO 6 — ANÁLISE DE CENÁRIOS (Pergunta C)
# =============================================================================

def aplicar_cenario(linha_base, ajustes):
    linha = linha_base.copy()
    for kpi, cfg in ajustes.items():
        if kpi not in linha.index or pd.isna(linha[kpi]):
            continue
        if cfg['tipo'] == 'delta':
            linha[kpi] = linha[kpi] + cfg['valor']
        elif cfg['tipo'] == 'fator':
            linha[kpi] = linha[kpi] * cfg['valor']
    return linha


def interpretar_com_llm(empresa, setor, cenario_nome, cenario_desc,
                        kpis_base, resultado_comparativo, api_key):
    if not api_key:
        return '[LLM não ativado — configure ANTHROPIC_API_KEY para ativar.]'
    try:
        import urllib.request
        linhas_res = '\n'.join([
            f'  {t}: {fmt_bi(vb)} → {fmt_bi(vc)} ({((vc-vb)/abs(vb)*100):+.1f}%)'
            for t, (vb, vc) in resultado_comparativo.items()
            if not pd.isna(vb) and not pd.isna(vc) and vb != 0
        ])
        kpis_str = '\n'.join([f'  {k}: {v:.4g}' for k, v in kpis_base.items()
                               if isinstance(v, (int, float)) and not pd.isna(v)])
        prompt = (
            f'Você é analista financeiro sênior especializado em empresas brasileiras.\n\n'
            f'Empresa: {empresa} | Setor: {setor}\n'
            f'Cenário: {cenario_nome}\n{cenario_desc}\n\n'
            f'KPIs atuais (seleção):\n{kpis_str}\n\n'
            f'Impacto previsto pelo modelo de ML:\n{linhas_res}\n\n'
            f'Forneça análise executiva em 4 parágrafos (máximo): '
            f'(1) impacto geral, (2) pontos positivos e riscos, '
            f'(3) comparação com perfil típico do setor {setor}, '
            f'(4) recomendação objetiva. Use linguagem para conselho de administração.'
        )
        payload = json.dumps({'model': 'claude-sonnet-4-20250514', 'max_tokens': 900,
                              'messages': [{'role': 'user', 'content': prompt}]}).encode()
        req = urllib.request.Request(
            'https://api.anthropic.com/v1/messages', data=payload,
            headers={'Content-Type': 'application/json',
                     'x-api-key': api_key, 'anthropic-version': '2023-06-01'})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())['content'][0]['text']
    except Exception as e:
        return f'[Erro LLM: {e}]'


def analisar_cenarios(treino, teste, melhor_por_target, log_f):
    log(f'\n{S}\n🎯 BLOCO 6 — CENÁRIOS ESTRATÉGICOS (Pergunta C)\n{S}', log_f)

    df_completo = pd.concat([treino, teste], ignore_index=True)
    df_completo['DT_FIM_EXERC'] = pd.to_datetime(df_completo['DT_FIM_EXERC'], errors='coerce')
    todos_resultados = []

    for setor, cnpj in EMPRESAS_ANCORA.items():
        df_emp = df_completo[df_completo['CNPJ_CIA'] == cnpj].sort_values('DT_FIM_EXERC')
        if df_emp.empty:
            log(f'\n  ⚠️  {setor} ({cnpj}): não encontrada.', log_f)
            continue

        nome_emp   = df_emp['DENOM_CIA'].iloc[-1]
        linha_base = df_emp.iloc[-1]

        log(f'\n{s2}\n  {nome_emp}  |  {setor}\n{s2}', log_f)

        # Baseline
        baseline_preds = {}
        for tgt_col in TARGETS:
            if tgt_col not in melhor_por_target:
                baseline_preds[tgt_col] = np.nan
                continue
            info  = melhor_por_target[tgt_col]
            feats = [f for f in info['features'] if f in linha_base.index]
            X_b   = np.nan_to_num(linha_base[feats].values.reshape(1, -1))
            baseline_preds[tgt_col] = info['modelo'].predict(X_b)[0]

        log(f'\n  Baseline:', log_f)
        for tgt_col, tgt_nome in TARGETS.items():
            log(f'    {tgt_nome:<30} {fmt_bi(baseline_preds.get(tgt_col, np.nan))}', log_f)

        for nome_cen, cfg_cen in CENARIOS.items():
            log(f'\n  [{nome_cen}]', log_f)
            linha_cen = aplicar_cenario(linha_base, cfg_cen['ajustes'])
            cen_preds = {}
            resultado_comp = {}

            for tgt_col, tgt_nome in TARGETS.items():
                if tgt_col not in melhor_por_target:
                    cen_preds[tgt_col] = np.nan
                    continue
                info  = melhor_por_target[tgt_col]
                feats = [f for f in info['features'] if f in linha_cen.index]
                X_c   = np.nan_to_num(linha_cen[feats].values.reshape(1, -1))
                cen_preds[tgt_col] = info['modelo'].predict(X_c)[0]

                vb, vc = baseline_preds.get(tgt_col, np.nan), cen_preds[tgt_col]
                resultado_comp[tgt_nome] = (vb, vc)
                var = ((vc - vb) / abs(vb) * 100) if not pd.isna(vb) and vb != 0 else np.nan
                log(f'    {tgt_nome:<30} {fmt_bi(vb)} → {fmt_bi(vc)} ({var:+.1f}%)', log_f)

                todos_resultados.append({
                    'empresa': nome_emp, 'setor': setor, 'cnpj': cnpj,
                    'cenario': nome_cen, 'target': tgt_nome,
                    'valor_baseline': vb, 'valor_cenario': vc, 'variacao_pct': var,
                })

            kpis_base = {k: v for k, v in linha_base.items()
                         if isinstance(v, (int, float)) and not pd.isna(v)}
            interp = interpretar_com_llm(nome_emp, setor, nome_cen,
                                         cfg_cen['descricao'], kpis_base,
                                         resultado_comp, ANTHROPIC_API_KEY)
            log(f'\n  Interpretação:\n  {interp}', log_f)

    if todos_resultados:
        pd.DataFrame(todos_resultados).to_csv(
            os.path.join(PASTA_RESULTADOS, 'cenarios_resultados.csv'),
            index=False, sep=';', encoding='utf-8-sig')
        log(f'\n  💾 cenarios_resultados.csv salvo.', log_f)

    return todos_resultados


# =============================================================================
# BLOCO 7 — RELATÓRIO PARA O TCC
# =============================================================================

def gerar_relatorio_tcc(melhor_por_target, resultados_setor, todos_cenarios, log_f):
    log(f'\n{S}\n📄 BLOCO 7 — RELATÓRIO PARA O TCC\n{S}', log_f)

    arq = os.path.join(PASTA_RESULTADOS, 'relatorio_modelagem.txt')
    with open(arq, 'w', encoding='utf-8') as f:
        f.write(S + '\nRELATÓRIO DE MODELAGEM — INSUMOS PARA O TCC\n' + S + '\n\n')

        f.write('== PERGUNTA A — MELHOR ALGORITMO POR TARGET ==\n\n')
        for tgt_col, tgt_nome in TARGETS.items():
            if tgt_col not in melhor_por_target:
                continue
            i = melhor_por_target[tgt_col]
            f.write(f'{tgt_nome}:\n')
            f.write(f'  Melhor algoritmo: {i["algoritmo"]}\n')
            f.write(f'  MAPE teste: {i["mape_teste"]:.1f}%\n')
            f.write(f'  R² teste  : {i["r2_teste"]:.3f}\n\n')

        f.write('\n== PERGUNTA B — DESEMPENHO POR SETOR ==\n\n')
        for tgt_col, setores in resultados_setor.items():
            tgt_nome = TARGETS.get(tgt_col, tgt_col)
            f.write(f'{tgt_nome}:\n')
            for setor, m in sorted(setores.items(), key=lambda x: x[1]['mape']):
                f.write(f'  {setor:<16} MAPE={m["mape"]:.1f}%  R²={m["r2"]:.3f}\n')
            f.write('\n')

        f.write('\n== PERGUNTA C — CENÁRIOS ESTRATÉGICOS ==\n\n')
        df_cen = pd.DataFrame(todos_cenarios) if todos_cenarios else pd.DataFrame()
        if not df_cen.empty:
            for empresa, grp in df_cen.groupby('empresa'):
                f.write(f'{empresa} ({grp["setor"].iloc[0]}):\n')
                for cen, sub in grp.groupby('cenario'):
                    f.write(f'  [{cen}]\n')
                    for _, row in sub.iterrows():
                        f.write(f'    {row["target"]:<30} '
                                f'{fmt_bi(row["valor_baseline"])} → '
                                f'{fmt_bi(row["valor_cenario"])} '
                                f'({row["variacao_pct"]:+.1f}%)\n')
                f.write('\n')

        f.write('\n== LIMITAÇÕES ==\n\n')
        limitacoes = [
            'Empresas com DFC pelo Método Direto têm EBITDA = NaN (sem D&A disponível).',
            'Cada empresa tem 5–10 períodos DFP — dataset pequeno aumenta variância.',
            'Cenários aplicam ajustes lineares; efeitos de segunda ordem não capturados.',
            'Modelo não incorpora choques macroeconômicos futuros (câmbio, juros, etc.).',
            'O Gradient Boosting pode sofrer overfitting em datasets muito pequenos.',
        ]
        for l in limitacoes:
            f.write(f'- {l}\n')

    log(f'  💾 relatorio_modelagem.txt salvo.', log_f)


# =============================================================================
# EXECUÇÃO
# =============================================================================

def executar_modelagem():
    garantir_pastas()
    arq_log = os.path.join(PASTA_RESULTADOS, 'log_modelagem.txt')

    with open(arq_log, 'w', encoding='utf-8') as log_f:
        treino, teste, features_por_target = carregar_dados(log_f)
        resultados_cv    = treinar_modelos(treino, features_por_target, log_f)
        melhor_p_target  = avaliar_no_teste(teste, resultados_cv, features_por_target, log_f)
        resultados_setor = avaliar_por_setor(teste, melhor_p_target, log_f)
        analisar_feature_importance(resultados_cv, log_f)
        todos_cenarios   = analisar_cenarios(treino, teste, melhor_p_target, log_f)
        gerar_relatorio_tcc(melhor_p_target, resultados_setor, todos_cenarios, log_f)

        log(f'\n{S}', log_f)
        log(f'✅ CONCLUÍDO — modelos em ./{PASTA_MODELOS}/  |  resultados em ./{PASTA_RESULTADOS}/', log_f)
        log(f'   Para ativar LLM: export ANTHROPIC_API_KEY="sua-chave"', log_f)
        log(S, log_f)


if __name__ == '__main__':
    executar_modelagem()
