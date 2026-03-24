# =============================================================================
# TREINAMENTO DOS MODELOS — ETAPA 4a DO CRISP-DM (Modelagem)
# TCC: Predição de Indicadores Financeiros com Machine Learning
# =============================================================================
# Entrada : cvm_dataset_treino.csv
#           cvm_features_selecionadas.csv
#
# Saídas  : modelos/
#               [target]_[algoritmo].joblib   — modelo serializado
#               treino_metadata.json          — hiperparâmetros e métricas de CV
#           logs/
#               log_treino.txt
#
# Responsabilidade: APENAS treinar e salvar modelos.
# Não avalia no teste, não gera gráficos, não aplica cenários.
# Isso garante que retreinar não apague resultados de avaliações anteriores.
#
# Algoritmos:
#   - Regressão Linear (Ridge)    — baseline interpretável
#   - SVR (kernel RBF)            — relações não lineares, datasets moderados
#   - Random Forest               — ensemble, fornece feature importance
#   - Gradient Boosting           — boosting sequencial, teto de desempenho
#
# Uso: python cvm_treino.py
# Dependências: pandas, numpy, scikit-learn, joblib
# =============================================================================

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model    import Ridge
from sklearn.svm             import SVR
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline        import Pipeline

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

ARQUIVO_TREINO   = 'cvm_dataset_treino.csv'
ARQUIVO_FEATURES = 'cvm_features_selecionadas.csv'
PASTA_MODELOS    = 'modelos'
PASTA_LOGS       = 'logs'
N_FOLDS          = 5
RANDOM_STATE     = 42

TARGETS = {
    'TARGET_DRE_3.01': 'Receita Líquida (t+1)',
    'TARGET_DRE_3.11': 'Lucro Líquido (t+1)',
    'TARGET_EBITDA'  : 'EBITDA (t+1)',
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
    """MAPE robusto: ignora zeros no denominador."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def garantir_pastas():
    for p in [PASTA_MODELOS, PASTA_LOGS]:
        os.makedirs(p, exist_ok=True)


# =============================================================================
# DEFINIÇÃO DOS MODELOS E GRADES DE HIPERPARÂMETROS
# =============================================================================

def definir_modelos():
    """
    Retorna dicionário com os 4 algoritmos, seus estimadores base e grades
    de busca de hiperparâmetros para o GridSearchCV.

    Regressão Linear (Ridge):
        Alpha controla a regularização L2 — essencial em dados financeiros
        onde múltiplos KPIs são derivados das mesmas contas (multicolinearidade).
        Pipeline com StandardScaler garante que escala não influencie os coeficientes.

    SVR (RBF):
        C: trade-off entre margem e erros de treinamento.
        Epsilon: largura do tubo insensível a erros — quanto menor, mais preciso
        mas mais suscetível a overfitting.
        Gamma: largura do kernel RBF — 'scale' usa 1/(n_features * Var(X)).

    Random Forest:
        max_depth=None permite crescimento completo das árvores (bagging controla
        a variância). min_samples_leaf evita folhas com 1 observação.
        max_features='sqrt' é padrão para regressão — reduz correlação entre árvores.

    Gradient Boosting:
        learning_rate pequeno + mais estimadores = melhor generalização.
        subsample < 1.0 aplica stochastic gradient boosting, reduzindo variância.
        max_depth=3 é conservador para datasets pequenos.
    """
    return {
        'Regressão Linear': {
            'estimador': Pipeline([
                ('scaler', StandardScaler()),
                ('ridge',  Ridge())
            ]),
            'grade': {
                'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            },
        },
        'SVR': {
            'estimador': Pipeline([
                ('scaler', StandardScaler()),
                ('svr',    SVR(kernel='rbf', max_iter=10000))
            ]),
            'grade': {
                'svr__C':       [0.1, 1.0, 10.0, 100.0],
                'svr__epsilon': [0.01, 0.05, 0.1, 0.5],
                'svr__gamma':   ['scale', 'auto'],
            },
        },
        'Random Forest': {
            'estimador': RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'grade': {
                'max_depth':        [None, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5],
                'max_features':     ['sqrt', 'log2'],
            },
        },
        'Gradient Boosting': {
            'estimador': GradientBoostingRegressor(
                random_state=RANDOM_STATE
            ),
            'grade': {
                'n_estimators':  [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth':     [3, 5],
                'subsample':     [0.7, 0.8, 1.0],
            },
        },
    }


# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================

def carregar_dados(log_f):
    log(f'\n{S}\n📂 CARREGAMENTO DOS DADOS DE TREINO\n{S}', log_f)

    for arq in [ARQUIVO_TREINO, ARQUIVO_FEATURES]:
        if not os.path.exists(arq):
            raise FileNotFoundError(
                f'"{arq}" não encontrado. '
                'Execute cvm_preparacao.py antes deste script.'
            )

    treino   = pd.read_csv(ARQUIVO_TREINO,   sep=';', encoding='utf-8-sig', low_memory=False)
    feats_df = pd.read_csv(ARQUIVO_FEATURES, sep=';', encoding='utf-8-sig')

    log(f'  Treino : {treino.shape[0]} linhas × {treino.shape[1]} colunas', log_f)

    features_por_target = {}
    for tgt_col, tgt_nome in TARGETS.items():
        feats = (feats_df[feats_df['target'] == tgt_nome]
                 .sort_values('rank')['feature']
                 .tolist())
        feats = [f for f in feats if f in treino.columns]
        features_por_target[tgt_col] = feats
        log(f'  {tgt_nome}: {len(feats)} features selecionadas', log_f)

    return treino, features_por_target


# =============================================================================
# TREINAMENTO PRINCIPAL
# =============================================================================

def treinar(treino, features_por_target, log_f):
    """
    Para cada target × algoritmo:
      1. GridSearchCV (k-fold interno) → melhores hiperparâmetros
      2. Retreina com melhores parâmetros em todo o conjunto de treino
      3. cross_validate com k-fold externo → métricas de generalização
      4. Salva modelo serializado + metadados JSON

    O uso de GridSearchCV com k-fold interno + cross_validate com k-fold
    externo (nested cross-validation) garante que as métricas reportadas
    não são infladas pela seleção de hiperparâmetros.
    """
    log(f'\n{S}\n🤖 TREINAMENTO COM GRIDSEARCHCV + K-FOLD (k={N_FOLDS})\n{S}', log_f)

    modelos_def = definir_modelos()
    kf_externo  = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    kf_interno  = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + 1)

    metadata = {}   # será salvo como JSON ao final

    for tgt_col, tgt_nome in TARGETS.items():
        if tgt_col not in treino.columns:
            log(f'\n  ⚠️  {tgt_nome}: coluna ausente no dataset — pulando.', log_f)
            continue

        feats = features_por_target.get(tgt_col, [])
        if not feats:
            log(f'\n  ⚠️  {tgt_nome}: sem features — pulando.', log_f)
            continue

        df_t = (treino[feats + [tgt_col]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna())

        n_obs = len(df_t)
        if n_obs < N_FOLDS * 3:
            log(f'\n  ⚠️  {tgt_nome}: apenas {n_obs} obs — insuficiente para {N_FOLDS}-fold.', log_f)
            continue

        X = df_t[feats].values
        y = df_t[tgt_col].values

        log(f'\n{s2}', log_f)
        log(f'  TARGET : {tgt_nome}', log_f)
        log(f'  Obs    : {n_obs}  |  Features: {len(feats)}', log_f)
        log(f'{s2}', log_f)
        log(f'  {"Algoritmo":<22} {"RMSE_CV":>14} {"MAE_CV":>14} {"MAPE_CV":>10} '
            f'{"R²_CV":>8}  Melhores parâmetros', log_f)
        log(f'  {"─"*22} {"─"*14} {"─"*14} {"─"*10} {"─"*8}  {"─"*30}', log_f)

        metadata[tgt_col] = {
            'target_nome'    : tgt_nome,
            'n_obs_treino'   : n_obs,
            'n_features'     : len(feats),
            'features'       : feats,
            'algoritmos'     : {},
        }

        for nome_alg, cfg in modelos_def.items():

            try:
                # ── Passo 1: GridSearchCV (k-fold interno) ────────────────
                gs = GridSearchCV(
                    estimator  = cfg['estimador'],
                    param_grid = cfg['grade'],
                    cv         = kf_interno,
                    scoring    = 'neg_mean_squared_error',
                    n_jobs     = -1,
                    refit      = True,
                )
                gs.fit(X, y)
                melhor_estimador = gs.best_estimator_

                # ── Passo 2: Validação cruzada externa com melhores params ─
                cv_res = cross_validate(
                    melhor_estimador, X, y,
                    cv      = kf_externo,
                    scoring = ['neg_mean_squared_error',
                               'neg_mean_absolute_error',
                               'r2'],
                    return_train_score = False,
                )

                rmse_cv = float(np.sqrt(-cv_res['test_neg_mean_squared_error'].mean()))
                mae_cv  = float(-cv_res['test_neg_mean_absolute_error'].mean())
                r2_cv   = float(cv_res['test_r2'].mean())

                # MAPE precisa de predições explícitas (sklearn não tem scorer nativo)
                mape_folds = []
                for idx_tr, idx_val in kf_externo.split(X):
                    melhor_estimador.fit(X[idx_tr], y[idx_tr])
                    pred_val = melhor_estimador.predict(X[idx_val])
                    mape_folds.append(mape_score(y[idx_val], pred_val))
                mape_cv = float(np.nanmean(mape_folds))

                # ── Passo 3: Treina final com todos os dados de treino ─────
                melhor_estimador.fit(X, y)

                # ── Passo 4: Salva modelo ─────────────────────────────────
                nome_arquivo = (
                    f'{tgt_col}_{nome_alg.replace(" ", "_").lower()}.joblib'
                )
                joblib.dump(
                    {
                        'modelo'      : melhor_estimador,
                        'features'    : feats,
                        'target_col'  : tgt_col,
                        'target_nome' : tgt_nome,
                        'algoritmo'   : nome_alg,
                        'params'      : gs.best_params_,
                        'metricas_cv' : {
                            'rmse': rmse_cv,
                            'mae' : mae_cv,
                            'mape': mape_cv,
                            'r2'  : r2_cv,
                        },
                    },
                    os.path.join(PASTA_MODELOS, nome_arquivo)
                )

                # ── Registro no metadata ──────────────────────────────────
                metadata[tgt_col]['algoritmos'][nome_alg] = {
                    'arquivo_joblib' : nome_arquivo,
                    'melhores_params': gs.best_params_,
                    'metricas_cv'    : {
                        'rmse': rmse_cv,
                        'mae' : mae_cv,
                        'mape': mape_cv,
                        'r2'  : r2_cv,
                    },
                }

                params_str = str(gs.best_params_)[:40]
                log(f'  {nome_alg:<22} {rmse_cv/1e9:>13.3f}bi {mae_cv/1e9:>13.3f}bi '
                    f'{mape_cv:>9.1f}% {r2_cv:>8.3f}  {params_str}', log_f)

            except Exception as e:
                log(f'  {nome_alg:<22} ❌ ERRO: {e}', log_f)
                metadata[tgt_col]['algoritmos'][nome_alg] = {'erro': str(e)}

    # ── Salva metadata JSON ───────────────────────────────────────────────────
    arq_meta = os.path.join(PASTA_MODELOS, 'treino_metadata.json')
    with open(arq_meta, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log(f'\n  💾 Metadata salvo: {arq_meta}', log_f)

    return metadata


# =============================================================================
# EXECUÇÃO
# =============================================================================

def executar_treino():
    garantir_pastas()
    arq_log = os.path.join(PASTA_LOGS, 'log_treino.txt')

    with open(arq_log, 'w', encoding='utf-8') as log_f:
        log_f.write('LOG DE TREINAMENTO — CVM\n')
        log_f.write('TCC: Predição de Indicadores Financeiros com ML\n')
        log_f.write(f'{S}\n\n')

        treino, features_por_target = carregar_dados(log_f)
        metadata = treinar(treino, features_por_target, log_f)

        # Sumário final
        log(f'\n{S}', log_f)
        log(f'✅ TREINAMENTO CONCLUÍDO', log_f)
        log(f'   Modelos salvos em: ./{PASTA_MODELOS}/', log_f)
        for tgt_col, info in metadata.items():
            log(f'\n   {info["target_nome"]}:', log_f)
            for alg, dados in info['algoritmos'].items():
                if 'metricas_cv' in dados:
                    m = dados['metricas_cv']
                    log(f'     {alg:<22} MAPE={m["mape"]:5.1f}%  R²={m["r2"]:6.3f}', log_f)
        log(f'\n   Próximo passo: python cvm_avaliacao.py', log_f)
        log(S, log_f)

    print(f'\n📄 Log salvo em: {arq_log}')
    return metadata


if __name__ == '__main__':
    executar_treino()