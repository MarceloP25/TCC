# =============================================================================
# PREPARAÇÃO DOS DADOS CVM — ETAPA 3 DO CRISP-DM
# TCC: Predição de Indicadores Financeiros com Machine Learning
# =============================================================================
# Entrada : cvm_eda_com_kpis.csv  (saída da EDA — cvm_processamento.py)
# Saídas  : cvm_dataset_final.csv        — dataset completo preparado
#           cvm_dataset_treino.csv        — split de treino (por tempo)
#           cvm_dataset_teste.csv         — split de teste  (por tempo)
#           cvm_features_selecionadas.csv — ranking de features por target
#           cvm_prep_relatorio.txt        — relatório textual de cada decisão
#
# Etapas implementadas neste script:
#   1. Carregamento e auditoria inicial
#   2. Filtragem: apenas DFP (anual/auditado) para modelagem de séries temporais
#   3. Remoção de colunas com >LIMIAR_NULOS de nulos (parametrizável)
#   4. Imputação de nulos pela mediana do setor (contas brutas e KPIs)
#   5. Winsorização por setor (IQR × FATOR_WINS) para outliers extremos
#   6. Engenharia de features: variáveis YoY (crescimento ano a ano)
#   7. Codificação do setor (one-hot) para uso como feature nos modelos
#   8. Criação dos targets: shift(-1) por empresa dentro do mesmo TIPO_DOC
#   9. Seleção de features: correlação de Pearson + Recursive Feature Elimination
#  10. Split temporal treino/teste (sem data leakage)
#  11. Salvamento dos artefatos e relatório de decisões
# =============================================================================
# Dependências: pandas, numpy, scikit-learn
# Uso         : python cvm_preparacao.py
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

# =============================================================================
# PARÂMETROS — ajuste aqui sem precisar alterar o código
# =============================================================================

ARQUIVO_ENTRADA = 'cvm_eda_com_kpis.csv'

# Targets principais a prever
# Cada chave é o nome da coluna de target; o valor é o nome legível.
TARGETS = {
    'DRE_3.01': 'Receita Líquida (t+1)',
    'DRE_3.11': 'Lucro Líquido (t+1)',
    'EBITDA':   'EBITDA (t+1)',
}

# Colunas de identificação — nunca usadas como features
COLUNAS_ID = [
    'CNPJ_CIA', 'DENOM_CIA', 'DT_FIM_EXERC', 'ANO_REF',
    'TIPO_DOC', 'SETOR',
]

# Limiar de nulos: colunas com mais do que este percentual são removidas
LIMIAR_NULOS = 0.80  # 80 %

# Fator IQR para winsorização (3× = conservador, preserva variações reais)
FATOR_WINS = 3.0

# Número de features a selecionar via RFE por target
N_FEATURES_RFE = 15

# Proporção do dataset reservada para teste (últimos períodos por empresa)
PROPORCAO_TESTE = 0.25  # ~25 % dos períodos mais recentes

# Contas brutas prioritárias para imputação antes dos KPIs
CONTAS_BRUTAS_PRIORITARIAS = [
    'DRE_3.01', 'DRE_3.02', 'DRE_3.03', 'DRE_3.04', 'DRE_3.05',
    'DRE_3.06', 'DRE_3.07', 'DRE_3.08', 'DRE_3.11',
    'BPA_1',    'BPA_1.01', 'BPA_1.01.01', 'BPA_1.01.02',
    'BPA_1.01.03', 'BPA_1.01.04', 'BPA_1.02', 'BPA_1.02.03', 'BPA_1.02.04',
    'BPP_2',    'BPP_2.01', 'BPP_2.01.04', 'BPP_2.02', 'BPP_2.02.01',
    'BPP_2.03',
    'DFC_MD_6.01', 'DFC_MD_6.02', 'DFC_MD_6.03',
    'DFC_MI_6.01', 'DFC_MI_6.02', 'DFC_MI_6.03',
    'DA_TOTAL', 'EBITDA',
]

# KPIs calculados na EDA
KPIS = [
    'MARGEM_BRUTA_%', 'MARGEM_EBIT_%', 'MARGEM_LIQUIDA_%', 'MARGEM_EBITDA_%',
    'ROE_%', 'ROA_%',
    'LIQUIDEZ_CORRENTE', 'LIQUIDEZ_IMEDIATA',
    'ENDIVIDAMENTO_%', 'ALAVANCAGEM_DE', 'DIVIDA_LIQUIDA', 'COBERTURA_JUROS',
    'GIRO_ATIVO',
    'FCO_SOBRE_RECEITA_%', 'FCO_SOBRE_LL_%',
]

S  = '=' * 75
s2 = '─' * 75

# =============================================================================
# UTILITÁRIOS
# =============================================================================

def log(msg, arquivo_log):
    print(msg)
    arquivo_log.write(msg + '\n')


def fmt_pct(v):
    return f'{v*100:.1f}%'


# =============================================================================
# ETAPA 1 — CARREGAMENTO E AUDITORIA INICIAL
# =============================================================================

def carregar_dataset(log_f):
    log(f'\n{S}\n📂 ETAPA 1 — CARREGAMENTO E AUDITORIA INICIAL\n{S}', log_f)

    if not os.path.exists(ARQUIVO_ENTRADA):
        raise FileNotFoundError(
            f'Arquivo "{ARQUIVO_ENTRADA}" não encontrado. '
            f'Execute cvm_processamento.py primeiro para gerar a EDA.'
        )

    df = pd.read_csv(ARQUIVO_ENTRADA, sep=';', encoding='utf-8-sig',
                     low_memory=False)
    df['DT_FIM_EXERC'] = pd.to_datetime(df['DT_FIM_EXERC'], errors='coerce')

    log(f'  Dimensões brutas    : {df.shape[0]} linhas × {df.shape[1]} colunas', log_f)
    log(f'  Empresas únicas     : {df["CNPJ_CIA"].nunique()}', log_f)
    log(f'  Intervalo temporal  : '
        f'{df["DT_FIM_EXERC"].min().date()} → {df["DT_FIM_EXERC"].max().date()}', log_f)

    if 'TIPO_DOC' in df.columns:
        log(f'\n  Registros por TIPO_DOC:', log_f)
        for t, n in df['TIPO_DOC'].value_counts().items():
            log(f'    {t}: {n}', log_f)

    return df


# =============================================================================
# ETAPA 2 — FILTRO DFP
# =============================================================================

def filtrar_dfp(df, log_f):
    """
    Mantém apenas DFP (Demonstrações Financeiras Padronizadas — anuais e
    auditadas externamente). ITR é trimestral e não auditado, o que o torna
    inadequado para comparações diretas entre empresas e entre períodos.

    Decisão metodológica: modelos treinados em DFP+ITR misturados teriam
    targets e features em escalas incomparáveis (resultado anual vs. trimestral)
    e produziriam previsões inconsistentes. Se no futuro quisermos modelos
    trimestrais, o ITR deve ser tratado como um dataset separado.
    """
    log(f'\n{S}\n🗂️  ETAPA 2 — FILTRO: APENAS DFP\n{S}', log_f)

    antes = len(df)
    if 'TIPO_DOC' in df.columns:
        df = df[df['TIPO_DOC'] == 'DFP'].copy()
    log(f'  Registros antes : {antes}', log_f)
    log(f'  Registros depois: {len(df)} '
        f'(removidos {antes - len(df)} ITR)', log_f)
    log(f'  Empresas        : {df["CNPJ_CIA"].nunique()}', log_f)
    log(f'  Períodos únicos : {df["DT_FIM_EXERC"].nunique()}', log_f)
    log(f'  Períodos por empresa (mín/max): '
        f'{df.groupby("CNPJ_CIA")["DT_FIM_EXERC"].count().min()} / '
        f'{df.groupby("CNPJ_CIA")["DT_FIM_EXERC"].count().max()}', log_f)
    return df


# =============================================================================
# ETAPA 3 — REMOÇÃO DE COLUNAS COM EXCESSO DE NULOS
# =============================================================================

def remover_colunas_nulas(df, log_f):
    """
    Remove colunas com mais de LIMIAR_NULOS de valores nulos.

    Fundamento: uma coluna com >80% de nulos não tem informação suficiente para
    contribuir com a modelagem e qualquer imputação seria majoritariamente
    fabricada, introduzindo viés. O limiar de 80% é conservador — mantém colunas
    com cobertura razoável mesmo que imperfeita.

    Colunas de identificação (COLUNAS_ID) são preservadas independentemente.
    """
    log(f'\n{S}\n🗑️  ETAPA 3 — REMOÇÃO DE COLUNAS COM >{int(LIMIAR_NULOS*100)}% NULOS\n{S}',
        log_f)

    colunas_dados = [c for c in df.columns if c not in COLUNAS_ID]
    taxa_nulos    = df[colunas_dados].isnull().mean()
    remover       = taxa_nulos[taxa_nulos > LIMIAR_NULOS].index.tolist()
    manter        = taxa_nulos[taxa_nulos <= LIMIAR_NULOS].index.tolist()

    log(f'  Colunas de dados analisadas : {len(colunas_dados)}', log_f)
    log(f'  Colunas removidas (>{int(LIMIAR_NULOS*100)}%) : {len(remover)}', log_f)
    log(f'  Colunas mantidas            : {len(manter)}', log_f)

    if remover:
        log(f'\n  Colunas removidas:', log_f)
        for c in sorted(remover):
            log(f'    {c:<40} {fmt_pct(taxa_nulos[c])} nulos', log_f)

    df = df.drop(columns=remover)
    return df


# =============================================================================
# ETAPA 4 — IMPUTAÇÃO DE NULOS (MEDIANA POR SETOR)
# =============================================================================

def imputar_nulos(df, log_f):
    """
    Imputa valores ausentes pela mediana do setor para cada coluna numérica.

    Escolha da mediana (em vez da média):
    - A mediana é robusta a outliers — em dados financeiros, uma empresa com
      resultado atípico não distorce a imputação das demais do setor.
    - Imputa pela mediana do SETOR (em vez do dataset inteiro) porque cada setor
      tem perfis financeiros estruturalmente diferentes: imputar a mediana geral
      de Margem EBIT para uma empresa de Commodities com dado faltante seria
      metodologicamente incorreto.

    Se uma coluna não tiver nenhum valor não-nulo em um setor específico
    (setor inteiro sem aquela conta), o valor permanece NaN — não fabricamos
    dados onde não existem.

    Ordem de imputação: contas brutas prioritárias → KPIs → demais.
    KPIs calculados a partir de contas brutas são imputados após as contas
    para evitar inconsistências (ex: imputar Margem Líquida antes do Lucro).
    """
    log(f'\n{S}\n🩹 ETAPA 4 — IMPUTAÇÃO (MEDIANA POR SETOR)\n{S}', log_f)

    colunas_num = [c for c in df.select_dtypes(include=np.number).columns
                   if c not in COLUNAS_ID]

    # Ordena: brutas prioritárias → KPIs → demais
    ordem = (
        [c for c in CONTAS_BRUTAS_PRIORITARIAS if c in colunas_num] +
        [c for c in KPIS if c in colunas_num] +
        [c for c in colunas_num
         if c not in CONTAS_BRUTAS_PRIORITARIAS and c not in KPIS]
    )

    total_antes  = df[colunas_num].isnull().sum().sum()
    imputados    = {}

    for col in ordem:
        nulos_antes = df[col].isnull().sum()
        if nulos_antes == 0:
            continue
        medianas = df.groupby('SETOR')[col].transform('median')
        # Fallback: se o setor inteiro for NaN, usa mediana global
        mediana_global = df[col].median()
        df[col] = df[col].fillna(medianas).fillna(mediana_global)
        nulos_depois = df[col].isnull().sum()
        n_imp = nulos_antes - nulos_depois
        if n_imp > 0:
            imputados[col] = n_imp

    total_depois = df[colunas_num].isnull().sum().sum()
    log(f'  Nulos antes da imputação : {total_antes:,}', log_f)
    log(f'  Nulos depois             : {total_depois:,}', log_f)
    log(f'  Valores imputados        : {total_antes - total_depois:,}', log_f)
    log(f'\n  Top 15 colunas com mais imputações:', log_f)
    for col, n in sorted(imputados.items(), key=lambda x: -x[1])[:15]:
        log(f'    {col:<40} {n:>6} valores imputados', log_f)

    return df


# =============================================================================
# ETAPA 5 — WINSORIZAÇÃO (TRATAMENTO DE OUTLIERS)
# =============================================================================

def winsorizacao(df, log_f):
    """
    Winsoriza cada coluna numérica por setor usando o critério IQR × FATOR_WINS.

    Por que winsorizar em vez de remover?
    - Em séries temporais financeiras, um outlier extremo pode ser um evento
      real (crise, aquisição, reestruturação) que não deve ser apagado, apenas
      moderado. A winsorização preserva o registro mas limita seu impacto no
      treinamento do modelo.
    - IQR × 3 (em vez do tradicional × 1.5) é conservador — só afeta valores
      genuinamente extremos, não variações normais do setor.

    Por setor: Petrobras (receita na casa dos trilhões) não deve definir o teto
    de winsorização para empresas de Tecnologia como Totvs. Cada setor tem sua
    própria distribuição natural de magnitudes.

    Registra quantos valores foram winsorizados por coluna para auditoria.
    """
    log(f'\n{S}\n📐 ETAPA 5 — WINSORIZAÇÃO (IQR × {FATOR_WINS})\n{S}', log_f)

    colunas_num = [c for c in df.select_dtypes(include=np.number).columns
                   if c not in COLUNAS_ID]
    total_wins  = 0
    detalhes    = {}

    for col in colunas_num:
        n_wins_col = 0
        for setor, idx in df.groupby('SETOR').groups.items():
            serie = df.loc[idx, col]
            q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
            iqr    = q3 - q1
            if iqr == 0:
                continue
            lo = q1 - FATOR_WINS * iqr
            hi = q3 + FATOR_WINS * iqr
            mask_lo = df.loc[idx, col] < lo
            mask_hi = df.loc[idx, col] > hi
            df.loc[idx[mask_lo], col] = lo
            df.loc[idx[mask_hi], col] = hi
            n_wins_col += mask_lo.sum() + mask_hi.sum()
        if n_wins_col > 0:
            detalhes[col] = n_wins_col
            total_wins += n_wins_col

    log(f'  Total de valores winsorizados: {total_wins}', log_f)
    if detalhes:
        log(f'\n  Colunas com mais winsorizações:', log_f)
        for col, n in sorted(detalhes.items(), key=lambda x: -x[1])[:10]:
            log(f'    {col:<40} {n:>5} valores', log_f)
    return df


# =============================================================================
# ETAPA 6 — ENGENHARIA DE FEATURES: CRESCIMENTO YOY
# =============================================================================

def criar_features_crescimento(df, log_f):
    """
    Cria variáveis de crescimento ano a ano (YoY) para as principais contas e KPIs.

    Por que YoY em vez de valores absolutos apenas?
    - O modelo precisa capturar não só o estado financeiro atual mas a TENDÊNCIA:
      uma empresa com Margem EBIT de 10% mas que cresceu de 5% para 10% é muito
      diferente de outra que caiu de 20% para 10%.
    - Variáveis de crescimento são também naturalmente normalizadas por escala,
      facilitando a comparação entre Petrobras (R$ 500 bi de receita) e Totvs
      (R$ 4 bi de receita) no mesmo modelo.

    Implementação crítica:
    - O shift é feito DENTRO de cada grupo (CNPJ_CIA, TIPO_DOC) e ordenado por
      DT_FIM_EXERC, garantindo que o crescimento calculado seja sempre do período
      anterior DA MESMA EMPRESA e DO MESMO TIPO DE DOCUMENTO.
    - pct_change com fill_method=None evita que o pandas impute o denominador
      automaticamente, o que poderia mascarar períodos faltantes.
    - replace([inf, -inf], NaN): quando o valor anterior é zero, pct_change
      retorna infinito. Tratamos como ausente — não fabricamos crescimento.

    Colunas YoY geradas: sufixo _YOY adicionado ao nome da coluna original.
    """
    log(f'\n{S}\n📈 ETAPA 6 — FEATURES DE CRESCIMENTO YOY\n{S}', log_f)

    contas_yoy = [
        # Contas brutas
        'DRE_3.01',  # Receita Líquida
        'DRE_3.03',  # Lucro Bruto
        'DRE_3.05',  # EBIT
        'DRE_3.11',  # Lucro Líquido
        'BPA_1',     # Ativo Total
        'BPA_1.01',  # Ativo Circulante
        'BPP_2.03',  # Patrimônio Líquido
        'EBITDA',
        # KPIs
        'MARGEM_BRUTA_%',
        'MARGEM_EBIT_%',
        'MARGEM_LIQUIDA_%',
        'MARGEM_EBITDA_%',
        'ROE_%',
        'ROA_%',
        'LIQUIDEZ_CORRENTE',
        'ENDIVIDAMENTO_%',
        'GIRO_ATIVO',
    ]

    df = df.sort_values(['CNPJ_CIA', 'TIPO_DOC', 'DT_FIM_EXERC'])
    criadas = []

    for col in contas_yoy:
        if col not in df.columns:
            continue
        col_yoy = f'{col}_YOY'
        df[col_yoy] = (
            df.groupby(['CNPJ_CIA', 'TIPO_DOC'])[col]
              .pct_change(fill_method=None)
              .replace([np.inf, -np.inf], np.nan)
              * 100  # em percentual para interpretabilidade
        )
        criadas.append(col_yoy)

    # Aceleração da Receita: YoY do YoY — captura se o crescimento está
    # acelerando ou desacelerando (segunda derivada da receita)
    if 'DRE_3.01_YOY' in df.columns:
        df['DRE_3.01_ACELERACAO'] = (
            df.groupby(['CNPJ_CIA', 'TIPO_DOC'])['DRE_3.01_YOY']
              .diff()
              .replace([np.inf, -np.inf], np.nan)
        )
        criadas.append('DRE_3.01_ACELERACAO')

    log(f'  Features YoY criadas: {len(criadas)}', log_f)
    for c in criadas:
        cob = df[c].notna().mean()
        log(f'    {c:<45} cobertura: {fmt_pct(cob)}', log_f)

    return df, criadas


# =============================================================================
# ETAPA 7 — CODIFICAÇÃO DO SETOR (ONE-HOT)
# =============================================================================

def codificar_setor(df, log_f):
    """
    Converte a variável categórica SETOR em colunas binárias (one-hot encoding).

    O setor é uma das features mais importantes: o modelo precisa saber que
    uma Margem EBIT de 5% é normal para Varejo mas baixíssima para Energia.
    Sem essa codificação, o modelo trataria todas as empresas como se pertencessem
    ao mesmo perfil financeiro.

    Nomenclatura: SETOR_Petróleo, SETOR_Energia, etc. (prefixo SETOR_).
    A categoria base (drop_first=False) é mantida para facilitar interpretação
    do feature importance no Random Forest — árvores lidam bem com dummies
    completos sem sofrer de multicolinearidade perfeita.
    """
    log(f'\n{S}\n🏷️  ETAPA 7 — CODIFICAÇÃO DO SETOR (ONE-HOT)\n{S}', log_f)

    dummies    = pd.get_dummies(df['SETOR'], prefix='SETOR')
    df         = pd.concat([df, dummies], axis=1)
    cols_setor = dummies.columns.tolist()

    log(f'  Colunas criadas: {cols_setor}', log_f)
    return df, cols_setor


# =============================================================================
# ETAPA 8 — CRIAÇÃO DOS TARGETS (SHIFT -1)
# =============================================================================

def criar_targets(df, log_f):
    """
    Cria as variáveis-alvo (targets) como o valor de cada indicador no período
    seguinte (t+1), usando shift(-1) dentro do grupo (CNPJ_CIA, TIPO_DOC).

    Por que shift(-1) em vez de usar o valor atual?
    - O objetivo do modelo é PREVER o futuro, não descrever o presente.
      Usar o valor atual como target seria trivial (o modelo aprenderia que
      o melhor preditor da receita atual é a receita atual — tautologia).
    - shift(-1) garante que o target[t] = valor[t+1]: dado o estado financeiro
      da empresa no período t, o modelo deve prever o que acontecerá em t+1.

    Critérios de qualidade do target:
    - O último período de cada empresa/TIPO_DOC sempre terá target = NaN
      (não há t+1 para o último registro). Essas linhas serão removidas do
      conjunto de treino/teste mas ficam no dataset para eventual uso como
      "período de previsão futuro".
    - A verificação de variação temporal garante que o target criado pertença
      ao período imediatamente seguinte (sem lacunas de anos).
    """
    log(f'\n{S}\n🎯 ETAPA 8 — CRIAÇÃO DOS TARGETS (SHIFT -1)\n{S}', log_f)

    df = df.sort_values(['CNPJ_CIA', 'TIPO_DOC', 'DT_FIM_EXERC'])
    cols_target = {}

    for col_orig, nome_legivel in TARGETS.items():
        if col_orig not in df.columns:
            log(f'  ⚠️  {col_orig} ausente no dataset — target {nome_legivel} não criado',
                log_f)
            continue
        col_target = f'TARGET_{col_orig}'

        # Shift dentro do grupo — nunca mistura empresas nem DFP com ITR
        df[col_target] = (
            df.groupby(['CNPJ_CIA', 'TIPO_DOC'])[col_orig]
              .shift(-1)
        )

        # Verifica se há saltos temporais (ex: de 2019 para 2021) que tornariam
        # o target incorreto. Nesses casos, o target é anulado.
        prox_ano = (
            df.groupby(['CNPJ_CIA', 'TIPO_DOC'])['ANO_REF']
              .shift(-1)
        )
        ano_esperado = df['ANO_REF'] + 1
        mask_salto   = (prox_ano != ano_esperado) & prox_ano.notna()
        df.loc[mask_salto, col_target] = np.nan

        n_validos = df[col_target].notna().sum()
        n_nans    = df[col_target].isna().sum()
        cols_target[col_orig] = col_target

        log(f'\n  Target: {col_target}', log_f)
        log(f'    Nome legível: {nome_legivel}', log_f)
        log(f'    Válidos     : {n_validos}', log_f)
        log(f'    NaN (último período ou salto): {n_nans}', log_f)

    return df, cols_target


# =============================================================================
# ETAPA 9 — SELEÇÃO DE FEATURES
# =============================================================================

def selecionar_features(df, cols_target, cols_yoy, cols_setor, log_f):
    """
    Identifica as features mais relevantes para cada target por dois métodos
    complementares:

    Método A — Correlação de Pearson:
      Mede a força da relação LINEAR entre cada feature e o target.
      Vantagem: interpretável e rápido.
      Limitação: não captura relações não lineares.

    Método B — Recursive Feature Elimination (RFE) com Ridge Regression:
      Treina um modelo Ridge repetidamente, removendo a feature menos relevante
      a cada iteração até atingir N_FEATURES_RFE features.
      A Ridge é usada como estimador base (em vez do Random Forest) porque:
        (1) é rápida para RFE iterativo,
        (2) produz coeficientes estáveis mesmo com multicolinearidade.
      O ranking final do RFE é agnóstico ao algoritmo final — serve como
      pré-seleção antes do treinamento dos modelos finais.

    A lista de features recomendadas combina as top-N do Pearson com as
    selecionadas pelo RFE, removendo duplicatas. Essa união garante que tanto
    relações lineares quanto não lineares estejam representadas.

    Colunas excluídas da seleção:
    - Colunas de identificação (COLUNAS_ID)
    - Os próprios targets (data leakage direto)
    - Colunas com variância zero (constantes não contribuem)
    - Colunas completamente ausentes
    """
    log(f'\n{S}\n🔍 ETAPA 9 — SELEÇÃO DE FEATURES\n{S}', log_f)

    # Pool de features candidatas
    colunas_excluir = set(COLUNAS_ID) | set(cols_target.values())
    features_pool   = [c for c in df.columns
                       if c not in colunas_excluir
                       and df[c].dtype in [np.float64, np.float32,
                                           np.int64, np.int32]
                       and df[c].nunique() > 1]

    resultados_selecao = {}

    for col_orig, col_target in cols_target.items():
        if col_target not in df.columns:
            continue
        nome = TARGETS.get(col_orig, col_orig)
        log(f'\n  {s2}', log_f)
        log(f'  Target: {nome}', log_f)
        log(f'  {s2}', log_f)

        # Linhas válidas para este target
        df_val = (df[features_pool + [col_target]]
                  .replace([np.inf, -np.inf], np.nan)
                  .dropna(subset=[col_target]))

        # Remove features com >50% de nulos no subconjunto válido
        taxa_nulos_feat = df_val[features_pool].isnull().mean()
        feats_ok = taxa_nulos_feat[taxa_nulos_feat <= 0.50].index.tolist()
        df_val   = df_val[feats_ok + [col_target]].dropna()

        if df_val.shape[0] < 20 or len(feats_ok) < 3:
            log(f'  ⚠️  Dados insuficientes para seleção ({df_val.shape[0]} linhas)', log_f)
            continue

        X = df_val[feats_ok].values
        y = df_val[col_target].values

        # ── Método A: Correlação de Pearson ──────────────────────────
        corrs = (pd.DataFrame({'feature': feats_ok,
                                'pearson': [abs(np.corrcoef(X[:, i], y)[0, 1])
                                            for i in range(X.shape[1])]})
                   .sort_values('pearson', ascending=False)
                   .dropna())

        top_pearson = corrs.head(N_FEATURES_RFE)['feature'].tolist()
        log(f'\n  [A] Top-{N_FEATURES_RFE} por Correlação de Pearson:', log_f)
        for _, row in corrs.head(N_FEATURES_RFE).iterrows():
            barra = '█' * int(row['pearson'] * 20)
            log(f'    {row["feature"]:<42} r={row["pearson"]:.3f}  {barra}', log_f)

        # ── Método B: RFE com Ridge ───────────────────────────────────
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X)
        n_sel   = min(N_FEATURES_RFE, len(feats_ok))
        rfe     = RFE(Ridge(alpha=1.0), n_features_to_select=n_sel)

        try:
            rfe.fit(X_sc, y)
            top_rfe = [feats_ok[i] for i, s in enumerate(rfe.support_) if s]
            log(f'\n  [B] Features selecionadas por RFE ({n_sel}):', log_f)
            for f in sorted(top_rfe):
                log(f'    {f}', log_f)
        except Exception as e:
            top_rfe = top_pearson
            log(f'\n  ⚠️  RFE falhou ({e}), usando Pearson como fallback', log_f)

        # ── União Pearson + RFE ───────────────────────────────────────
        features_finais = list(dict.fromkeys(top_pearson + top_rfe))
        log(f'\n  Features recomendadas (união A+B): {len(features_finais)}', log_f)
        for f in features_finais:
            log(f'    {f}', log_f)

        resultados_selecao[col_orig] = {
            'target_col':     col_target,
            'target_nome':    nome,
            'features':       features_finais,
            'pearson_ranking': corrs[['feature', 'pearson']].to_dict('records'),
            'n_obs':          len(df_val),
        }

    # Salva ranking consolidado em CSV para consulta futura
    rows_csv = []
    for col_orig, info in resultados_selecao.items():
        for rank, rec in enumerate(info['pearson_ranking'], 1):
            rows_csv.append({
                'target':   info['target_nome'],
                'rank':     rank,
                'feature':  rec['feature'],
                'pearson_r': round(rec['pearson'], 4),
                'selecionada_rfe': rec['feature'] in info['features'],
            })
    pd.DataFrame(rows_csv).to_csv(
        'cvm_features_selecionadas.csv', index=False, sep=';', encoding='utf-8-sig'
    )
    log(f'\n  💾 Ranking salvo em: cvm_features_selecionadas.csv', log_f)

    return resultados_selecao


# =============================================================================
# ETAPA 10 — SPLIT TEMPORAL TREINO / TESTE
# =============================================================================

def split_temporal(df, cols_target, log_f):
    """
    Divide o dataset em treino e teste respeitando a ordem temporal por empresa.

    Por que split temporal e não aleatório?
    - Em séries temporais financeiras, o split aleatório causaria data leakage:
      o modelo poderia aprender com dados do futuro para prever o passado, o
      que inflacionaria artificialmente as métricas e não refletiria a capacidade
      real de generalização.
    - O split temporal correto usa os PRIMEIROS ~75% dos períodos de cada empresa
      para treino e os ÚLTIMOS ~25% para teste, simulando o cenário real de uso:
      treinar com histórico e prever os períodos mais recentes.

    Implementação:
    - O corte é feito por empresa individualmente para garantir que todas as
      empresas tenham representação no treino, mesmo as com menos histórico.
    - Apenas linhas com target válido (não NaN) são incluídas nos splits.
    - A coluna "SPLIT" é adicionada ao dataset final para rastreabilidade.
    """
    log(f'\n{S}\n✂️  ETAPA 10 — SPLIT TEMPORAL TREINO / TESTE\n{S}', log_f)

    # Remove linhas sem nenhum target válido (último período de cada empresa)
    col_targets_lista = list(cols_target.values())
    targets_existentes = [c for c in col_targets_lista if c in df.columns]
    df_model = df.dropna(subset=targets_existentes, how='all').copy()

    df_model = df_model.sort_values(['CNPJ_CIA', 'DT_FIM_EXERC'])
    df_model['SPLIT'] = 'treino'

    for cnpj, grupo in df_model.groupby('CNPJ_CIA'):
        idx_sorted = grupo.index.tolist()
        n          = len(idx_sorted)
        n_teste    = max(1, int(np.ceil(n * PROPORCAO_TESTE)))
        idx_teste  = idx_sorted[-n_teste:]
        df_model.loc[idx_teste, 'SPLIT'] = 'teste'

    treino = df_model[df_model['SPLIT'] == 'treino']
    teste  = df_model[df_model['SPLIT'] == 'teste']

    log(f'  Total de linhas com target válido : {len(df_model)}', log_f)
    log(f'  Treino : {len(treino)} linhas '
        f'({fmt_pct(len(treino)/len(df_model))})', log_f)
    log(f'  Teste  : {len(teste)} linhas '
        f'({fmt_pct(len(teste)/len(df_model))})', log_f)

    # Mostra distribuição por empresa
    log(f'\n  Períodos por empresa (treino/teste):', log_f)
    for cnpj, g in df_model.groupby('CNPJ_CIA'):
        nome   = g['DENOM_CIA'].iloc[0][:30]
        n_tr   = (g['SPLIT'] == 'treino').sum()
        n_te   = (g['SPLIT'] == 'teste').sum()
        datas  = f"{g['DT_FIM_EXERC'].min().year}–{g['DT_FIM_EXERC'].max().year}"
        log(f'    {nome:<32} treino={n_tr}  teste={n_te}  [{datas}]', log_f)

    return df_model, treino, teste


# =============================================================================
# ETAPA 11 — SALVAMENTO
# =============================================================================

def salvar_artefatos(df_full, df_model, treino, teste, log_f):
    log(f'\n{S}\n💾 ETAPA 11 — SALVAMENTO DOS ARTEFATOS\n{S}', log_f)

    enc = 'utf-8-sig'
    sep = ';'

    df_full.to_csv('cvm_dataset_final.csv',  index=False, sep=sep, encoding=enc)
    df_model.to_csv('cvm_dataset_modelo.csv', index=False, sep=sep, encoding=enc)
    treino.to_csv('cvm_dataset_treino.csv',   index=False, sep=sep, encoding=enc)
    teste.to_csv('cvm_dataset_teste.csv',     index=False, sep=sep, encoding=enc)

    log(f'  cvm_dataset_final.csv   — dataset completo preparado '
        f'({df_full.shape[0]}×{df_full.shape[1]})', log_f)
    log(f'  cvm_dataset_modelo.csv  — linhas com target válido '
        f'({df_model.shape[0]} linhas)', log_f)
    log(f'  cvm_dataset_treino.csv  — split de treino ({len(treino)} linhas)', log_f)
    log(f'  cvm_dataset_teste.csv   — split de teste  ({len(teste)} linhas)', log_f)
    log(f'  cvm_features_selecionadas.csv — ranking de features por target', log_f)

    # Sumário final
    log(f'\n{S}', log_f)
    log(f'✅ PREPARAÇÃO CONCLUÍDA', log_f)
    log(f'   Dataset final: {df_full.shape[0]} registros × {df_full.shape[1]} colunas',
        log_f)

    targets_disp = [c for c in df_full.columns if c.startswith('TARGET_')]
    log(f'   Targets disponíveis ({len(targets_disp)}):', log_f)
    for t in targets_disp:
        cob = df_full[t].notna().mean()
        log(f'     {t:<35} cobertura: {fmt_pct(cob)}', log_f)

    feats_yoy = [c for c in df_full.columns if '_YOY' in c or '_ACELERACAO' in c]
    feats_setor = [c for c in df_full.columns if c.startswith('SETOR_')]
    log(f'   Features YoY criadas : {len(feats_yoy)}', log_f)
    log(f'   Features de setor    : {len(feats_setor)}', log_f)
    log(S, log_f)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def preparar_dados():
    with open('cvm_prep_relatorio.txt', 'w', encoding='utf-8') as log_f:
        log_f.write('RELATÓRIO DE PREPARAÇÃO DOS DADOS — CVM\n')
        log_f.write('TCC: Predição de Indicadores Financeiros com ML\n')
        log_f.write(f'{S}\n\n')

        # Etapas sequenciais
        df = carregar_dataset(log_f)
        df = filtrar_dfp(df, log_f)
        df = remover_colunas_nulas(df, log_f)
        df = imputar_nulos(df, log_f)
        df = winsorizacao(df, log_f)
        df, cols_yoy = criar_features_crescimento(df, log_f)
        df, cols_setor = codificar_setor(df, log_f)
        df, cols_target = criar_targets(df, log_f)

        # Seleção de features (requer targets criados)
        sel = selecionar_features(df, cols_target, cols_yoy, cols_setor, log_f)

        # Split temporal
        df_model, treino, teste = split_temporal(df, cols_target, log_f)

        # Salva tudo
        salvar_artefatos(df, df_model, treino, teste, log_f)

    print('\n📄 Relatório completo salvo em: cvm_prep_relatorio.txt')
    return df, treino, teste, sel


if __name__ == '__main__':
    df, treino, teste, features_por_target = preparar_dados()