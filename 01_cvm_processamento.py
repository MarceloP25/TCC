# =============================================================================
# PROCESSAMENTO DE DADOS CVM — DFP / ITR
# TCC: Predição de Indicadores Financeiros com Machine Learning
# =============================================================================
# Etapa atual (CRISP-DM): Compreensão dos Dados
# Objetivo  : Ingestão dos ZIPs da CVM + EDA completa para identificar
#             contas disponíveis, cobertura temporal, qualidade dos dados
#             e KPIs financeiros relevantes para a etapa de modelagem.
# =============================================================================
# Dependências: pandas, numpy, matplotlib, seaborn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import zipfile
import os
import re
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

PASTAS        = ['./TCC_dados/DFP', './TCC_dados/ITR']
ARQUIVO_SAIDA = 'cvm_dados_processados.csv'

TIPOS_DEMONSTRATIVOS = [
    'BPA_con',    # Balanço Patrimonial Ativo
    'BPP_con',    # Balanço Patrimonial Passivo
    'DRE_con',    # Demonstração do Resultado
    'DFC_MD_con', # Fluxo de Caixa — Método Direto
    'DFC_MI_con', # Fluxo de Caixa — Método Indireto (Energia, Commodities)
    'DVA_con',    # Demonstração do Valor Adicionado
    'DMPL_con',   # Mutações do Patrimônio Líquido
    'DRA_con',    # Resultado Abrangente
]

EMPRESAS_POR_SETOR = {
    'Petróleo': [
        '33.000.167/0001-01',  # Petrobras
        '07.354.482/0001-42',  # Prio
        '33.256.439/0001-08',  # Ultrapar
        '33.453.598/0001-23',  # Raízen
        '34.274.233/0001-02',  # Vibra Energia
    ],
    'Energia': [
        '02.474.103/0001-40',  # Engie Brasil
        '03.220.438/0001-73',  # Equatorial Energia
        '07.859.971/0001-14',  # Taesa
        '02.429.144/0001-93',  # CPFL Energia
        '02.998.611/0001-04',  # ISA CTEEP
    ],
    'Varejo': [
        '92.754.738/0001-71',  # Lojas Renner
        '47.960.950/0001-21',  # Magazine Luiza
        '61.079.117/0001-05',  # Alpargatas
        '16.590.234/0001-48',  # Arezzo
        '24.990.777/0001-09',  # Grupo Mateus
    ],
    'Commodities': [
        '33.592.510/0001-54',  # Vale
        '33.359.392/0001-75',  # Gerdau
        '16.404.287/0001-55',  # Suzano
        '89.637.490/0001-45',  # Klabin
        '51.503.388/0001-43',  # São Martinho
    ],
    'Tecnologia': [
        '84.429.695/0001-11',  # WEG
        '53.113.791/0001-22',  # Totvs
        '02.351.877/0001-52',  # Locaweb
        '07.689.002/0001-89',  # Embraer
        '16.670.085/0001-55',  # Localiza
    ],
}

CNPJS_FILTRO    = [c for setor in EMPRESAS_POR_SETOR.values() for c in setor]
CNPJ_PARA_SETOR = {c: s for s, cs in EMPRESAS_POR_SETOR.items() for c in cs}

# -----------------------------------------------------------------------------
# Mapa de contas CVM → nomes legíveis
# Usado na EDA para rotular colunas com nomes de relatórios financeiros reais
# Fonte: estrutura padrão de DREs e BPs consolidados da CVM (COSIF/ITG 1000)
# -----------------------------------------------------------------------------
MAPA_CONTAS = {
    # DRE
    'DRE_3.01': 'Receita Líquida',
    'DRE_3.02': 'Custo dos Produtos/Serviços',
    'DRE_3.03': 'Lucro Bruto',
    'DRE_3.04': 'Despesas Operacionais',
    'DRE_3.05': 'EBIT',
    'DRE_3.06': 'Resultado Financeiro Líquido',
    'DRE_3.07': 'Resultado antes do IR/CSLL',
    'DRE_3.08': 'IR e CSLL',
    'DRE_3.09': 'Resultado de Operações Descont.',
    'DRE_3.10': 'Lucro/Prejuízo Consolidado',
    'DRE_3.11': 'Lucro Líquido do Período',
    # BPA
    'BPA_1':       'Ativo Total',
    'BPA_1.01':    'Ativo Circulante',
    'BPA_1.01.01': 'Caixa e Equiv. de Caixa',
    'BPA_1.01.02': 'Aplicações Financeiras CP',
    'BPA_1.01.03': 'Contas a Receber',
    'BPA_1.01.04': 'Estoques',
    'BPA_1.02':    'Ativo Não Circulante',
    'BPA_1.02.01': 'Aplicações Financeiras LP',
    'BPA_1.02.03': 'Imobilizado',
    'BPA_1.02.04': 'Intangível',
    # BPP
    'BPP_2':       'Passivo Total + PL',
    'BPP_2.01':    'Passivo Circulante',
    'BPP_2.01.04': 'Empréstimos CP',
    'BPP_2.02':    'Passivo Não Circulante',
    'BPP_2.02.01': 'Empréstimos LP',
    'BPP_2.03':    'Patrimônio Líquido',
    'BPP_2.03.01': 'Capital Social',
    'BPP_2.03.05': 'Lucros/Prejuízos Acumulados',
    # DFC
    'DFC_MD_6.01': 'FC Operacional (MD)',
    'DFC_MD_6.02': 'FC de Investimento (MD)',
    'DFC_MD_6.03': 'FC de Financiamento (MD)',
    'DFC_MI_6.01': 'FC Operacional (MI)',
    'DFC_MI_6.02': 'FC de Investimento (MI)',
    'DFC_MI_6.03': 'FC de Financiamento (MI)',
    # DVA
    'DVA_7.08':    'Valor Adicionado Total',
    # D&A e EBITDA — calculados, não são contas CVM diretas
    'DA_TOTAL':      'Deprec. & Amortização (DFC_MI)',
    'EBITDA':        'EBITDA',
    'MARGEM_EBITDA_%': 'Margem EBITDA (%)',
    'MARGEM_BRUTA_%':  'Margem Bruta (%)',
    'MARGEM_EBIT_%':   'Margem EBIT (%)',
    'MARGEM_LIQUIDA_%':'Margem Líquida (%)',
    'ROE_%':           'ROE (%)',
    'ROA_%':           'ROA (%)',
    'LIQUIDEZ_CORRENTE':'Liquidez Corrente',
    'LIQUIDEZ_IMEDIATA':'Liquidez Imediata',
    'ENDIVIDAMENTO_%': 'Endividamento (%)',
    'ALAVANCAGEM_DE':  'Alavancagem D/E',
    'DIVIDA_LIQUIDA':  'Dívida Líquida',
    'COBERTURA_JUROS': 'Cobertura de Juros',
    'GIRO_ATIVO':      'Giro do Ativo',
    'FCO_SOBRE_RECEITA_%': 'FCO/Receita (%)',
    'FCO_SOBRE_LL_%':  'FCO/Lucro Líq. (%)',
}

# =============================================================================
# 2. FUNÇÕES DE INGESTÃO
# =============================================================================

def normalizar_cnpj(cnpj):
    """
    Normaliza CNPJ para XX.XXX.XXX/XXXX-XX.
    Arquivos CVM mais antigos armazenam sem formatação.
    """
    digits = re.sub(r'\D', '', str(cnpj))
    if len(digits) == 14:
        return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:]}"
    return cnpj


def normalizar_escala(df):
    """
    Converte VL_CONTA para reais conforme ESCALA_MOEDA.
      'UNIDADE' → sem alteração
      'MIL'     → × 1.000
      'BILHÃO'  → × 1.000.000.000 (arquivos legados)
    """
    if 'ESCALA_MOEDA' not in df.columns:
        return df
    df  = df.copy()
    esc = df['ESCALA_MOEDA'].str.upper().str.strip()
    df.loc[esc == 'MIL',    'VL_CONTA'] *= 1_000
    df.loc[esc == 'BILHÃO', 'VL_CONTA'] *= 1_000_000_000
    return df


def limpar_df_base(df, cnpjs_filtro):
    """Filtra exercício atual, normaliza CNPJ e escala monetária."""
    df = df[df['ORDEM_EXERC'] == 'ÚLTIMO'].copy()
    if cnpjs_filtro and 'CNPJ_CIA' in df.columns:
        df['CNPJ_CIA'] = df['CNPJ_CIA'].apply(normalizar_cnpj)
        df = df[df['CNPJ_CIA'].isin(cnpjs_filtro)]
    return normalizar_escala(df)


def pivotar_demonstrativo(df, prefixo):
    """Formato longo → largo usando CD_CONTA. DENOM_CIA re-adicionada após pivot."""
    df = df.copy()
    df['COLUNA'] = prefixo + '_' + df['CD_CONTA'].astype(str).str.strip()
    pivoted = df.pivot_table(
        index=['CNPJ_CIA', 'DT_FIM_EXERC'],
        columns='COLUNA', values='VL_CONTA', aggfunc='first'
    ).reset_index()
    pivoted.columns.name = None
    nomes = df.groupby('CNPJ_CIA')['DENOM_CIA'].last().reset_index()
    return pivoted.merge(nomes, on='CNPJ_CIA', how='left')


def pivotar_dmpl(df, prefixo='DMPL'):
    """DMPL: combina CD_CONTA + COLUNA_DF para evitar colisões de nomes."""
    df = df.copy()
    if 'COLUNA_DF' in df.columns:
        df['COLUNA'] = (prefixo + '_' + df['CD_CONTA'].astype(str).str.strip()
                        + '_' + df['COLUNA_DF'].astype(str).str.strip())
    else:
        df['COLUNA'] = prefixo + '_' + df['CD_CONTA'].astype(str).str.strip()
    pivoted = df.pivot_table(
        index=['CNPJ_CIA', 'DT_FIM_EXERC'],
        columns='COLUNA', values='VL_CONTA', aggfunc='first'
    ).reset_index()
    pivoted.columns.name = None
    nomes = df.groupby('CNPJ_CIA')['DENOM_CIA'].last().reset_index()
    return pivoted.merge(nomes, on='CNPJ_CIA', how='left')


def extrair_prefixo(tipo):
    return tipo.replace('_con', '').replace('_ind', '').upper()


# Palavras-chave que identificam linhas de Depreciação & Amortização no DFC_MI.
# A CVM não padroniza o nome nem o número da subconta — cada empresa nomeia
# livremente dentro de 6.01.01.xx. Por isso usamos busca por texto.
_KEYWORDS_DA = re.compile(r'deprecia|amortiz', re.IGNORECASE)


def extrair_da_dfc_mi(df_mi_longo):
    """
    Extrai Depreciação & Amortização (D&A) a partir do DFC_MI em formato longo,
    ANTES do pivot para formato largo.

    Por que antes do pivot?
    Após o pivot, a coluna DS_CONTA (descrição da conta) desaparece — ela é usada
    apenas para montar o nome da coluna pivotada. Sem ela, não é possível identificar
    quais subcontas contêm D&A, pois o número da subconta (ex: 6.01.01.02 vs
    6.01.01.06) varia de empresa para empresa e não é padronizado pela CVM.

    Estratégia (validada com dados reais do ZIP dfp_cia_aberta_2025.zip):
      - Filtra subcontas dentro de 6.01.01.xx (ajustes ao resultado no MI)
      - Seleciona linhas cujo DS_CONTA contenha 'deprecia' ou 'amortiz'
      - Soma todas as linhas por empresa/período (pode haver múltiplas linhas,
        ex: 'Depreciação e amortização' + 'Amortização de ativo de direito de uso')

    Retorna DataFrame com colunas: CNPJ_CIA, DT_FIM_EXERC, DA_TOTAL
    """
    if df_mi_longo is None or df_mi_longo.empty:
        return pd.DataFrame(columns=['CNPJ_CIA', 'DT_FIM_EXERC', 'DA_TOTAL'])

    mask = (
        df_mi_longo['CD_CONTA'].str.startswith('6.01.01') &
        df_mi_longo['DS_CONTA'].str.contains(_KEYWORDS_DA, na=False)
    )
    da_linhas = df_mi_longo[mask]

    if da_linhas.empty:
        return pd.DataFrame(columns=['CNPJ_CIA', 'DT_FIM_EXERC', 'DA_TOTAL'])

    da_agg = (da_linhas
              .groupby(['CNPJ_CIA', 'DT_FIM_EXERC'])['VL_CONTA']
              .sum()
              .reset_index()
              .rename(columns={'VL_CONTA': 'DA_TOTAL'}))
    return da_agg


def processar_zip(caminho_zip, ano, tipos, cnpjs_filtro):
    """Lê um ZIP da CVM e retorna DataFrame consolidado por empresa/período."""
    dfs_do_ano  = []
    df_mi_longo = None   # guardamos o DFC_MI em formato longo para extrair D&A
    nome_zip    = os.path.basename(caminho_zip).upper()
    tipo_doc    = 'DFP' if 'DFP' in nome_zip else 'ITR'

    with zipfile.ZipFile(caminho_zip, 'r') as z:
        arquivos_no_zip = z.namelist()
        for tipo in tipos:
            candidatos     = [f'dfp_cia_aberta_{tipo}_{ano}.csv',
                              f'itr_cia_aberta_{tipo}_{ano}.csv']
            csv_encontrado = next((c for c in candidatos if c in arquivos_no_zip), None)
            if not csv_encontrado:
                print(f"   ⚠️  {tipo} ({ano}) não encontrado — pulando.")
                continue
            try:
                with z.open(csv_encontrado) as f:
                    df = pd.read_csv(f, sep=';', encoding='latin-1', low_memory=False)
                df = limpar_df_base(df, cnpjs_filtro)
                if df.empty:
                    print(f"   ⚠️  {tipo} ({ano}): sem dados após filtros.")
                    continue
                prefixo = extrair_prefixo(tipo)

                # Captura o DFC_MI em formato longo ANTES do pivot para extração de D&A.
                # O DS_CONTA (descrição) é perdido após o pivot — sem ele não é possível
                # identificar as linhas de depreciação por keyword.
                if 'DFC_MI' in tipo.upper():
                    df_mi_longo = df.copy()

                pivoted = pivotar_dmpl(df, prefixo) if 'DMPL' in tipo \
                          else pivotar_demonstrativo(df, prefixo)
                dfs_do_ano.append(pivoted)
                print(f"   ✅ {tipo}: {len(pivoted)} registros, "
                      f"{len(pivoted.columns)} colunas")
            except Exception as e:
                print(f"   ❌ Erro em {tipo} ({ano}): {e}")

    if not dfs_do_ano:
        return None

    df_cons = dfs_do_ano[0]
    for df_extra in dfs_do_ano[1:]:
        df_cons = pd.merge(df_cons, df_extra, on=['CNPJ_CIA', 'DT_FIM_EXERC'],
                           how='outer', suffixes=('', '_dup'))
        df_cons = df_cons.loc[:, ~df_cons.columns.str.endswith('_dup')]

    # Extrai D&A do DFC_MI e incorpora ao dataset consolidado.
    # Empresas que usam DFC_MD (Método Direto) não têm seção de ajustes →
    # DA_TOTAL ficará NaN para essas empresas, documentado como limitação.
    da = extrair_da_dfc_mi(df_mi_longo)
    if not da.empty:
        df_cons = df_cons.merge(da, on=['CNPJ_CIA', 'DT_FIM_EXERC'], how='left')
        n_da = da['CNPJ_CIA'].nunique()
        print(f"   📐 D&A (DFC_MI): {n_da} empresa(s) com D&A identificado")
    else:
        df_cons['DA_TOTAL'] = np.nan
        print(f"   ⚠️  D&A: nenhuma linha encontrada no DFC_MI deste arquivo")

    df_cons['ANO_REF']  = int(ano)
    df_cons['TIPO_DOC'] = tipo_doc
    df_cons['SETOR']    = df_cons['CNPJ_CIA'].map(CNPJ_PARA_SETOR)
    return df_cons


# =============================================================================
# 3. PROCESSAMENTO PRINCIPAL
# =============================================================================

def processar_tudo():
    dados_acumulados = []
    for pasta in PASTAS:
        if not os.path.exists(pasta):
            print(f"📁 Pasta '{pasta}' não encontrada — pulando.")
            continue
        for nome_zip in sorted(f for f in os.listdir(pasta) if f.endswith('.zip')):
            match = re.search(r'\b(20\d{2})\b', nome_zip)
            if not match:
                print(f"⚠️  Ano não identificado em: {nome_zip}")
                continue
            ano = match.group(1)
            print(f"\n🔄 Processando: {nome_zip} (ano {ano})")
            res = processar_zip(os.path.join(pasta, nome_zip), ano,
                                TIPOS_DEMONSTRATIVOS, CNPJS_FILTRO)
            if res is not None:
                dados_acumulados.append(res)
                print(f"   📊 {len(res)} registros consolidados")

    if not dados_acumulados:
        print("\n❌ Nenhum dado processado. Verifique pastas e ZIPs.")
        return None

    df = pd.concat(dados_acumulados, ignore_index=True)
    df = df.sort_values(['CNPJ_CIA', 'DT_FIM_EXERC']).reset_index(drop=True)
    df['DT_FIM_EXERC'] = pd.to_datetime(df['DT_FIM_EXERC'], errors='coerce')

    df.to_csv(ARQUIVO_SAIDA, index=False, sep=';', encoding='utf-8-sig')

    sem_setor = df[df['SETOR'].isna()]['CNPJ_CIA'].unique()
    if len(sem_setor):
        print(f"\n  ⚠️  {len(sem_setor)} CNPJ(s) sem setor — verifique formatação:")
        for c in sem_setor:
            print(f"     {c}")

    arquivos_setor = []
    for setor, df_s in df.dropna(subset=['SETOR']).groupby('SETOR'):
        nome = (setor.lower()
                .replace(' ', '_').replace('é', 'e')
                .replace('ó', 'o').replace('ô', 'o'))
        arq  = f'cvm_dados_processados_{nome}.csv'
        df_s.to_csv(arq, index=False, sep=';', encoding='utf-8-sig')
        arquivos_setor.append((setor, arq, len(df_s)))

    sep = '=' * 60
    print(f"\n{sep}\n✅ Processamento concluído!")
    print(f"   Empresas: {df['CNPJ_CIA'].nunique()} | "
          f"Linhas: {len(df)} | Colunas: {len(df.columns)}")
    print(f"   Arquivo geral: {ARQUIVO_SAIDA}")
    for setor, arq, n in arquivos_setor:
        print(f"   [{setor}]  {arq}  ({n} registros)")
    print(sep)
    return df


# =============================================================================
# 4. EDA — ANÁLISE EXPLORATÓRIA COMPLETA
# =============================================================================

S  = '=' * 70
s2 = '─' * 70


def _fmt_reais(v):
    """Formata número como R$ bilhões/milhões para leitura rápida."""
    if pd.isna(v):
        return 'N/A'
    b = v / 1e9
    return f"R$ {b:,.1f} bi" if abs(b) >= 1 else f"R$ {v/1e6:,.1f} mi"


def eda_visao_geral(df):
    """BLOCO 1 — Estrutura do dataset: shape, empresas, períodos, tipo de doc."""
    print(f"\n{S}\n📊 BLOCO 1 — VISÃO GERAL DO DATASET\n{S}")
    print(f"  Dimensões          : {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"  Empresas únicas    : {df['CNPJ_CIA'].nunique()}")
    print(f"  Períodos únicos    : {df['DT_FIM_EXERC'].nunique()}")
    print(f"  Intervalo temporal : "
          f"{df['DT_FIM_EXERC'].min().date()} → "
          f"{df['DT_FIM_EXERC'].max().date()}")

    if 'TIPO_DOC' in df.columns:
        print(f"\n  Registros por tipo de documento:")
        print(df['TIPO_DOC'].value_counts().to_string())

    print(f"\n{s2}\n  Empresas por setor:\n{s2}")
    resumo = (df[['CNPJ_CIA', 'DENOM_CIA', 'SETOR']]
              .drop_duplicates()
              .sort_values(['SETOR', 'DENOM_CIA']))
    for setor, grupo in resumo.groupby('SETOR'):
        print(f"\n  [{setor}]")
        for _, row in grupo.iterrows():
            print(f"    {row['CNPJ_CIA']}  |  {row['DENOM_CIA']}")

    print(f"\n  Registros por setor:")
    print(df.groupby('SETOR').size().rename('Registros').to_string())


def eda_cobertura_temporal(df):
    """
    BLOCO 2 — Cobertura temporal por empresa.
    Quantos períodos cada empresa possui — essencial para avaliar a
    viabilidade de séries temporais na modelagem preditiva.
    Distingue DFP (anual, auditado) de ITR (trimestral) pois para a
    modelagem de séries temporais apenas o DFP é comparável entre períodos.
    """
    print(f"\n{S}\n📅 BLOCO 2 — COBERTURA TEMPORAL POR EMPRESA\n{S}")

    # Mostra breakdown por TIPO_DOC se ambos estiverem presentes
    if 'TIPO_DOC' in df.columns and df['TIPO_DOC'].nunique() > 1:
        print(f"\n  ⚠️  Dataset contém DFP e ITR simultaneamente.")
        print(f"  Para modelagem de séries temporais use apenas DFP (anual/auditado).")
        print(f"\n  Registros por tipo:")
        for tipo, n in df['TIPO_DOC'].value_counts().items():
            print(f"    {tipo}: {n}")

    # Cobertura por empresa — separa DFP de ITR
    for tipo_doc in (['DFP', 'ITR'] if 'TIPO_DOC' in df.columns else [None]):
        df_t = df[df['TIPO_DOC'] == tipo_doc] if tipo_doc else df
        if df_t.empty:
            continue
        label = f" [{tipo_doc}]" if tipo_doc else ""

        cob = (df_t.groupby(['SETOR', 'CNPJ_CIA', 'DENOM_CIA'])['DT_FIM_EXERC']
                 .agg(Periodos='count', Inicio='min', Fim='max')
                 .reset_index()
                 .sort_values(['SETOR', 'Periodos'], ascending=[True, False]))

        print(f"\n{s2}\n  Cobertura{label}\n{s2}")
        print(f"  {'Empresa':<35} {'Setor':<14} {'Períodos':>8} "
              f"{'Início':>12} {'Fim':>12}")
        print(f"  {'─'*35} {'─'*14} {'─'*8} {'─'*12} {'─'*12}")
        for _, row in cob.iterrows():
            print(f"  {row['DENOM_CIA'][:34]:<35} {row['SETOR']:<14} "
                  f"{row['Periodos']:>8} {str(row['Inicio'].date()):>12} "
                  f"{str(row['Fim'].date()):>12}")

        min_p, max_p = cob['Periodos'].min(), cob['Periodos'].max()
        print(f"\n  Mínimo: {min_p} período(s) | Máximo: {max_p} período(s)")
        if tipo_doc == 'DFP' and min_p < 3:
            print("  ⚠️  Empresas com menos de 3 períodos DFP limitam a modelagem temporal.")


def eda_contas_disponiveis(df):
    """
    BLOCO 3 — Inventário de contas disponíveis por demonstrativo.
    Para cada conta: código CVM, nome em linguagem financeira e
    percentual de preenchimento. Define o universo de KPIs calculáveis.
    """
    print(f"\n{S}\n📂 BLOCO 3 — CONTAS DISPONÍVEIS POR DEMONSTRATIVO\n{S}")

    prefixos = {
        'DRE':    'Demonstração do Resultado',
        'BPA':    'Balanço Patrimonial — Ativo',
        'BPP':    'Balanço Patrimonial — Passivo/PL',
        'DFC_MD': 'Fluxo de Caixa — Método Direto',
        'DFC_MI': 'Fluxo de Caixa — Método Indireto',
        'DVA':    'Demonstração do Valor Adicionado',
        'DMPL':   'Mutações do Patrimônio Líquido',
        'DRA':    'Resultado Abrangente',
    }

    total_contas = 0
    for pref, descricao in prefixos.items():
        cols = sorted(c for c in df.columns if c.startswith(pref + '_'))
        if not cols:
            print(f"\n  {descricao} ({pref}) — nenhuma coluna encontrada")
            continue

        print(f"\n  {descricao} ({pref}) — {len(cols)} contas")
        print(f"  {'Código CVM':<24} {'Nome / Descrição':<38} {'Preench.':>8}")
        print(f"  {'─'*24} {'─'*38} {'─'*8}")
        for col in cols:
            pct   = df[col].notna().mean() * 100
            nome  = MAPA_CONTAS.get(col, '—')
            barra = '█' * int(pct / 10) + '░' * (10 - int(pct / 10))
            print(f"  {col:<24} {nome:<38} {pct:>6.1f}%  {barra}")
        total_contas += len(cols)

    print(f"\n  Total de contas no dataset: {total_contas}")


def eda_qualidade_dados(df):
    """
    BLOCO 4 — Qualidade dos dados.
    Nulos por coluna, cobertura das contas-alvo por setor e
    verificação de valores anômalos (negativos onde não esperado).
    """
    print(f"\n{S}\n🔍 BLOCO 4 — QUALIDADE DOS DADOS\n{S}")

    cols_num = df.select_dtypes(include='number').columns
    nulos    = df[cols_num].isnull().sum()
    pct_nulo = (nulos / len(df) * 100).round(1)
    resumo   = pd.DataFrame({'Nulos': nulos, '% Nulo': pct_nulo})
    resumo   = resumo[resumo['Nulos'] > 0].sort_values('% Nulo', ascending=False)

    print(f"\n  Colunas com valores ausentes: {len(resumo)}")
    print(f"\n  {'Coluna':<30} {'Nulos':>8} {'% Nulo':>8}  Completude")
    print(f"  {'─'*30} {'─'*8} {'─'*8}  {'─'*20}")
    for col, row in resumo.head(30).iterrows():
        comp  = 100 - row['% Nulo']
        barra = '█' * int(comp / 5) + '░' * (20 - int(comp / 5))
        print(f"  {col:<30} {int(row['Nulos']):>8} {row['% Nulo']:>7.1f}%  {barra}")
    if len(resumo) > 30:
        print(f"  ... e mais {len(resumo) - 30} colunas com nulos")

    # Cobertura das contas-alvo por setor
    contas_check = {
        'Receita Líquida': 'DRE_3.01',
        'Lucro Bruto':     'DRE_3.03',
        'EBIT':            'DRE_3.05',
        'Res. antes IR':   'DRE_3.07',
        'Lucro Líquido':   'DRE_3.11',
        'Ativo Total':     'BPA_1',
        'Ativo Circ.':     'BPA_1.01',
        'Passivo Circ.':   'BPP_2.01',
        'Patrimônio Líq.': 'BPP_2.03',
    }

    print(f"\n{s2}\n  Cobertura das contas-alvo por setor (%):\n{s2}")
    linhas = []
    for nome, col in contas_check.items():
        if col not in df.columns:
            continue
        linha = {'Conta': nome}
        for setor, grupo in df.groupby('SETOR'):
            linha[setor] = f"{grupo[col].notna().mean()*100:.0f}%"
        linhas.append(linha)
    if linhas:
        print(pd.DataFrame(linhas).set_index('Conta').to_string())

    # Anomalias
    print(f"\n{s2}\n  Verificação de valores anômalos:\n{s2}")
    for col, nome in [('DRE_3.01', 'Receita Líquida'),
                      ('BPA_1',    'Ativo Total'),
                      ('BPA_1.01', 'Ativo Circulante')]:
        if col not in df.columns:
            continue
        n = (df[col] < 0).sum()
        status = f"⚠️  {n} valor(es) negativo(s)" if n > 0 else "✅ Sem negativos"
        print(f"  {nome:<24} ({col})  {status}")


def eda_estatisticas_descritivas(df):
    """
    BLOCO 5 — Estatísticas descritivas das principais contas.
    Média, mediana, desvio padrão, mínimo e máximo por conta e por setor.
    Base para entender escala, dispersão e heterogeneidade setorial.
    """
    print(f"\n{S}\n📈 BLOCO 5 — ESTATÍSTICAS DESCRITIVAS\n{S}")

    contas = {
        'Receita Líquida': 'DRE_3.01',
        'Lucro Bruto':     'DRE_3.03',
        'EBIT':            'DRE_3.05',
        'Lucro Líquido':   'DRE_3.11',
        'Ativo Total':     'BPA_1',
        'Ativo Circ.':     'BPA_1.01',
        'Caixa e Equiv.':  'BPA_1.01.01',
        'Passivo Circ.':   'BPP_2.01',
        'Patrimônio Líq.': 'BPP_2.03',
    }
    cols_exist = {n: c for n, c in contas.items() if c in df.columns}

    print(f"\n  {'Conta':<24} {'Média':>16} {'Mediana':>16} "
          f"{'Desvio P.':>16} {'Mín':>14} {'Máx':>14}")
    print(f"  {'─'*24} {'─'*16} {'─'*16} {'─'*16} {'─'*14} {'─'*14}")
    for nome, col in cols_exist.items():
        s = df[col].dropna()
        if s.empty:
            continue
        print(f"  {nome:<24} {_fmt_reais(s.mean()):>16} {_fmt_reais(s.median()):>16} "
              f"{_fmt_reais(s.std()):>16} {_fmt_reais(s.min()):>14} "
              f"{_fmt_reais(s.max()):>14}")

    if 'DRE_3.01' in df.columns:
        print(f"\n{s2}\n  Mediana da Receita Líquida por setor:\n{s2}")
        for setor, val in (df.groupby('SETOR')['DRE_3.01']
                             .median()
                             .sort_values(ascending=False)
                             .items()):
            print(f"  {setor:<16} {_fmt_reais(val)}")


def eda_kpis_financeiros(df):
    """
    BLOCO 6 — KPIs financeiros calculados a partir das contas disponíveis.

    São os indicadores presentes em relatórios financeiros reais (RI, relatórios
    de analistas, B3) e os principais candidatos a features e targets na modelagem.

    Rentabilidade — medem eficiência em gerar lucro a partir da receita/ativos:
      Margem Bruta (%)     = Lucro Bruto / Receita Líquida
      Margem EBIT (%)      = EBIT / Receita Líquida
      Margem Líquida (%)   = Lucro Líquido / Receita Líquida
      ROE (%)              = Lucro Líquido / Patrimônio Líquido
      ROA (%)              = Lucro Líquido / Ativo Total

    Liquidez — medem capacidade de honrar obrigações de curto prazo:
      Liquidez Corrente    = Ativo Circulante / Passivo Circulante
      Liquidez Imediata    = Caixa / Passivo Circulante

    Endividamento — medem grau de dependência de capital de terceiros:
      Endividamento (%)    = Passivo Total / Ativo Total
      Alavancagem D/E      = Dívida Bruta / Patrimônio Líquido

    Geração de Caixa — medem qualidade do lucro (caixa vs. competência):
      FCO / Receita (%)    = Fluxo de Caixa Operacional / Receita Líquida
      FCO / Lucro Líq. (%) = Fluxo de Caixa Operacional / Lucro Líquido
    """
    print(f"\n{S}\n💡 BLOCO 6 — KPIs FINANCEIROS CALCULADOS\n{S}")

    df = df.copy()

    def safe_div(num, den):
        return np.where(
            den.isna() | (den == 0) | num.isna(),
            np.nan, num / den
        )

    # Extrai séries base (None se coluna ausente)
    def _col(c):
        return df[c] if c in df.columns else None

    rec    = _col('DRE_3.01')
    lb     = _col('DRE_3.03')
    ebit   = _col('DRE_3.05')
    res_fin= _col('DRE_3.06')   # Resultado Financeiro (negativo = despesa líquida)
    ll     = _col('DRE_3.11')
    at     = _col('BPA_1')
    ac     = _col('BPA_1.01')
    cx     = _col('BPA_1.01.01')
    apl_cp = _col('BPA_1.01.02')  # Aplicações financeiras CP
    pc     = _col('BPP_2.01')
    pl     = _col('BPP_2.03')
    bpp_r  = _col('BPP_2')
    div_cp = _col('BPP_2.01.04')
    div_lp = _col('BPP_2.02.01')
    fco_md = _col('DFC_MD_6.01')
    fco_mi = _col('DFC_MI_6.01')
    fco    = fco_md if fco_md is not None else fco_mi
    da     = _col('DA_TOTAL')   # Depreciação & Amortização — extraída do DFC_MI

    kpis_calc = {}

    # ── Rentabilidade ────────────────────────────────────────────────
    if rec is not None and lb   is not None: kpis_calc['MARGEM_BRUTA_%']      = safe_div(lb,   rec) * 100
    if rec is not None and ebit is not None: kpis_calc['MARGEM_EBIT_%']       = safe_div(ebit, rec) * 100
    if rec is not None and ll   is not None: kpis_calc['MARGEM_LIQUIDA_%']    = safe_div(ll,   rec) * 100
    if pl  is not None and ll   is not None: kpis_calc['ROE_%']               = safe_div(ll,   pl)  * 100
    if at  is not None and ll   is not None: kpis_calc['ROA_%']               = safe_div(ll,   at)  * 100

    # ── Liquidez ──────────────────────────────────────────────────────
    if ac  is not None and pc   is not None: kpis_calc['LIQUIDEZ_CORRENTE']   = safe_div(ac,   pc)
    if cx  is not None and pc   is not None: kpis_calc['LIQUIDEZ_IMEDIATA']   = safe_div(cx,   pc)

    # ── Endividamento e estrutura de capital ──────────────────────────
    if bpp_r is not None and at is not None and pl is not None:
        kpis_calc['ENDIVIDAMENTO_%'] = safe_div(bpp_r - pl, at) * 100
    if div_cp is not None and div_lp is not None and pl is not None:
        kpis_calc['ALAVANCAGEM_DE']  = safe_div(div_cp.fillna(0) + div_lp.fillna(0), pl)

    # Dívida Líquida = Dívida Bruta − Caixa − Aplicações Financeiras CP
    # Positivo: empresa deve mais do que tem disponível
    # Negativo: empresa tem mais caixa do que dívida (caixa líquido)
    if div_cp is not None and div_lp is not None:
        divida_bruta = div_cp.fillna(0) + div_lp.fillna(0)
        caixa_total  = cx.fillna(0) if cx is not None else pd.Series(0, index=df.index)
        if apl_cp is not None:
            caixa_total = caixa_total + apl_cp.fillna(0)
        kpis_calc['DIVIDA_LIQUIDA'] = divida_bruta - caixa_total

    # Índice de Cobertura de Juros = EBIT / |Despesas Financeiras líquidas|
    # Resultado Financeiro (3.06) é normalmente negativo (despesa > receita).
    # Usamos o valor absoluto para que o índice seja positivo quando
    # o EBIT cobre os juros. Abaixo de 1.5x começa a ser preocupante.
    if ebit is not None and res_fin is not None:
        desp_fin_abs = res_fin.abs()
        kpis_calc['COBERTURA_JUROS'] = safe_div(ebit, desp_fin_abs)

    # ── Eficiência ────────────────────────────────────────────────────
    # Giro do Ativo = Receita / Ativo Total
    # Mede quantas vezes o ativo "girou" em receita no período.
    # Alta intensidade de capital (Commodities, Energia) → giro baixo.
    # Baixo capital (Tecnologia, Varejo de serviços) → giro alto.
    if rec is not None and at is not None:
        kpis_calc['GIRO_ATIVO'] = safe_div(rec, at)

    # ── Geração de Caixa ──────────────────────────────────────────────
    if fco is not None and rec is not None:  kpis_calc['FCO_SOBRE_RECEITA_%'] = safe_div(fco,  rec) * 100
    if fco is not None and ll  is not None:  kpis_calc['FCO_SOBRE_LL_%']      = safe_div(fco,  ll)  * 100

    # ── EBITDA — calculado a partir do D&A extraído do DFC_MI ─────────
    # EBITDA = EBIT + D&A
    # Fonte do D&A: subcontas 6.01.01.xx do DFC Método Indireto onde
    # DS_CONTA contém 'deprecia' ou 'amortiz'. A depreciação NÃO aparece
    # como linha separada no DRE padrão da CVM — confirmado com dados reais.
    # Empresas que publicam DFC pelo Método Direto terão EBITDA = NaN.
    if ebit is not None and da is not None:
        ebitda_serie                = ebit + da.fillna(0)
        ebitda_serie[da.isna()]     = np.nan  # preserva NaN onde D&A ausente
        kpis_calc['EBITDA']         = ebitda_serie
        if rec is not None:
            kpis_calc['MARGEM_EBITDA_%'] = safe_div(ebitda_serie, rec) * 100
        n_validos = ebitda_serie.notna().sum()
        n_total   = len(ebitda_serie)
        print(f"\n  📐 EBITDA calculado: {n_validos}/{n_total} registros "
              f"({n_total - n_validos} sem DFC_MI → NaN)")
    else:
        print("\n  ⚠️  EBITDA não calculado: EBIT (DRE_3.05) ou DA_TOTAL ausente")

    for nome, serie in kpis_calc.items():
        df[nome] = pd.Series(serie, index=df.index)

    grupos_kpi = {
        'Rentabilidade (%)':   ['MARGEM_BRUTA_%', 'MARGEM_EBIT_%',
                                 'MARGEM_LIQUIDA_%', 'MARGEM_EBITDA_%',
                                 'ROE_%', 'ROA_%'],
        'EBITDA':              ['EBITDA', 'DA_TOTAL'],
        'Liquidez (x)':        ['LIQUIDEZ_CORRENTE', 'LIQUIDEZ_IMEDIATA'],
        'Endividamento':       ['ENDIVIDAMENTO_%', 'ALAVANCAGEM_DE',
                                'DIVIDA_LIQUIDA', 'COBERTURA_JUROS'],
        'Eficiência':          ['GIRO_ATIVO'],
        'Geração de Caixa':    ['FCO_SOBRE_RECEITA_%', 'FCO_SOBRE_LL_%'],
    }

    print(f"\n  {'KPI':<28} {'Média':>10} {'Mediana':>10} "
          f"{'Desvio P.':>10} {'Mín':>10} {'Máx':>10} {'Válidos':>8}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    for grupo, nomes in grupos_kpi.items():
        print(f"\n  [{grupo}]")
        for kpi in nomes:
            if kpi not in df.columns:
                continue
            s = df[kpi].replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue
            print(f"  {kpi:<28} {s.mean():>10.2f} {s.median():>10.2f} "
                  f"{s.std():>10.2f} {s.min():>10.2f} {s.max():>10.2f} "
                  f"{len(s):>8}")

    print(f"\n{s2}\n  Margem Líquida e ROE — mediana por setor:\n{s2}")
    print(f"  {'Setor':<16} {'Margem Líq. (%)':>18} {'ROE (%)':>12}")
    print(f"  {'─'*16} {'─'*18} {'─'*12}")
    for setor, grupo in df.groupby('SETOR'):
        ml  = grupo['MARGEM_LIQUIDA_%'].replace([np.inf,-np.inf],np.nan).median() \
              if 'MARGEM_LIQUIDA_%' in df.columns else np.nan
        roe = grupo['ROE_%'].replace([np.inf,-np.inf],np.nan).median() \
              if 'ROE_%' in df.columns else np.nan
        print(f"  {setor:<16} {ml:>17.2f}%  {roe:>10.2f}%")

    return df, list(kpis_calc.keys())


def eda_correlacoes(df, kpis):
    """
    BLOCO 7 — Correlações entre KPIs e contas brutas.
    Identifica variáveis que se movem juntas — base para a
    feature selection da próxima etapa do CRISP-DM.
    Salva heatmap em eda_correlacoes.png.
    """
    print(f"\n{S}\n🔗 BLOCO 7 — CORRELAÇÕES ENTRE VARIÁVEIS\n{S}")

    contas_brutas = ['DRE_3.01', 'DRE_3.03', 'DRE_3.05', 'DRE_3.11',
                     'BPA_1', 'BPA_1.01', 'BPP_2.01', 'BPP_2.03']
    cols_corr = [c for c in contas_brutas + kpis if c in df.columns]
    df_corr   = (df[cols_corr]
                 .replace([np.inf, -np.inf], np.nan)
                 .dropna(how='all'))

    if df_corr.shape[1] < 2:
        print("  Colunas insuficientes para calcular correlações.")
        return

    # Usa MAPA_CONTAS para todos os nomes — cobre tanto contas brutas quanto KPIs derivados
    renomeia    = {c: MAPA_CONTAS.get(c, c) for c in df_corr.columns}
    df_corr     = df_corr.rename(columns=renomeia)
    corr_matrix = df_corr.corr(method='pearson')

    for alvo_col, alvo_nome in [('DRE_3.01', 'Receita Líquida'),
                                 ('DRE_3.11', 'Lucro Líquido')]:
        alvo_leg = MAPA_CONTAS.get(alvo_col, alvo_col)
        if alvo_leg not in corr_matrix.columns:
            continue
        print(f"\n  Top correlações com {alvo_nome}:")
        corrs = (corr_matrix[alvo_leg]
                 .drop(alvo_leg, errors='ignore')
                 .abs()
                 .sort_values(ascending=False)
                 .head(10))
        for var, val in corrs.items():
            sinal = '+' if corr_matrix.loc[var, alvo_leg] > 0 else '−'
            barra = '█' * int(abs(val) * 20)
            print(f"  {sinal} {var:<38} r={val:.3f}  {barra}")

    try:
        fig, ax = plt.subplots(figsize=(14, 11))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdYlGn', center=0, linewidths=0.5,
                    annot_kws={'size': 7}, ax=ax)
        ax.set_title('Matriz de Correlação — KPIs Financeiros CVM',
                     fontsize=13, pad=12)
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig('eda_correlacoes.png', dpi=150)
        plt.close()
        print(f"\n  💾 Heatmap salvo: eda_correlacoes.png")
    except Exception as e:
        print(f"\n  ⚠️  Heatmap não gerado: {e}")


def eda_evolucao_temporal(df):
    """
    BLOCO 8 — Evolução temporal por setor.
    Gráfico da mediana da Receita Líquida e Lucro Líquido por setor
    ao longo dos períodos — revela tendências e sazonalidade.
    Salvo em eda_evolucao_temporal.png.
    """
    print(f"\n{S}\n📉 BLOCO 8 — EVOLUÇÃO TEMPORAL POR SETOR\n{S}")

    contas = {n: c for n, c in
              [('Receita Líquida', 'DRE_3.01'),
               ('Lucro Líquido',   'DRE_3.11'),
               ('EBITDA',          'EBITDA')]
              if c in df.columns}

    if not contas:
        print("  DRE_3.01 / DRE_3.11 não encontradas — bloco pulado.")
        return

    try:
        fig, axes = plt.subplots(1, len(contas),
                                 figsize=(7 * len(contas), 5), squeeze=False)
        for idx, (nome, col) in enumerate(contas.items()):
            ax = axes[0][idx]
            for setor, grupo in df.groupby('SETOR'):
                ts = (grupo.groupby('DT_FIM_EXERC')[col]
                            .median().dropna().sort_index())
                if len(ts) < 2:
                    continue
                ax.plot(ts.index, ts.values / 1e9, marker='o', label=setor)
            ax.set_title(f'{nome} — mediana por setor (R$ bi)')
            ax.set_xlabel('Período')
            ax.set_ylabel('R$ bilhões')
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig('eda_evolucao_temporal.png', dpi=150)
        plt.close()
        print(f"  💾 Gráfico salvo: eda_evolucao_temporal.png")
    except Exception as e:
        print(f"  ⚠️  Gráfico não gerado: {e}")


def eda_inventario_kpis(kpis_disponiveis, df):
    """
    BLOCO 9 — Inventário final de KPIs para a modelagem.
    Classifica cada KPI como calculado, parcialmente disponível ou
    indisponível. Serve como guia para a etapa de Preparação dos Dados.
    """
    print(f"\n{S}\n📋 BLOCO 9 — INVENTÁRIO DE KPIs PARA A MODELAGEM\n{S}")

    todos_kpis = {
        # Rentabilidade
        'MARGEM_BRUTA_%':      ('Rentabilidade',  'DRE_3.01', 'DRE_3.03'),
        'MARGEM_EBIT_%':       ('Rentabilidade',  'DRE_3.01', 'DRE_3.05'),
        'MARGEM_LIQUIDA_%':    ('Rentabilidade',  'DRE_3.01', 'DRE_3.11'),
        'MARGEM_EBITDA_%':     ('Rentabilidade',  'DRE_3.01', 'DA_TOTAL+DRE_3.05'),
        'ROE_%':               ('Rentabilidade',  'BPP_2.03', 'DRE_3.11'),
        'ROA_%':               ('Rentabilidade',  'BPA_1',    'DRE_3.11'),
        # EBITDA
        'EBITDA':              ('EBITDA',          'DRE_3.05', 'DA_TOTAL'),
        'DA_TOTAL':            ('EBITDA',          'DFC_MI 6.01.01.xx', '—'),
        # Liquidez
        'LIQUIDEZ_CORRENTE':   ('Liquidez',        'BPA_1.01', 'BPP_2.01'),
        'LIQUIDEZ_IMEDIATA':   ('Liquidez',        'BPA_1.01.01', 'BPP_2.01'),
        # Endividamento
        'ENDIVIDAMENTO_%':     ('Endividamento',   'BPP_2',    'BPA_1'),
        'ALAVANCAGEM_DE':      ('Endividamento',   'BPP_2.01.04', 'BPP_2.03'),
        'DIVIDA_LIQUIDA':      ('Endividamento',   'BPP_2.01.04+BPP_2.02.01', 'BPA_1.01.01'),
        'COBERTURA_JUROS':     ('Endividamento',   'DRE_3.05', 'DRE_3.06'),
        # Eficiência
        'GIRO_ATIVO':          ('Eficiência',      'DRE_3.01', 'BPA_1'),
        # Geração de caixa
        'FCO_SOBRE_RECEITA_%': ('Fluxo de Caixa',  'DFC_*.6.01', 'DRE_3.01'),
        'FCO_SOBRE_LL_%':      ('Fluxo de Caixa',  'DFC_*.6.01', 'DRE_3.11'),
        # Crescimento — derivados com shift/pct_change na Preparação dos Dados
        'CRESC_RECEITA_YOY_%': ('Crescimento',     'DRE_3.01',  '—'),
        'CRESC_LUCRO_YOY_%':   ('Crescimento',     'DRE_3.11',  '—'),
        'CRESC_EBITDA_YOY_%':  ('Crescimento',     'EBITDA',    '—'),
        'CRESC_ATIVO_YOY_%':   ('Crescimento',     'BPA_1',     '—'),
    }

    print(f"\n  {'KPI':<26} {'Grupo':<16} {'Status':<15} {'Cobertura':>10}")
    print(f"  {'─'*26} {'─'*16} {'─'*15} {'─'*10}")

    for kpi, (grupo, c1, _) in todos_kpis.items():
        if kpi in kpis_disponiveis and kpi in df.columns:
            cob    = df[kpi].replace([np.inf,-np.inf],np.nan).notna().mean()*100
            status = '✅ Calculado'
        elif c1.split('/')[0].strip().replace('*','MD') in df.columns:
            cob    = df[c1.split('/')[0].strip()
                         .replace('*','MD')].notna().mean()*100 \
                     if c1.split('/')[0].strip().replace('*','MD') in df.columns \
                     else 0.0
            status = '⚠️  Parcial'
        else:
            cob    = 0.0
            status = '❌ Indisponível'
        print(f"  {kpi:<26} {grupo:<16} {status:<15} {cob:>9.1f}%")

    print(f"\n{s2}")
    print("  ✅ 'Calculado'    → pronto para uso na modelagem")
    print("  ⚠️  'Parcial'      → requer tratamento na Preparação dos Dados")
    print("  📐 'Crescimento'  → será derivado com shift/pct_change na próxima etapa")
    print("  ❌ 'Indisponível' → conta base ausente no dataset")
    print(S)


# =============================================================================
# 5. EXECUÇÃO
# =============================================================================

def executar_eda(df):
    """Executa todos os 9 blocos de EDA em sequência."""
    eda_visao_geral(df)
    eda_cobertura_temporal(df)
    eda_contas_disponiveis(df)
    eda_qualidade_dados(df)
    eda_estatisticas_descritivas(df)
    df, kpis = eda_kpis_financeiros(df)
    eda_correlacoes(df, kpis)
    eda_evolucao_temporal(df)
    eda_inventario_kpis(kpis, df)

    # Salva dataset enriquecido com KPIs já calculados
    df.to_csv('cvm_eda_com_kpis.csv', index=False, sep=';', encoding='utf-8-sig')
    print(f"\n💾 Dataset com KPIs salvo em: cvm_eda_com_kpis.csv")
    print(f"   ({df.shape[0]} registros × {df.shape[1]} colunas)\n")
    return df


if __name__ == '__main__':
    # ── Passo 1: Ingestão e consolidação dos ZIPs da CVM ─────────────
    df = processar_tudo()
    if df is None:
        raise SystemExit("Nenhum dado processado. Verifique as pastas e os ZIPs.")

    # ── Passo 2: EDA completa — 9 blocos ─────────────────────────────
    df = executar_eda(df)