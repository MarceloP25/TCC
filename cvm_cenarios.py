# =============================================================================
# ANÁLISE DE CENÁRIOS ESTRATÉGICOS — ETAPA 5 DO CRISP-DM (Implantação)
# TCC: Predição de Indicadores Financeiros com Machine Learning
# =============================================================================
# Entrada : resultados/avaliacao_resultados.json   — gerado por cvm_avaliacao.py
#           modelos/[target]_[algoritmo].joblib     — gerado por cvm_treino.py
#           cvm_dataset_treino.csv                 — gerado por cvm_preparacao.py
#           cvm_dataset_teste.csv                  — gerado por cvm_preparacao.py
#
# Saídas  : resultados/
#               cenarios_resultados.csv     — tabela comparativa por empresa/cenário
#               cenarios_resultados.json    — estrutura completa para o TCC
#               relatorio_cenarios.txt      — narrativa pronta para o TCC
#               graficos/
#                   cenario_[empresa]_[setor].png
#           logs/
#               log_cenarios.txt
#
# Responsabilidade: usar modelos JÁ TREINADOS para simular cenários
# estratégicos e interpretar os resultados com auxílio de IA Generativa.
# Pode ser executado múltiplas vezes sem retreinar os modelos.
#
# ─────────────────────────────────────────────────────────────────────────────
# INTEGRAÇÃO COM IA GENERATIVA
# ─────────────────────────────────────────────────────────────────────────────
# Padrão ativo: Google Gemini 1.5 Flash (GRATUITO)
#   - Tier gratuito: 15 req/min, 1 milhão tokens/dia
#   - Chave gratuita em: https://aistudio.google.com
#   - Sem cartão de crédito necessário
#   - Configurar: export GEMINI_API_KEY="sua-chave-aqui"
#
# Alternativas documentadas (desativadas por padrão — requerem conta paga):
#   - Anthropic Claude  : export LLM_PROVIDER="anthropic"
#                         export ANTHROPIC_API_KEY="sua-chave"
#   - OpenAI GPT-4o     : export LLM_PROVIDER="openai"
#                         export OPENAI_API_KEY="sua-chave"
#   - Ollama (local)    : export LLM_PROVIDER="ollama"
#                         (requer Ollama instalado localmente — 100% offline)
#
# Para trocar de provedor: ajuste LLM_PROVIDER no ambiente ou diretamente
# na constante LLM_PROVIDER abaixo. O prompt enviado é idêntico para todos.
# ─────────────────────────────────────────────────────────────────────────────
#
# Uso: python cvm_cenarios.py  (execute APÓS cvm_treino.py e cvm_avaliacao.py)
# Dependências: pandas, numpy, scikit-learn, joblib, matplotlib
# =============================================================================

import os
import json
import time
import warnings
import joblib
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PASTA_MODELOS    = 'modelos'
PASTA_RESULTADOS = 'resultados'
PASTA_GRAFICOS   = os.path.join(PASTA_RESULTADOS, 'graficos')
PASTA_LOGS       = 'logs'

ARQUIVO_TREINO       = 'cvm_dataset_treino.csv'
ARQUIVO_TESTE        = 'cvm_dataset_teste.csv'
AVALIACAO_JSON       = os.path.join(PASTA_RESULTADOS, 'avaliacao_resultados.json')

TARGETS = {
    'TARGET_DRE_3.01': 'Receita Líquida (t+1)',
    'TARGET_DRE_3.11': 'Lucro Líquido (t+1)',
    'TARGET_EBITDA'  : 'EBITDA (t+1)',
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO PROVEDOR DE LLM
# Altere LLM_PROVIDER aqui ou via variável de ambiente para trocar de provedor.
# Opções: 'gemini' | 'anthropic' | 'openai' | 'ollama'
# ─────────────────────────────────────────────────────────────────────────────
LLM_PROVIDER   = os.environ.get('LLM_PROVIDER',    'gemini')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY',  '')
ANTHROPIC_KEY  = os.environ.get('ANTHROPIC_API_KEY', '')
OPENAI_KEY     = os.environ.get('OPENAI_API_KEY',  '')
OLLAMA_URL     = os.environ.get('OLLAMA_URL',      'http://localhost:11434')
OLLAMA_MODEL   = os.environ.get('OLLAMA_MODEL',    'llama3')

# Empresas âncora — uma por setor — para a análise de cenários
EMPRESAS_ANCORA = {
    'Petróleo'   : {'cnpj': '33.000.167/0001-01', 'nome': 'Petrobras'},
    'Energia'    : {'cnpj': '03.220.438/0001-73', 'nome': 'Equatorial Energia'},
    'Varejo'     : {'cnpj': '47.960.950/0001-21', 'nome': 'Magazine Luiza'},
    'Commodities': {'cnpj': '33.592.510/0001-54', 'nome': 'Vale'},
    'Tecnologia' : {'cnpj': '84.429.695/0001-11', 'nome': 'WEG'},
}

# ─────────────────────────────────────────────────────────────────────────────
# DEFINIÇÃO DOS 3 CENÁRIOS ESTRATÉGICOS
# ─────────────────────────────────────────────────────────────────────────────
# Cada cenário define:
#   descricao  : texto explicativo para o LLM e para o TCC
#   hipotese   : o que o cenário testa (pergunta de negócio implícita)
#   ajustes    : dicionário de ajustes nos KPIs
#                  tipo='delta' → soma valor ao KPI original
#                  tipo='fator' → multiplica KPI original pelo fator
#
# Os valores dos ajustes são baseados em movimentos financeiros realistas:
#   Alavancagem: captação equivalente a 0,5x PL é comum em expansões corporativas
#   Eficiência : redução de 10% em custos é meta típica de programas de eficiência
#   CAPEX       : aumento de 30% em investimentos é relevante sem ser extremo
# ─────────────────────────────────────────────────────────────────────────────
CENARIOS = {

    'Expansão com Alavancagem': {
        'descricao': (
            'A empresa capta nova dívida de longo prazo equivalente a 0,5 vez '
            'o Patrimônio Líquido atual. O endividamento e a alavancagem aumentam, '
            'a cobertura de juros e a liquidez corrente são pressionadas, mas abre-se '
            'espaço para crescimento de receita via expansão de capacidade ou aquisições.'
        ),
        'hipotese': (
            'O crescimento de receita gerado pela nova capacidade compensa '
            'o custo financeiro adicional da dívida captada?'
        ),
        'ajustes': {
            # Dívida nova eleva alavancagem e endividamento geral
            'ALAVANCAGEM_DE'   : {'tipo': 'delta', 'valor':  0.50},
            'ENDIVIDAMENTO_%'  : {'tipo': 'delta', 'valor':  8.00},
            # Novas despesas financeiras reduzem cobertura de juros
            'COBERTURA_JUROS'  : {'tipo': 'fator', 'valor':  0.75},
            # Caixa parcialmente consumido na transação → liquidez menor
            'LIQUIDEZ_CORRENTE': {'tipo': 'fator', 'valor':  0.90},
            'LIQUIDEZ_IMEDIATA': {'tipo': 'fator', 'valor':  0.85},
            # Expectativa de crescimento de receita pela nova capacidade
            'DRE_3.01_YOY'    : {'tipo': 'delta', 'valor':  5.00},
        },
    },

    'Eficiência Operacional': {
        'descricao': (
            'A empresa implementa um programa estrutural de redução de custos '
            'operacionais (CPV + despesas operacionais) de 10%. As margens operacional, '
            'EBITDA e líquida melhoram diretamente, sem impacto no endividamento, '
            'nos ativos ou na estrutura de capital.'
        ),
        'hipotese': (
            'Um ganho de eficiência operacional de 10% nos custos '
            'se traduz em quanto de melhora nos indicadores de resultado futuro?'
        ),
        'ajustes': {
            # Redução de custos melhora todas as margens diretamente
            'MARGEM_BRUTA_%'   : {'tipo': 'delta', 'valor':  2.50},
            'MARGEM_EBIT_%'    : {'tipo': 'delta', 'valor':  3.00},
            'MARGEM_EBITDA_%'  : {'tipo': 'delta', 'valor':  3.50},
            'MARGEM_LIQUIDA_%' : {'tipo': 'delta', 'valor':  2.00},
            # Melhor lucro líquido melhora ROE e ROA
            'ROE_%'            : {'tipo': 'delta', 'valor':  2.00},
            'ROA_%'            : {'tipo': 'delta', 'valor':  1.00},
            # FCO melhora junto com as margens
            'FCO_SOBRE_RECEITA_%': {'tipo': 'delta', 'valor': 1.50},
        },
    },

    'Investimento em Capacidade (CAPEX)': {
        'descricao': (
            'A empresa aumenta em 30% o investimento em imobilizado em relação ao '
            'exercício anterior. No curto prazo, o caixa é pressionado e o giro do '
            'ativo cai (mais ativos para a mesma receita atual). Em t+1, espera-se '
            'crescimento de receita pela nova capacidade instalada.'
        ),
        'hipotese': (
            'Um investimento pesado em capacidade produtiva '
            'gera crescimento de receita suficiente para justificar a '
            'pressão de curto prazo na liquidez e no fluxo de caixa?'
        ),
        'ajustes': {
            # Mais ativos para a mesma receita → giro cai
            'GIRO_ATIVO'           : {'tipo': 'fator', 'valor':  0.88},
            # Caixa consumido no investimento → liquidez pressionada
            'LIQUIDEZ_IMEDIATA'    : {'tipo': 'fator', 'valor':  0.85},
            # FCO absorve parte do investimento
            'FCO_SOBRE_RECEITA_%'  : {'tipo': 'fator', 'valor':  0.90},
            # Ativo total cresce com o novo imobilizado
            'CRESC_ATIVO_YOY_%'   : {'tipo': 'delta', 'valor': 12.00},
            # Expectativa de crescimento de receita pela nova capacidade
            'DRE_3.01_YOY'        : {'tipo': 'delta', 'valor':  8.00},
        },
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
        f.flush()


def fmt_bi(v):
    """Formata valor monetário em R$ bilhões."""
    if pd.isna(v):
        return 'N/A'
    return f'R$ {v / 1e9:,.2f} bi'


def fmt_var(v):
    """Formata variação percentual com sinal."""
    if pd.isna(v):
        return 'N/A'
    sinal = '+' if v >= 0 else ''
    return f'{sinal}{v:.1f}%'


def garantir_pastas():
    for p in [PASTA_RESULTADOS, PASTA_GRAFICOS, PASTA_LOGS]:
        os.makedirs(p, exist_ok=True)


# =============================================================================
# MÓDULO DE IA GENERATIVA — ADAPTADOR MULTI-PROVEDOR
# =============================================================================
# Arquitetura: o prompt é construído uma única vez e enviado para qualquer
# provedor via a função chamar_llm(). Trocar de provedor não altera o prompt
# nem o restante do código — apenas a função de envio HTTP muda.
# =============================================================================

def _prompt_cenario(empresa, setor, cenario_nome, cenario_desc,
                    hipotese, kpis_base, resultado_comparativo):
    """
    Constrói o prompt enviado ao LLM.
    Idêntico independentemente do provedor escolhido.
    """
    linhas_kpis = '\n'.join([
        f'  {k}: {v:.4g}'
        for k, v in sorted(kpis_base.items())
        if isinstance(v, (int, float)) and not pd.isna(v)
        and k not in ('CNPJ_CIA', 'ANO_REF')
    ][:20])  # limita a 20 KPIs para não estourar o contexto

    linhas_resultado = '\n'.join([
        f'  {t}: {fmt_bi(vb)} → {fmt_bi(vc)}  ({fmt_var(((vc-vb)/abs(vb)*100) if vb and vb != 0 else float("nan"))})'
        for t, (vb, vc) in resultado_comparativo.items()
        if not pd.isna(vb) and not pd.isna(vc)
    ])

    return f"""Você é um analista financeiro sênior especializado em empresas brasileiras de capital aberto.

EMPRESA  : {empresa}
SETOR    : {setor}
CENÁRIO  : {cenario_nome}

DESCRIÇÃO DO CENÁRIO:
{cenario_desc}

HIPÓTESE A AVALIAR:
{hipotese}

PRINCIPAIS KPIs ATUAIS DA EMPRESA (período base):
{linhas_kpis}

IMPACTO PREVISTO PELO MODELO DE MACHINE LEARNING (t+1):
{linhas_resultado}

Com base nesses dados, redija uma análise executiva objetiva em exatamente 4 parágrafos:

1. IMPACTO GERAL: avalie o efeito financeiro do cenário nos indicadores previstos.
2. PONTOS POSITIVOS E RISCOS: identifique oportunidades e riscos da decisão estratégica.
3. COMPARAÇÃO SETORIAL: compare com o perfil típico do setor {setor} — a reação seria esperada ou surpreendente?
4. RECOMENDAÇÃO: forneça uma recomendação objetiva considerando o balanço entre oportunidade e risco.

Use linguagem direta adequada para um conselho de administração. Cite os números relevantes."""


def _gemini(prompt, api_key):
    """
    Chama Google Gemini 1.5 Flash via REST.
    Gratuito: https://aistudio.google.com/app/apikey
    Limite: 15 req/min | 1M tokens/dia
    """
    if not api_key:
        return (
            '[Gemini não configurado]\n'
            'Para ativar: export GEMINI_API_KEY="sua-chave"\n'
            'Obtenha sua chave gratuita em: https://aistudio.google.com/app/apikey'
        )

    url = (
        'https://generativelanguage.googleapis.com/v1beta/models/'
        f'gemini-1.5-flash:generateContent?key={api_key}'
    )
    payload = json.dumps({
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature'    : 0.3,   # baixo para análise financeira objetiva
            'maxOutputTokens': 900,
        },
    }).encode('utf-8')

    req = urllib.request.Request(
        url, data=payload,
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data['candidates'][0]['content']['parts'][0]['text']
    except urllib.error.HTTPError as e:
        corpo = e.read().decode('utf-8', errors='ignore')
        return f'[Erro Gemini HTTP {e.code}]: {corpo[:300]}'
    except Exception as e:
        return f'[Erro Gemini]: {e}'


def _anthropic(prompt, api_key):
    """
    Chama Anthropic Claude Sonnet via REST.
    Requer conta paga: https://console.anthropic.com
    """
    if not api_key:
        return '[Anthropic não configurado — configure ANTHROPIC_API_KEY]'

    payload = json.dumps({
        'model'     : 'claude-sonnet-4-20250514',
        'max_tokens': 900,
        'messages'  : [{'role': 'user', 'content': prompt}],
    }).encode('utf-8')

    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=payload,
        headers={
            'Content-Type'      : 'application/json',
            'x-api-key'         : api_key,
            'anthropic-version' : '2023-06-01',
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data['content'][0]['text']
    except Exception as e:
        return f'[Erro Anthropic]: {e}'


def _openai(prompt, api_key):
    """
    Chama OpenAI GPT-4o via REST.
    Requer conta paga: https://platform.openai.com
    """
    if not api_key:
        return '[OpenAI não configurado — configure OPENAI_API_KEY]'

    payload = json.dumps({
        'model'      : 'gpt-4o',
        'max_tokens' : 900,
        'temperature': 0.3,
        'messages'   : [{'role': 'user', 'content': prompt}],
    }).encode('utf-8')

    req = urllib.request.Request(
        'https://api.openai.com/v1/chat/completions',
        data=payload,
        headers={
            'Content-Type' : 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data['choices'][0]['message']['content']
    except Exception as e:
        return f'[Erro OpenAI]: {e}'


def _ollama(prompt, base_url, model):
    """
    Chama modelo local via Ollama (100% offline, sem API key).
    Instalação: https://ollama.com
    Modelos recomendados: llama3, mistral, gemma2
    """
    url     = f'{base_url.rstrip("/")}/api/generate'
    payload = json.dumps({
        'model' : model,
        'prompt': prompt,
        'stream': False,
        'options': {'temperature': 0.3, 'num_predict': 900},
    }).encode('utf-8')

    req = urllib.request.Request(
        url, data=payload,
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get('response', '[Resposta vazia do Ollama]')
    except Exception as e:
        return f'[Erro Ollama]: {e}'


def chamar_llm(prompt):
    """
    Roteador principal. Direciona o prompt para o provedor configurado
    em LLM_PROVIDER e retorna o texto da resposta.

    Adiciona um pequeno delay entre chamadas para respeitar os rate limits
    do tier gratuito do Gemini (15 req/min = 1 a cada 4 segundos).
    """
    time.sleep(4)   # rate limit conservador para o tier gratuito

    provider = LLM_PROVIDER.lower()

    if provider == 'gemini':
        return _gemini(prompt, GEMINI_API_KEY)
    elif provider == 'anthropic':
        return _anthropic(prompt, ANTHROPIC_KEY)
    elif provider == 'openai':
        return _openai(prompt, OPENAI_KEY)
    elif provider == 'ollama':
        return _ollama(prompt, OLLAMA_URL, OLLAMA_MODEL)
    else:
        return (
            f'[Provedor "{LLM_PROVIDER}" desconhecido]\n'
            f'Opções válidas: gemini | anthropic | openai | ollama'
        )


# =============================================================================
# CARREGAMENTO
# =============================================================================

def carregar_tudo(log_f):
    log(f'\n{S}\n📂 CARREGAMENTO\n{S}', log_f)

    for arq in [AVALIACAO_JSON, ARQUIVO_TREINO, ARQUIVO_TESTE]:
        if not os.path.exists(arq):
            raise FileNotFoundError(
                f'"{arq}" não encontrado.\n'
                'Execute cvm_treino.py e cvm_avaliacao.py antes deste script.'
            )

    with open(AVALIACAO_JSON, encoding='utf-8') as f:
        avaliacao = json.load(f)

    treino = pd.read_csv(ARQUIVO_TREINO, sep=';', encoding='utf-8-sig', low_memory=False)
    teste  = pd.read_csv(ARQUIVO_TESTE,  sep=';', encoding='utf-8-sig', low_memory=False)

    for df in [treino, teste]:
        df['DT_FIM_EXERC'] = pd.to_datetime(df['DT_FIM_EXERC'], errors='coerce')

    # Carrega o melhor modelo por target a partir do JSON de avaliação
    modelos = {}
    for tgt_col, info in avaliacao.get('melhor_por_target', {}).items():
        arq_joblib = os.path.join(PASTA_MODELOS, info['arquivo_joblib'])
        if not os.path.exists(arq_joblib):
            log(f'  ⚠️  Modelo não encontrado: {arq_joblib}', log_f)
            continue
        modelos[tgt_col] = joblib.load(arq_joblib)
        log(f'  ✅ {tgt_col}: {info["algoritmo"]} '
            f'(MAPE={info["mape_teste"]:.1f}%)', log_f)

    log(f'\n  Provedor LLM ativo: {LLM_PROVIDER.upper()}', log_f)
    if LLM_PROVIDER == 'gemini':
        status = '✅ API Key configurada' if GEMINI_API_KEY else '⚠️  GEMINI_API_KEY não definida'
        log(f'  Status Gemini     : {status}', log_f)

    return avaliacao, treino, teste, modelos


# =============================================================================
# APLICAÇÃO DOS CENÁRIOS
# =============================================================================

def aplicar_ajustes(linha_base, ajustes):
    """
    Aplica os ajustes do cenário sobre os KPIs de uma linha do dataset.

    tipo='delta': soma o valor ao KPI original
      Ex: ALAVANCAGEM_DE delta +0.5 → se original era 1.2, vira 1.7

    tipo='fator': multiplica o KPI pelo fator
      Ex: COBERTURA_JUROS fator 0.75 → se original era 4.0, vira 3.0

    KPIs ausentes ou NaN na linha base são ignorados silenciosamente.
    Retorna a linha com os ajustes aplicados (cópia, não modifica original).
    """
    linha = linha_base.copy()
    for kpi, cfg in ajustes.items():
        if kpi not in linha.index:
            continue
        val = linha[kpi]
        if pd.isna(val):
            continue
        if cfg['tipo'] == 'delta':
            linha[kpi] = val + cfg['valor']
        elif cfg['tipo'] == 'fator':
            linha[kpi] = val * cfg['valor']
    return linha


def prever_com_linha(linha, modelo_artefato):
    """
    Executa a predição para uma linha de dados usando o modelo carregado.
    Filtra apenas as features que existem na linha, preenche NaN com 0
    (não há como imputar numa linha única sem contexto de distribuição).
    Retorna o valor predito ou NaN em caso de erro.
    """
    modelo = modelo_artefato['modelo']
    feats  = modelo_artefato['features']

    feats_disp = [f for f in feats if f in linha.index]
    if not feats_disp:
        return np.nan

    X = linha[feats_disp].values.reshape(1, -1)
    X = np.where(np.isnan(X.astype(float)), 0.0, X.astype(float))

    try:
        return float(modelo.predict(X)[0])
    except Exception:
        return np.nan


# =============================================================================
# ANÁLISE PRINCIPAL DE CENÁRIOS
# =============================================================================

def analisar_cenarios(avaliacao, treino, teste, modelos, log_f):
    """
    Para cada empresa âncora × cada cenário estratégico:

    1. Identifica o período mais recente disponível da empresa
       (último registro do dataset completo, prioriza teste)
    2. Calcula o baseline: predição sem nenhum ajuste
    3. Para cada cenário, aplica os ajustes nos KPIs e prediz novamente
    4. Calcula a variação percentual (cenário vs. baseline) por target
    5. Monta o prompt e envia ao LLM para interpretação executiva
    6. Registra tudo em memória para geração posterior de relatórios

    Estrutura de retorno:
    {
      'Petróleo': {
        'empresa': 'Petrobras',
        'periodo_base': '2024-12-31',
        'baseline': {'TARGET_DRE_3.01': 580e9, ...},
        'cenarios': {
          'Expansão com Alavancagem': {
            'predicoes': {'TARGET_DRE_3.01': 610e9, ...},
            'variacoes': {'TARGET_DRE_3.01': +5.2, ...},
            'interpretacao': '...'
          }, ...
        }
      }, ...
    }
    """
    log(f'\n{S}\n🎯 ANÁLISE DE CENÁRIOS ESTRATÉGICOS\n{S}', log_f)

    df_completo = pd.concat([treino, teste], ignore_index=True)
    df_completo['DT_FIM_EXERC'] = pd.to_datetime(df_completo['DT_FIM_EXERC'], errors='coerce')

    todos_resultados = {}
    linhas_csv       = []

    for setor, emp_info in EMPRESAS_ANCORA.items():
        cnpj     = emp_info['cnpj']
        nome_emp = emp_info['nome']

        df_emp = (df_completo[df_completo['CNPJ_CIA'] == cnpj]
                  .sort_values('DT_FIM_EXERC'))

        if df_emp.empty:
            log(f'\n  ⚠️  {nome_emp} ({cnpj}): não encontrada no dataset.', log_f)
            continue

        linha_base  = df_emp.iloc[-1]
        periodo_str = (str(linha_base['DT_FIM_EXERC'].date())
                       if pd.notna(linha_base['DT_FIM_EXERC']) else 'N/D')

        log(f'\n{s2}', log_f)
        log(f'  🏢 {nome_emp}  |  {setor}  |  Período base: {periodo_str}', log_f)
        log(f'{s2}', log_f)

        # ── Baseline ─────────────────────────────────────────────────────────
        baseline = {}
        for tgt_col in TARGETS:
            if tgt_col in modelos:
                baseline[tgt_col] = prever_com_linha(linha_base, modelos[tgt_col])
            else:
                baseline[tgt_col] = np.nan

        log(f'\n  Baseline (sem cenário):', log_f)
        for tgt_col, tgt_nome in TARGETS.items():
            log(f'    {tgt_nome:<30} {fmt_bi(baseline.get(tgt_col, np.nan))}', log_f)

        todos_resultados[setor] = {
            'empresa'     : nome_emp,
            'cnpj'        : cnpj,
            'periodo_base': periodo_str,
            'baseline'    : baseline,
            'cenarios'    : {},
        }

        # ── Cenários ─────────────────────────────────────────────────────────
        for nome_cen, cfg_cen in CENARIOS.items():
            log(f'\n  📋 Cenário: {nome_cen}', log_f)

            linha_cen  = aplicar_ajustes(linha_base, cfg_cen['ajustes'])
            predicoes  = {}
            variacoes  = {}
            resultado_para_llm = {}

            for tgt_col, tgt_nome in TARGETS.items():
                if tgt_col not in modelos:
                    predicoes[tgt_col] = np.nan
                    variacoes[tgt_col] = np.nan
                    continue

                pred_cen = prever_com_linha(linha_cen, modelos[tgt_col])
                pred_base = baseline.get(tgt_col, np.nan)

                predicoes[tgt_col] = pred_cen
                if not pd.isna(pred_base) and pred_base != 0:
                    variacoes[tgt_col] = (pred_cen - pred_base) / abs(pred_base) * 100
                else:
                    variacoes[tgt_col] = np.nan

                resultado_para_llm[tgt_nome] = (pred_base, pred_cen)

                log(f'    {tgt_nome:<30} {fmt_bi(pred_base)} → {fmt_bi(pred_cen)} '
                    f'({fmt_var(variacoes[tgt_col])})', log_f)

                linhas_csv.append({
                    'setor'         : setor,
                    'empresa'       : nome_emp,
                    'cnpj'          : cnpj,
                    'periodo_base'  : periodo_str,
                    'cenario'       : nome_cen,
                    'target'        : tgt_nome,
                    'baseline'      : pred_base,
                    'cenario_valor' : pred_cen,
                    'variacao_pct'  : variacoes[tgt_col],
                })

            # ── Interpretação LLM ─────────────────────────────────────────
            kpis_base_dict = {
                k: v for k, v in linha_base.items()
                if isinstance(v, (int, float)) and not pd.isna(v)
            }
            prompt = _prompt_cenario(
                nome_emp, setor, nome_cen,
                cfg_cen['descricao'], cfg_cen['hipotese'],
                kpis_base_dict, resultado_para_llm
            )

            log(f'\n  💬 Solicitando interpretação ao LLM ({LLM_PROVIDER})...', log_f)
            interpretacao = chamar_llm(prompt)
            log(f'\n  📝 Interpretação:\n', log_f)
            for linha_interp in interpretacao.split('\n'):
                log(f'     {linha_interp}', log_f)

            todos_resultados[setor]['cenarios'][nome_cen] = {
                'predicoes'     : predicoes,
                'variacoes'     : variacoes,
                'interpretacao' : interpretacao,
            }

    # ── Salva CSV ─────────────────────────────────────────────────────────────
    if linhas_csv:
        pd.DataFrame(linhas_csv).to_csv(
            os.path.join(PASTA_RESULTADOS, 'cenarios_resultados.csv'),
            index=False, sep=';', encoding='utf-8-sig'
        )
        log(f'\n  💾 cenarios_resultados.csv salvo.', log_f)

    return todos_resultados


# =============================================================================
# GRÁFICOS DE CENÁRIOS
# =============================================================================

def gerar_graficos_cenarios(todos_resultados, log_f):
    """
    Para cada empresa âncora, gera um gráfico de barras agrupadas mostrando
    o impacto percentual de cada cenário nos três targets.
    Facilita a comparação visual entre cenários e entre empresas.
    """
    log(f'\n{S}\n📈 GRÁFICOS DE CENÁRIOS\n{S}', log_f)

    for setor, dados in todos_resultados.items():
        nome_emp = dados['empresa']
        cenarios = list(dados['cenarios'].keys())

        if not cenarios:
            continue

        # Coleta variações percentuais
        tgt_nomes = list(TARGETS.values())
        n_targets = len(tgt_nomes)
        n_cenarios = len(cenarios)

        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5), squeeze=False)

        cores_cenarios = ['#1F3864', '#1A5E35', '#C65911']

        for t_idx, (tgt_col, tgt_nome) in enumerate(TARGETS.items()):
            ax = axes[0][t_idx]
            x  = np.arange(n_cenarios)
            vals = [
                dados['cenarios'][c]['variacoes'].get(tgt_col, np.nan)
                for c in cenarios
            ]

            bars = ax.bar(
                x, vals,
                color=[cores_cenarios[i % len(cores_cenarios)] for i in range(n_cenarios)],
                edgecolor='white', linewidth=0.8, width=0.6
            )

            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ypos = bar.get_height() + (0.2 if v >= 0 else -0.8)
                    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                            fmt_var(v), ha='center', va='bottom',
                            fontsize=9, fontweight='bold')

            ax.axhline(y=0, color='black', linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [c.replace(' (CAPEX)', '\n(CAPEX)').replace(' com ', '\ncom ')
                 for c in cenarios],
                fontsize=8
            )
            ax.set_ylabel('Variação vs. Baseline (%)', fontsize=9)
            ax.set_title(tgt_nome.replace(' (t+1)', ''), fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)

        plt.suptitle(
            f'Impacto dos Cenários Estratégicos — {nome_emp} ({setor})',
            fontsize=12, y=1.02
        )
        plt.tight_layout()

        nome_arq = f'cenario_{nome_emp.replace(" ", "_").lower()[:20]}_{setor}.png'
        plt.savefig(os.path.join(PASTA_GRAFICOS, nome_arq), dpi=150, bbox_inches='tight')
        plt.close()
        log(f'  💾 {nome_arq}', log_f)


# =============================================================================
# RELATÓRIOS (JSON + TEXTUAL)
# =============================================================================

def gerar_relatorios(todos_resultados, log_f):
    """
    Gera dois relatórios:

    1. cenarios_resultados.json
       Estrutura hierárquica completa: setor → empresa → cenário → métricas +
       interpretação LLM. Referência programática para uso futuro.

    2. relatorio_cenarios.txt
       Narrativa organizada por empresa e cenário, com os números e as
       interpretações do LLM. Pronto para ser adaptado na seção de
       Resultados (Pergunta C) e na Conclusão do TCC.
    """
    log(f'\n{S}\n📄 GERAÇÃO DE RELATÓRIOS DE CENÁRIOS\n{S}', log_f)

    # ── JSON ──────────────────────────────────────────────────────────────────
    # Converte numpy/float para tipos serializáveis
    def _serializar(obj):
        if isinstance(obj, (np.floating, float)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _serializar(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serializar(i) for i in obj]
        return obj

    arq_json = os.path.join(PASTA_RESULTADOS, 'cenarios_resultados.json')
    with open(arq_json, 'w', encoding='utf-8') as f:
        json.dump(_serializar(todos_resultados), f, ensure_ascii=False, indent=2)
    log(f'  💾 cenarios_resultados.json salvo.', log_f)

    # ── Relatório textual ─────────────────────────────────────────────────────
    arq_txt = os.path.join(PASTA_RESULTADOS, 'relatorio_cenarios.txt')
    with open(arq_txt, 'w', encoding='utf-8') as f:

        f.write(S + '\n')
        f.write('RELATÓRIO DE CENÁRIOS ESTRATÉGICOS — INSUMOS PARA O TCC\n')
        f.write('Seção: Resultados (Pergunta C) e Conclusão\n')
        f.write(S + '\n\n')

        f.write('PERGUNTA C:\n')
        f.write('Como decisões financeiras estratégicas impactam os indicadores\n')
        f.write('financeiros futuros de empresas brasileiras de capital aberto,\n')
        f.write('e como modelos de ML combinados com IA Generativa podem simular\n')
        f.write('esses cenários para apoiar a tomada de decisão gerencial?\n\n')
        f.write(s2 + '\n\n')

        for setor, dados in todos_resultados.items():
            nome_emp = dados['empresa']
            f.write(f'EMPRESA: {nome_emp}  |  SETOR: {setor}\n')
            f.write(f'Período base: {dados["periodo_base"]}\n')
            f.write(s2 + '\n\n')

            # Tabela de baseline
            f.write('  Indicadores previstos (sem cenário — baseline):\n')
            for tgt_col, tgt_nome in TARGETS.items():
                val = dados['baseline'].get(tgt_col, np.nan)
                f.write(f'    {tgt_nome:<32} {fmt_bi(val)}\n')
            f.write('\n')

            # Cada cenário
            for nome_cen, cen_dados in dados['cenarios'].items():
                desc_cen = CENARIOS[nome_cen]['descricao']
                hip_cen  = CENARIOS[nome_cen]['hipotese']

                f.write(f'  [{nome_cen}]\n')
                f.write(f'  Descrição : {desc_cen}\n')
                f.write(f'  Hipótese  : {hip_cen}\n\n')

                f.write(f'  {"Target":<32} {"Baseline":>14} {"Cenário":>14} '
                        f'{"Variação":>10}\n')
                f.write(f'  {"─"*32} {"─"*14} {"─"*14} {"─"*10}\n')

                for tgt_col, tgt_nome in TARGETS.items():
                    vb  = dados['baseline'].get(tgt_col, np.nan)
                    vc  = cen_dados['predicoes'].get(tgt_col, np.nan)
                    var = cen_dados['variacoes'].get(tgt_col, np.nan)
                    f.write(f'  {tgt_nome:<32} {fmt_bi(vb):>14} '
                            f'{fmt_bi(vc):>14} {fmt_var(var):>10}\n')

                f.write(f'\n  Análise do LLM ({LLM_PROVIDER.upper()}):\n')
                for linha_llm in cen_dados['interpretacao'].split('\n'):
                    f.write(f'  {linha_llm}\n')
                f.write('\n' + s2 + '\n\n')

        f.write('\n' + S + '\n')
        f.write('CONSIDERAÇÕES METODOLÓGICAS SOBRE OS CENÁRIOS\n')
        f.write(S + '\n\n')
        consideracoes = [
            ('Ajustes lineares',
             'Os cenários aplicam variações nos KPIs de forma linear e isolada. '
             'Na prática, decisões financeiras geram efeitos de segunda ordem '
             '(reação do mercado, renegociação de contratos, mudanças de rating) '
             'não capturados por esse modelo.'),
            ('Período de horizonte',
             'O modelo prediz t+1 (próximo exercício anual). Decisões de CAPEX '
             'geralmente levam 2-3 anos para impactar receita de forma plena. '
             'A magnitude do impacto em t+1 pode subestimar o efeito total.'),
            ('Variáveis exógenas',
             'O modelo não incorpora variáveis macroeconômicas (SELIC, câmbio, '
             'preço de commodities). Setores como Petróleo e Commodities têm '
             'resultados altamente correlatos com preços internacionais.'),
            ('Interpretação do LLM',
             f'As análises geradas pelo {LLM_PROVIDER.upper()} são baseadas no '
             'contexto fornecido pelo modelo e no conhecimento prévio do modelo '
             'sobre os setores. Devem ser tratadas como suporte analítico, não '
             'como recomendações financeiras formais.'),
        ]
        for titulo, texto in consideracoes:
            f.write(f'{titulo}:\n  {texto}\n\n')

    log(f'  💾 relatorio_cenarios.txt salvo.', log_f)


# =============================================================================
# EXECUÇÃO
# =============================================================================

def executar_cenarios():
    garantir_pastas()
    arq_log = os.path.join(PASTA_LOGS, 'log_cenarios.txt')

    with open(arq_log, 'w', encoding='utf-8') as log_f:
        log_f.write('LOG DE CENÁRIOS — CVM\n')
        log_f.write('TCC: Predição de Indicadores Financeiros com ML\n')
        log_f.write(f'Provedor LLM: {LLM_PROVIDER.upper()}\n')
        log_f.write(f'{S}\n\n')

        avaliacao, treino, teste, modelos = carregar_tudo(log_f)
        todos_resultados = analisar_cenarios(avaliacao, treino, teste, modelos, log_f)
        gerar_graficos_cenarios(todos_resultados, log_f)
        gerar_relatorios(todos_resultados, log_f)

        log(f'\n{S}', log_f)
        log(f'✅ CENÁRIOS CONCLUÍDOS', log_f)
        log(f'   Resultados em: ./{PASTA_RESULTADOS}/', log_f)
        log(f'   Relatório TCC: {PASTA_RESULTADOS}/relatorio_cenarios.txt', log_f)
        log(f'   JSON completo: {PASTA_RESULTADOS}/cenarios_resultados.json', log_f)
        log(S, log_f)

    print(f'\n📄 Log salvo em: {arq_log}')
    return todos_resultados


if __name__ == '__main__':
    executar_cenarios()