# %%
# script de decomposição de serie temporal

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Carrega a série padronizada
df = pd.read_csv("C:/Users/RODRIGUES/Desktop/ANALISE_EXPLORATORIO/PIPELINE/Pipeline_de_Tendências_Temporais/dados_padronizados1.csv")
df['data_producao'] = pd.to_datetime(df['data_producao'])
serie = df.set_index('data_producao')['volume_m3'].asfreq('D').fillna(0)

# Decomposição aditiva
decomp = seasonal_decompose(serie, model='additive', period=30)

# Visualização
decomp.plot()
plt.suptitle("Decomposição da Série Temporal - Volume Produzido")
plt.tight_layout()
plt.show()


# %%
# script de decomposição de serie temporal

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Carrega a série padronizada
df = pd.read_csv("C:/Users/RODRIGUES/Desktop/ANALISE_EXPLORATORIO/PIPELINE/Pipeline_de_Tendências_Temporais/dados_padronizados1.csv")
df['data_producao'] = pd.to_datetime(df['data_producao'])
serie = df.set_index('data_producao')['volume_m3'].asfreq('D').fillna(0)

# Decomposição aditiva
decomp = seasonal_decompose(serie, model='additive', period=30)

# Visualização
decomp.plot()
plt.suptitle("Decomposição da Série Temporal - Volume Produzido")
plt.tight_layout()
plt.show()


# %%
# Garante que o diretório exista
import os
os.makedirs("C:/Users/RODRIGUES/Desktop/ANALISE_EXPLORATORIO/PIPELINE/Pipeline_de_Tendencias_Temporais/compo", exist_ok=True)

# Exporta os componentes
decomp.trend.to_csv("C:/Users/RODRIGUES/Desktop/ANALISE_EXPLORATORIO/PIPELINE/Pipeline_de_Tendencias_Temporais/trend.csv")
decomp.seasonal.to_csv("C:/Users/RODRIGUES/Desktop/ANALISE_EXPLORATORIO/PIPELINE/Pipeline_de_Tendencias_Temporais/seasonal.csv")
decomp.resid.to_csv("C:/Users/RODRIGUES/Desktop/ANALISE_EXPLORATORIO/PIPELINE/Pipeline_de_Tendencias_Temporais/resid.csv")



# %%
# SCRIPT DE DETENÇÃO DE ANOMALIA NOS RESIDIUS COM ISOLATION FOREST

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# 1. Carrega a série padronizada
df = pd.read_csv("C:/Users/RODRIGUES/Desktop/ANALISE_EXPLORATORIO/PIPELINE/Pipeline_de_Tendências_Temporais/dados_padronizados1.csv")
df['data_producao'] = pd.to_datetime(df['data_producao'])
serie = df.set_index('data_producao')['volume_m3'].asfreq('D').fillna(0)

# 2. Decomposição aditiva
decomp = seasonal_decompose(serie, model='additive', period=30)
residuos = decomp.resid.fillna(0)

# 3. Prepara os dados para o modelo
X_resid = residuos.values.reshape(-1, 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resid)

# 4. Define e treina o modelo Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    max_samples='auto',
    random_state=42
)
iso_forest.fit(X_scaled)

# 5. Predição
y_pred = iso_forest.predict(X_scaled)
scores = iso_forest.decision_function(X_scaled)

# 6. Resultado em DataFrame
df_anomalias = pd.DataFrame({
    'data_producao': residuos.index,
    'residuo': residuos.values,
    'anomaly': y_pred,
    'score': scores
}).set_index('data_producao')

# 7. Visualização das anomalias
plt.figure(figsize=(14, 6))
plt.plot(df_anomalias.index, df_anomalias['residuo'], label='Resíduo', color='blue')
plt.scatter(df_anomalias[df_anomalias['anomaly'] == -1].index,
            df_anomalias[df_anomalias['anomaly'] == -1]['residuo'],
            color='red', label='Anomalias')
plt.title('Detecção de Anomalias nos Resíduos - Isolation Forest')
plt.xlabel('Data de Produção')
plt.ylabel('Resíduo da Série Temporal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import pandas as pd

# Verifica se df_anomalias existe
if 'df_anomalias' in globals():
    # Cria colunas auxiliares
    df_anomalias = df_anomalias.copy()
    df_anomalias['dia'] = df_anomalias.index.date
    df_anomalias['mes'] = df_anomalias.index.to_period('M').astype(str)

    # Indicadores
    total_anomalias = (df_anomalias['anomaly'] == -1).sum()
    dias_com_anomalia = df_anomalias[df_anomalias['anomaly'] == -1]['dia'].nunique()
    total_dias = df_anomalias['dia'].nunique()
    percentual_dias_anomalia = round((dias_com_anomalia / total_dias) * 100, 2)
    score_medio_anomalias = round(df_anomalias[df_anomalias['anomaly'] == -1]['score'].mean(), 4)

    distribuicao_mensal = (
        df_anomalias[df_anomalias['anomaly'] == -1]
        .groupby('mes')
        .size()
        .reset_index(name='quantidade_anomalias')
    )

    resumo_indicadores = pd.DataFrame({
        'Indicador': [
            'Total de Anomalias',
            '% de Anomalias por dia',
            'Score Médio das Anomalias'
        ],
        'Valor': [
            total_anomalias,
            percentual_dias_anomalia,
            score_medio_anomalias
        ]
    })

    # Caminho de exportação
    caminho_base = "C:/Users/RODRIGUES/Desktop/ANALISE_EXPLORATORIO/PIPELINE/Pipeline_de_Tendências_Temporais/INDICADORES DE ANOMALIA/"

    # Exporta os arquivos
    df_anomalias.to_csv(f"{caminho_base}anomalias_diarias.csv")
    distribuicao_mensal.to_csv(f"{caminho_base}anomalias_mensais.csv", index=False)
    resumo_indicadores.to_csv(f"{caminho_base}indicadores_resumo.csv", index=False)

    print("✅ Arquivos CSV exportados com sucesso para o diretório do Power BI!")

else:
    print("❌ Erro: df_anomalias não está definido. Execute primeiro o script de detecção de anomalias.")


# %%
# ============================================================
# Integração com Power BI - Pipeline de Anomalias Temporais
# Autor: Elias Rodrigues Umbar Zimbeti
# Versão: 2.0
# Data: 2025-12-11
# ============================================================

import pandas as pd
import logging
import os

# 🔹 Configuração de Logs para rastreabilidade
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 🔹 Caminho do arquivo CSV exportado
caminho_csv = r"C:/Users/RODRIGUES/Desktop/ANALISE_EXPLORATORIO/PIPELINE/Pipeline_de_Tendências_Temporais/INDICADORES DE ANOMALIA/anomalias_diarias.csv"

# 🔹 Verifica se o arquivo existe
if not os.path.exists(caminho_csv):
    logging.error(f"Arquivo não encontrado: {caminho_csv}")
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv}")

# 🔹 Carrega o CSV e transforma 'data_producao' em datetime
df_anomalias = pd.read_csv(caminho_csv, parse_dates=['data_producao'])

# 🔹 Define 'data_producao' como índice
df_anomalias.set_index('data_producao', inplace=True)

# 🔹 Cria colunas auxiliares
df_anomalias['dia'] = df_anomalias.index.date
df_anomalias['mes'] = df_anomalias.index.to_period('M').astype(str)

# 🔹 Normaliza o score para [0,1]
df_anomalias['score_normalizado'] = (
    (df_anomalias['score'] - df_anomalias['score'].min()) /
    (df_anomalias['score'].max() - df_anomalias['score'].min())
)

# ============================================================
# Indicadores principais
# ============================================================

total_anomalias = (df_anomalias['anomaly'] == -1).sum()
dias_com_anomalia = df_anomalias[df_anomalias['anomaly'] == -1]['dia'].nunique()
total_dias = df_anomalias['dia'].nunique()
percentual_dias_anomalia = round((dias_com_anomalia / total_dias) * 100, 2)
score_medio_anomalias = round(df_anomalias[df_anomalias['anomaly'] == -1]['score'].mean(), 4)

# 🔹 Logs dos indicadores
logging.info(f"Total de anomalias: {total_anomalias}")
logging.info(f"% de dias com anomalias: {percentual_dias_anomalia}")
logging.info(f"Score médio das anomalias: {score_medio_anomalias}")

# 🔹 Distribuição mensal
distribuicao_mensal = (
    df_anomalias[df_anomalias['anomaly'] == -1]
    .groupby('mes')
    .size()
    .reset_index(name='quantidade_anomalias')
)

# 🔹 Resumo dos indicadores
resumo_indicadores = pd.DataFrame({
    'Indicador': [
        'Total de Anomalias',
        '% de Dias com Anomalias',
        'Score Médio das Anomalias'
    ],
    'Valor': [
        total_anomalias,
        percentual_dias_anomalia,
        score_medio_anomalias
    ]
})

# ============================================================
# Exportação para CSV (para integração com Power BI)
# ============================================================

resumo_indicadores.to_csv("resumo_indicadores.csv", index=False)
distribuicao_mensal.to_csv("distribuicao_mensal.csv", index=False)
df_anomalias.to_csv("anomalias_processadas.csv", index=False)

# 🔹 Escolha qual DataFrame importar no Power BI
dataset = resumo_indicadores  # ou df_anomalias ou distribuicao_mensal



