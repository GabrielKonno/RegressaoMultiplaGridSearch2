# Modelo de Previsão de Temperatura

## Visão Geral
Este projeto implementa um modelo de machine learning para prever temperaturas utilizando o algoritmo Gradient Boosting Regressor. O modelo é otimizado através de GridSearchCV, uma técnica que busca automaticamente os melhores hiperparâmetros para maximizar o desempenho das previsões.

## Detalhamento do Funcionamento

### 1. Carregamento e Preparação dos Dados
```python
features = pd.read_excel("temps.xlsx")
features = pd.get_dummies(features)
```
- Carrega dados de temperatura do arquivo Excel
- Converte variáveis categóricas em numéricas usando one-hot encoding

### 2. Separação de Features e Target
```python
labels = np.array(features['actual'])
features = features.drop('actual', axis=1)
```
- 'actual': temperatura real (variável alvo)
- Features: demais variáveis que serão usadas para previsão

### 3. Divisão em Dados de Treino e Teste
```python
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42
)
```
- 75% dos dados para treino
- 25% dos dados para teste
- Random state fixo para reprodutibilidade

### 4. Otimização de Hiperparâmetros
```python
parameters = {
    'learning_rate': [0.03],
    'subsample': [0.2],
    'n_estimators': [100, 500, 1000, 1500],
    'max_depth': [8]
}

grid_search = GridSearchCV(GradientBoostingRegressor(), 
                         parameters, 
                         scoring='r2', 
                         cv=2, 
                         n_jobs=-1)
```
Parâmetros testados:
- Learning rate: taxa de aprendizado do modelo (fixada em 0.03)
- Subsample: fração de amostras usadas (fixada em 0.2)
- N_estimators: número de árvores de decisão (testando diferentes valores)
- Max_depth: profundidade máxima das árvores (fixada em 8)

### 5. Avaliação do Modelo
O modelo é avaliado usando várias métricas:
- R² (Coeficiente de Determinação): indica quanto da variância dos dados é explicada pelo modelo
- MAE (Erro Médio Absoluto): média das diferenças absolutas entre previsões e valores reais
- MSE (Erro Quadrático Médio): média dos quadrados dos erros
- RMSE (Raiz do Erro Quadrático Médio): raiz quadrada do MSE

## Configuração dos Hiperparâmetros Finais
```python
grb_tunned = GradientBoostingRegressor(
    learning_rate=0.03,
    max_depth=8,
    n_estimators=100,
    subsample=0.2
)
```
Estes são os parâmetros otimizados encontrados pelo GridSearch que proporcionaram o melhor desempenho do modelo.

## Requisitos e Instalação
1. Python 3.x
2. Bibliotecas necessárias:
```bash
pip install pandas numpy scikit-learn
```

## Como Executar
1. Clone o repositório
2. Certifique-se que o arquivo 'temps.xlsx' está no diretório
3. Execute:
```bash
python grid_search_aula.py
```

## Estrutura do Projeto
- `grid_search_aula.py`: Script principal com a implementação do modelo
- `temps.xlsx`: Dataset contendo dados históricos de temperatura

## Possíveis Melhorias
1. Engenharia de Features:
   - Criação de novas variáveis
   - Transformações nas features existentes

2. Otimização do Modelo:
   - Testar mais valores de hiperparâmetros
   - Aumentar o número de folds na validação cruzada
   - Experimentar diferentes algoritmos de machine learning

3. Análise de Dados:
   - Implementar visualizações
   - Análise de importância das features
   - Tratamento de outliers

4. Produtização:
   - Criar API para o modelo
   - Implementar pipeline de dados
   - Adicionar logs e monitoramento
