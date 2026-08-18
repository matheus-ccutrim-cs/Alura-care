# Alura Care — Machine Learning para Classificação de Exames

Projeto desenvolvido para explorar técnicas de **Machine Learning, análise de dados e seleção de atributos** aplicadas a um conjunto de dados de exames.

O objetivo é construir e avaliar modelos capazes de classificar diagnósticos a partir das características presentes nos exames, investigando ao mesmo tempo quais variáveis são mais relevantes para a classificação.

> **Nota:** este é um projeto educacional de Ciência de Dados e Machine Learning e não deve ser utilizado para diagnóstico médico real.

## Sobre o projeto

Bases de dados com muitas variáveis podem aumentar a complexidade de um modelo sem necessariamente melhorar sua capacidade de previsão.

Neste projeto, são exploradas diferentes estratégias para analisar e reduzir a dimensionalidade dos dados, mantendo as informações mais relevantes para a classificação.

O notebook percorre etapas como:

* análise inicial dos dados;
* identificação de valores ausentes;
* criação de um modelo de classificação;
* comparação com um modelo baseline;
* padronização dos dados;
* análise visual das variáveis;
* identificação de variáveis altamente correlacionadas;
* seleção automática de atributos;
* avaliação através de matriz de confusão;
* redução de dimensionalidade;
* visualização dos dados em duas dimensões.

---

## Técnicas utilizadas

### Random Forest

O principal algoritmo de classificação utilizado no projeto é o `RandomForestClassifier`.

O primeiro modelo, utilizando os atributos disponíveis após o tratamento inicial dos dados, alcançou aproximadamente:

**92,40% de acurácia**

Também foi utilizado um `DummyClassifier` como baseline para comparação com o modelo treinado.

---

### Padronização dos dados

Como os exames possuem escalas bastante diferentes, foi utilizado:

```python
StandardScaler
```

A padronização facilita a comparação entre as variáveis e melhora principalmente as etapas de análise e visualização.

---

### Análise de correlação

Uma matriz de correlação foi construída para identificar exames que carregavam informações muito semelhantes.

Foram encontradas correlações superiores a **0,99** entre algumas variáveis.

A análise permitiu identificar atributos redundantes e testar a classificação após sua remoção.

---

## Seleção de atributos

Um dos principais objetivos do projeto é verificar se é possível utilizar uma quantidade menor de exames sem comprometer significativamente a classificação.

Foram utilizadas diferentes estratégias.

### SelectKBest

Foi utilizado o `SelectKBest` com o teste estatístico `chi2` para selecionar **5 atributos**.

```python
SelectKBest(chi2, k=5)
```

Após a seleção, o conjunto de teste passou a possuir apenas cinco características por amostra.

A matriz de confusão obtida foi:

```text
[[100,  5],
 [  8, 58]]
```

Isso mostra que é possível realizar uma classificação competitiva utilizando apenas uma pequena parcela das características disponíveis.

### RFE — Recursive Feature Elimination

Também foi utilizado o **Recursive Feature Elimination (RFE)**.

O método elimina atributos progressivamente de acordo com a importância estimada pelo modelo, mantendo neste experimento:

```python
n_features_to_select = 5
```

A seleção foi realizada utilizando um `RandomForestClassifier` como estimador.

### RFECV

Para automatizar a escolha da quantidade de atributos, o projeto utiliza também:

```python
RFECV
```

Nesse caso, a seleção é combinada com **validação cruzada de 5 folds**, utilizando acurácia como métrica:

```python
RFECV(
    estimator=classificador,
    cv=5,
    step=1,
    scoring="accuracy"
)
```

O experimento com RFECV apresentou aproximadamente **93% de acurácia** no conjunto de teste.

Também foi analisada graficamente a relação entre:

**Número de exames × Acurácia**

permitindo observar o impacto da quantidade de atributos no desempenho do modelo.

---

## 📈 Visualização dos dados

O projeto utiliza diferentes técnicas de visualização para entender melhor o comportamento das variáveis.

### Violin plots

Gráficos de violino são utilizados para comparar a distribuição dos exames entre as classes de diagnóstico.

Os dados são padronizados antes de algumas dessas visualizações para facilitar a comparação entre atributos com escalas diferentes.

### Heatmap de correlação

Uma matriz de correlação é apresentada através de um heatmap para identificar relações fortes entre os exames.

### PCA

O **Principal Component Analysis (PCA)** é utilizado para reduzir o conjunto de dados para duas dimensões.

```python
PCA(n_components=2)
```

Isso permite representar visualmente as observações em um plano 2D.

### t-SNE

Também é utilizado o **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:

```python
TSNE(n_components=2)
```

A técnica permite explorar visualmente como as diferentes classes se distribuem após a redução de dimensionalidade.

---

## Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Estrutura do projeto

```text
Alura-Care/
│
├── Alura_care.ipynb
├── exames.csv
└── README.md
```

O notebook está organizado em cinco etapas principais:

```text
Aula 01 — Dados com muitas dimensões
Aula 02 — Avançando e explorando dados
Aula 03 — Dados correlacionados
Aula 04 — Automatizando a seleção
Aula 05 — Visualizando os dados no plano
```

---

## Como executar

### 1. Clone o repositório

```bash
git clone <URL-DO-REPOSITORIO>
cd Alura-Care
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 4. Execute o Jupyter Notebook

```bash
jupyter notebook
```

Abra o arquivo:

```text
Alura_care.ipynb
```

O arquivo `exames.csv` deve estar disponível no mesmo diretório do notebook.

---

## Principais aprendizados

Durante o desenvolvimento deste projeto foram explorados conceitos importantes de Machine Learning e Ciência de Dados, incluindo:

* preparação e exploração de dados;
* tratamento de valores ausentes;
* divisão entre treino e teste;
* criação de modelos baseline;
* classificação com Random Forest;
* padronização de atributos;
* análise de correlação;
* feature selection;
* SelectKBest;
* Recursive Feature Elimination;
* validação cruzada com RFECV;
* matriz de confusão;
* PCA;
* t-SNE;
* visualização de dados multidimensionais.

---

## Conclusão

Os experimentos mostram que **mais atributos não significam necessariamente um modelo melhor**.

Ao analisar correlações e aplicar técnicas de seleção de atributos, é possível reduzir a quantidade de variáveis utilizadas pelo modelo e ainda preservar um bom desempenho de classificação.

O projeto demonstra, na prática, a importância da **seleção de features e da redução de dimensionalidade** na construção de modelos de Machine Learning mais simples, interpretáveis e eficientes.
