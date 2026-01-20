**pos_fiap_ativ01**

Repositório dedicado à entrega da Atividade 01 do curso de Pós-Graduação da FIAP (IA para DEVs).

**Sistema de Diagnóstico de Câncer de Mama**

Sistema de Machine Learning para classificação de tumores mamários como benignos ou malignos.


**1. Instruções de Execução**

**1.1. Pré-requisitos**

Antes de executar o projeto, você precisa ter instalado:

• Python 3.8

• pip (gerenciador de pacotes do Python)

**1.2. Instalação**

Instalar as bibliotecas necessária, abra o terminal e execute:

• pip install pandas numpy matplotlib seaborn scikit-learn



**1.3. Estrutura de Arquivos**

Organize os arquivos desta forma:

pasta/
│
├── Atv01/
│   └── data.csv          <- Arquivo de dados (OBRIGATÓRIO)
│
└── main.py               <- Código Python


**1.4. Como Executar**

IDE (VSCode, PyCharm, etc)

• Abra o arquivo main.py na sua IDE

• Clique em "Run" ou pressione F5





**2. Dataset (ou link para download)**

• O dataset esta presente no repositório: Atv01/data.csv

• Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download





**3. Resultados obtidos (prints, gráficos e análises)**

• A análise e os gráficos gerados a partir do prompt estão presentes em Atv01/resultado_prompt 

• Os arquivos se chamam analise_prompt.txt e imagens_prompt.pdf





**4. Relatório técnico explicando**

**4.1. Estratégias de Pré-processamento**

**4.1.1. Exploração e Qualidade dos Dados**

O dataset analisado possui 569 amostras e 33 colunas, sendo a variável alvo diagnosis, que indica se o tumor é Benigno (B) ou Maligno (M). Durante a análiseinicial, foram observados os seguintes pontos:

• Não há valores ausentes nas variáveis relevantes do dataset.

• A coluna Unnamed: 32 contém apenas valores nulos.

• A coluna id representa apenas um identificador, sem valor preditivo.

• Não foram identificadas linhas duplicadas.

Essas análises garantiram que o conjunto de dados possui boa qualidade para modelagem.



**4.1.2. Limpeza dos Dados**

Com base na exploração inicial, foram realizadas as seguintes ações:

• Remoção de colunas irrelevantes: id e Unnamed: 32 foram excluídas por não contribuírem para o aprendizado dos modelos.

• O dataset final passou a conter 30 variáveis explicativas e 1 variável alvo.



**4.1.3. Tratamento da Variável Alvo**

A variável diagnosis é categórica e foi convertida para formato numérico utilizando Label Encoding:

• Benigno (B) → 0

• Maligno (M) → 1

A distribuição das classes é relativamente balanceada:

• Benigno: ~62,7%

• Maligno: ~37,3%

Por esse motivo, não foi necessário aplicar técnicas de balanceamento, como oversampling ou undersampling.



**4.1.4. Análise Estatística e Correlação**

Foram analisadas as distribuições das principais features, especialmente aquelas relacionadas a medidas geométricas do tumor (radius, perimeter, area). Observou se que:

• Tumores malignos apresentam, em média, valores mais elevados nessas características.

• Há forte correlação positiva entre algumas variáveis e o diagnóstico, principalmente:

  • concave points_worst

  • perimeter_worst
  
  • radius_worst
  
  • area_worst

Essa análise reforça a relevância clínica dessas variáveis.



**4.1.5. Normalização e Divisão dos Dados**



Os dados foram divididos em:

• 70% Treino

• 15% Validação

• 15% Teste

A divisão foi realizada de forma estratificada, preservando a proporção das classes.



Em seguida, foi aplicado o StandardScaler, garantindo que todas as features numéricas estivessem na mesma escala. Essa etapa é fundamental principalmente para modelos sensíveis à magnitude das variáveis, como o SVM.



**4.2. Modelos Utilizados e Justificativa**

**4.2.1. Support Vector Machine (SVM)**

O primeiro modelo avaliado foi o SVM com kernel RBF, escolhido pelos seguintes motivos:

• Excelente desempenho em problemas de classificação binária.

• Capacidade de lidar com fronteiras de decisão não lineares.

• Robustez em datasets de média dimensão, como este.

O uso do StandardScaler é essencial para o bom desempenho do SVM, garantindo que nenhuma variável domine o processo de otimização.



**4.2.2. Random Forest**

O segundo modelo foi o Random Forest, um ensemble de árvores de decisão, escolhido por:

• Ser robusto a outliers e ruídos.

• Capturar relações não lineares entre as variáveis.

• Fornecer importância das features, o que aumenta a interpretabilidade do modelo.

Além disso, o Random Forest tende a apresentar ótimo desempenho sem necessidade intensa de ajuste fino de hiperparâmetros.



**4.3. Resultados e Interpretação dos Dados**

**4.3.1 Desempenho dos Modelos**



Os modelos foram avaliados utilizando as métricas:

• Accuracy

• Precision

• Recall

• F1-Score

• ROC-AUC

Resultados no Conjunto de Teste:

• SVM:

  • Accuracy: ~96,5%
  
  • Recall (Maligno): ~90,6%
  
  • ROC-AUC: 1,00

• Random Forest:

  • Accuracy: ~97,7%
  
  • Recall (Maligno): ~93,8%
  
  • ROC-AUC: ~0,998

Ambos os modelos apresentaram desempenho elevado, porém o Random Forest se destacou ligeiramente, principalmente no recall da classe maligna, métrica crítica em aplicações médicas, onde minimizar falsos negativos é essencial.


**4.3.2 Matrizes de Confusão e Curvas ROC**

• As matrizes de confusão mostram baixo número de erros de classificação.

• As curvas ROC apresentaram áreas próximas de 1, indicando excelente capacidade de separação entre tumores benignos e malignos.



**4.3.3 Importância das Features**

A análise de importância das variáveis, obtida pelo Random Forest, mostrou que as features mais relevantes estão relacionadas a:

• Área do tumor (area_worst, area_mean)

• Pontos côncavos (concave points_mean, concave points_worst)

• Medidas de raio e perímetro (radius_worst, perimeter_worst)

Esses resultados são consistentes com o conhecimento médico, indicando que tumores malignos tendem a apresentar formas mais irregulares e maiores dimensões.
