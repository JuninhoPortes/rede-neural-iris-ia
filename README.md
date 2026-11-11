# Rede Neural Artificial — Iris

Atividade da disciplina de Inteligência Artificial da UNIPAC Barbacena.

## Objetivo

Implementar do zero, em Python, uma rede neural artificial com uma camada escondida para classificar as três espécies do dataset Iris.

## Tecnologias e restrições

- Python e NumPy para os cálculos da rede neural;
- Scikit-Learn exclusivamente para carregar e preparar os dados (`load_iris`, `train_test_split` e `StandardScaler`);
- Nenhum modelo ou otimizador pronto de redes neurais é utilizado.

## Dataset e pré-processamento

- Iris: 150 amostras, 4 características, 3 classes;
- Treinamento: 120 amostras (80%); teste: 30 amostras (20%);
- Divisão estratificada com `random_state=42`;
- Padronização para média 0 e desvio padrão 1, ajustada apenas sobre o conjunto de treinamento;
- Codificação dos rótulos no formato One-Hot.

## Arquitetura e treinamento

- Entrada: 4 características;
- Camada escondida: 8 neurônios com ReLU;
- Saída: 3 neurônios com Softmax;
- Pesos aleatórios, vieses aleatórios e semente 42;
- Inicialização He na primeira camada;
- Forward Pass, Cross-Entropy Loss e Backpropagation implementados com NumPy;
- Gradiente Descendente em lote completo por 1.000 épocas, com taxa de aprendizado 0,05.

## Como executar (Windows / PowerShell)

Na raiz do projeto:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## Resultados

O terminal informa a evolução da perda ao longo das épocas e, após o treinamento, executa a inferência no conjunto de teste, exibindo a quantidade de acertos e a acurácia percentual. A acurácia é calculada diretamente com NumPy, sem usar `sklearn.metrics`.

## Estrutura do projeto

- `main.py`: carregamento, preparação dos dados, arquitetura da rede neural, treinamento, avaliação e verificações automáticas;
- `requirements.txt`: bibliotecas e versões utilizadas;
- `.gitignore`: definição dos arquivos e diretórios ignorados pelo Git;
- `README.md`: documentação, instruções de execução e descrição do projeto.
