import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# ETAPA 2 - PRE-PROCESSAMENTO DOS DADOS
# ============================================================

def preparar_dados_iris():
    # Carregamento do dataset Iris
    iris = load_iris()

    X = iris.data
    y = iris.target

    quantidade_classes = len(iris.target_names)

    # Divisao: 80% treinamento e 20% teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Padronizacao sem vazamento de dados
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)

    # Conversao dos rotulos para One-Hot Encoding
    y_treino_onehot = np.eye(quantidade_classes)[y_treino]
    y_teste_onehot = np.eye(quantidade_classes)[y_teste]

    return (
        X_treino,
        X_teste,
        y_treino,
        y_teste,
        y_treino_onehot,
        y_teste_onehot
    )


# ============================================================
# ETAPA 3 - ARQUITETURA DA REDE NEURAL
# ============================================================

class RedeNeuralIris:

    def __init__(self, entradas=4, ocultos=8, saidas=3, seed=42):
        # Gerador aleatorio para resultados reproduziveis
        rng = np.random.default_rng(seed)

        # Pesos da entrada para a camada escondida (He/ReLU)
        self.W1 = rng.normal(
            0,
            np.sqrt(2.0 / entradas),
            size=(entradas, ocultos)
        )

        self.b1 = rng.normal(0, 0.01, size=(1, ocultos))

        # Pesos da camada escondida para a camada de saida
        self.W2 = rng.normal(
            0,
            np.sqrt(1.0 / ocultos),
            size=(ocultos, saidas)
        )

        self.b2 = rng.normal(0, 0.01, size=(1, saidas))

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def derivada_relu(z):
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        # Estabilidade numerica: subtracao do maior logit
        z_estavel = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_estavel)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # Camada escondida: transformacao linear + ReLU
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        # Camada de saida: transformacao linear + Softmax
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    # ========================================================
    # ETAPA 4 - TREINAMENTO DA REDE NEURAL
    # ========================================================

    @staticmethod
    def entropia_cruzada(y_real, probabilidades):
        # Evita log(0)
        probabilidades_seguras = np.clip(probabilidades, 1e-12, 1.0)

        return -np.mean(
            np.sum(y_real * np.log(probabilidades_seguras), axis=1)
        )

    def backward(self, X, y_real):
        quantidade_amostras = X.shape[0]

        # Gradientes da saida: Softmax + Cross-Entropy
        delta_saida = (self.a2 - y_real) / quantidade_amostras

        dW2 = self.a1.T @ delta_saida
        db2 = np.sum(delta_saida, axis=0, keepdims=True)

        # Gradientes da camada escondida pela regra da cadeia
        delta_oculta = (
            delta_saida @ self.W2.T
        ) * self.derivada_relu(self.z1)

        dW1 = X.T @ delta_oculta
        db1 = np.sum(delta_oculta, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def treinar(self, X, y_real, epocas=1000, taxa_aprendizado=0.05):
        # Loss antes de qualquer atualizacao
        perda_inicial = self.entropia_cruzada(y_real, self.forward(X))

        for epoca in range(1, epocas + 1):
            # Forward Pass
            self.forward(X)

            # Backpropagation
            dW1, db1, dW2, db2 = self.backward(X, y_real)

            # Gradiente Descendente
            self.W1 -= taxa_aprendizado * dW1
            self.b1 -= taxa_aprendizado * db1
            self.W2 -= taxa_aprendizado * dW2
            self.b2 -= taxa_aprendizado * db2

            if epoca == 1 or epoca % 100 == 0:
                perda = self.entropia_cruzada(y_real, self.forward(X))
                print(f"Epoca {epoca:4d}/{epocas} | Loss: {perda:.6f}")

        perda_final = self.entropia_cruzada(y_real, self.forward(X))
        return perda_inicial, perda_final


# ============================================================
# EXECUCAO E VALIDACAO (ETAPAS 1 A 5)
# ============================================================

def main():
    print("=== REDE NEURAL ARTIFICIAL - IRIS ===")

    (
        X_treino,
        X_teste,
        y_treino,
        y_teste,
        y_treino_onehot,
        y_teste_onehot
    ) = preparar_dados_iris()

    print("\n1. PRE-PROCESSAMENTO")
    print(f"Amostras de treinamento: {X_treino.shape}")
    print(f"Amostras de teste: {X_teste.shape}")
    print(f"One-Hot treinamento: {y_treino_onehot.shape}")
    print(f"One-Hot teste: {y_teste_onehot.shape}")

    rede = RedeNeuralIris(entradas=4, ocultos=8, saidas=3, seed=42)

    print("\n2. ARQUITETURA DA REDE")
    print(f"Pesos W1: {rede.W1.shape}")
    print(f"Vieses b1: {rede.b1.shape}")
    print(f"Pesos W2: {rede.W2.shape}")
    print(f"Vieses b2: {rede.b2.shape}")

    probabilidades_iniciais = rede.forward(X_treino)

    print("\n3. FORWARD PASS (ANTES DO TREINAMENTO)")
    print(f"Entrada: {X_treino.shape}")
    print(f"Camada escondida: {rede.a1.shape}")
    print(f"Saida: {probabilidades_iniciais.shape}")
    print(f"Classe real: {y_treino[0]}")
    print("Probabilidades iniciais:", np.round(probabilidades_iniciais[0], 4))
    print("Soma das probabilidades:", np.sum(probabilidades_iniciais[0]))

    # Regressao das etapas anteriores
    assert X_treino.shape == (120, 4)
    assert X_teste.shape == (30, 4)
    assert y_treino_onehot.shape == (120, 3)
    assert y_teste_onehot.shape == (30, 3)
    assert rede.W1.shape == (4, 8)
    assert rede.b1.shape == (1, 8)
    assert rede.W2.shape == (8, 3)
    assert rede.b2.shape == (1, 3)
    assert probabilidades_iniciais.shape == (120, 3)
    assert rede.a1.shape == (120, 8)
    assert np.all(rede.a1 >= 0)
    assert np.all(np.isfinite(probabilidades_iniciais))
    assert np.all((probabilidades_iniciais >= 0) & (probabilidades_iniciais <= 1))
    assert np.allclose(np.sum(probabilidades_iniciais, axis=1), 1.0)
    assert np.array_equal(
        rede.derivada_relu(np.array([-2.0, 0.0, 3.0])),
        np.array([0.0, 0.0, 1.0])
    )
    print("Arquitetura e Forward Pass validados!")

    print("\n4. TREINAMENTO - GRADIENTE DESCENDENTE")
    W1_antes = rede.W1.copy()
    b1_antes = rede.b1.copy()
    W2_antes = rede.W2.copy()
    b2_antes = rede.b2.copy()

    perda_inicial, perda_final = rede.treinar(
        X_treino,
        y_treino_onehot,
        epocas=1000,
        taxa_aprendizado=0.05
    )

    print("\n5. VALIDACAO DO TREINAMENTO")
    print(f"Loss inicial: {perda_inicial:.6f}")
    print(f"Loss final:   {perda_final:.6f}")

    assert np.isfinite(perda_final) and perda_final < perda_inicial
    assert not np.array_equal(W1_antes, rede.W1)
    assert not np.array_equal(b1_antes, rede.b1)
    assert not np.array_equal(W2_antes, rede.W2)
    assert not np.array_equal(b2_antes, rede.b2)
    print("Treinamento validado com sucesso!")

    # ========================================================
    # ETAPA 5 - AVALIACAO NO CONJUNTO DE TESTE
    # ========================================================

    print("\n6. AVALIACAO DO MODELO NO CONJUNTO DE TESTE")

    # Faz inferencia nas 30 amostras nunca usadas para treinar
    probabilidades_teste = rede.forward(X_teste)

    # Seleciona a classe com maior probabilidade para cada flor
    classes_previstas = np.argmax(probabilidades_teste, axis=1)

    # Acuracia calculada manualmente, sem sklearn.metrics
    quantidade_acertos = int(np.sum(classes_previstas == y_teste))
    total_amostras = len(y_teste)
    acuracia = (quantidade_acertos / total_amostras) * 100

    print(f"Formato das probabilidades: {probabilidades_teste.shape}")
    print(f"Classificacoes corretas: {quantidade_acertos}/{total_amostras}")
    print(f"Classificacoes incorretas: {total_amostras - quantidade_acertos}")
    print(f"Acuracia no teste: {acuracia:.2f}%")

    print("\nExemplo das primeiras 5 previsoes:")
    for i in range(5):
        print(
            f"Amostra {i + 1}: real={y_teste[i]} | "
            f"prevista={classes_previstas[i]} | "
            f"probabilidades={np.round(probabilidades_teste[i], 4)}"
        )

    # Confere dimensoes, probabilidades e consistencia da acuracia
    assert probabilidades_teste.shape == (30, 3)
    assert classes_previstas.shape == (30,)
    assert np.all(np.isfinite(probabilidades_teste))
    assert np.all((probabilidades_teste >= 0) & (probabilidades_teste <= 1))
    assert np.allclose(np.sum(probabilidades_teste, axis=1), 1.0)
    assert np.all((classes_previstas >= 0) & (classes_previstas < 3))
    assert 0 <= quantidade_acertos <= total_amostras
    assert 0 <= acuracia <= 100

    print("\nAvaliacao concluida e validada com sucesso!")
    print("Todas as etapas da atividade foram executadas.")


if __name__ == "__main__":
    main()
