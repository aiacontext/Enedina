import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericProcessor(nn.Module):
    """
    Processador genérico para transformar entradas numéricas em embeddings.
    Este processador é utilizado especificamente para equações, convertendo dados numéricos,
    como coeficientes de equações ou matrizes de características, em uma representação vetorial.

    Atributos:
        fc (nn.Linear): Camada linear que converte a entrada numérica na dimensão desejada de embedding.

    Métodos:
        forward(x): Aplica uma transformação linear seguida por uma ativação ReLU à entrada x.
    """

    def __init__(self, input_dim, emb_dim):
        """
        Inicializa o processador de equações com uma camada linear.

        Parâmetros:
            input_dim (int): Dimensão da entrada numérica, que pode representar o número de características de uma equação.
            emb_dim (int): Dimensão do embedding desejado para a saída.
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        """
        Processa a entrada numérica através de uma camada linear e uma ativação ReLU.

        Parâmetros:
            x (Tensor): Tensor de entrada que representa dados numéricos de equações.

        Retorna:
            Tensor: Tensor de saída com a dimensão de embedding especificada.
        """
        return F.relu(self.fc(x))
