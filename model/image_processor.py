import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericProcessor(nn.Module):
    """
    Processador genérico que transforma entradas numéricas em embeddings.
    É utilizado para processar imagens ou outras entradas numéricas, convertendo-as
    em uma representação vetorial que pode ser utilizada em operações de aprendizado subsequentes.

    Atributos:
        fc (nn.Linear): Camada linear que transforma a entrada em uma dimensão de embedding desejada.

    Métodos:
        forward(x): Aplica a transformação linear e uma ativação ReLU à entrada x.
    """

    def __init__(self, input_dim, emb_dim):
        """
        Inicializa o processador genérico com uma camada linear.

        Parâmetros:
            input_dim (int): Dimensão da entrada.
            emb_dim (int): Dimensão do embedding desejado.
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        """
        Processa a entrada x através de uma camada linear e ativação ReLU.

        Parâmetros:
            x (Tensor): Tensor de entrada que pode ser uma imagem achatada ou qualquer outra entrada numérica.

        Retorna:
            Tensor: Tensor de saída com a dimensão de embedding especificada.
        """
        return F.relu(self.fc(x))
