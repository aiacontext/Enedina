import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseTextEmbedding(nn.Module):
    """
    Camada de embedding para texto com atenção multi-cabeça.
    Realiza embeddings de tokens de texto e aplica atenção multi-cabeça.
    Este módulo é projetado para capturar contextos complexos em textos, o que é essencial
    para tarefas de processamento de linguagem natural em engenharia e outras áreas técnicas.

    Atributos:
        num_tokens (int): Número de tokens únicos no vocabulário.
        emb_dim (int): Dimensão do embedding.
        attention (nn.MultiheadAttention): Camada de atenção multi-cabeça.

    Métodos:
        forward(x): Passa a entrada x através do embedding e da atenção.
    """

    def __init__(self, num_tokens, emb_dim):
        """
        Inicializa a camada de embedding com atenção esparsa para texto.

        Parâmetros:
            num_tokens (int): Número total de tokens no vocabulário.
            emb_dim (int): Dimensão do vetor de embedding.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim, num_heads=8, batch_first=True)

    def forward(self, x):
        """
        Aplica o processo de embedding seguido por atenção multi-cabeça à entrada x.

        Parâmetros:
            x (Tensor): Tensor de tokens de texto com shape [batch_size, seq_length].

        Retorna:
            Tensor: Tensor processado com atenção com shape [batch_size, seq_length, emb_dim].
        """
        x = self.embedding(x)
        x, _ = self.attention(x, x, x)
        return x
