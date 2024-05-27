import torch
import torch.nn as nn


class TransformerExpert(nn.Module):
    """
    Especialista em domínio específico usando um encoder Transformer.
    Projetado para processar embeddings e realizar tarefas específicas de domínio,
    como análise de texto, processamento de imagens, ou interpretação de equações,
    usando a poderosa arquitetura Transformer para capturar dependências complexas.

    Atributos:
        transformer_encoder (nn.TransformerEncoder): O encoder Transformer que realiza a transformação dos dados.

    Métodos:
        forward(x): Passa a entrada x através do encoder Transformer.
    """

    def __init__(self, emb_dim, num_heads, num_layers, ff_dim):
        """
        Inicializa o especialista Transformer com uma configuração específica para tarefas de domínio.

        Parâmetros:
            emb_dim (int): Dimensão do embedding que também serve como dimensão do modelo no Transformer.
            num_heads (int): Número de cabeças de atenção multi-cabeça no Transformer.
            num_layers (int): Número de camadas do encoder Transformer.
            ff_dim (int): Dimensão da camada feed-forward interna do Transformer.
        """
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                                       batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Aplica o encoder Transformer à entrada x.

        Parâmetros:
            x (Tensor): Tensor de entrada com embeddings. Deve ter as dimensões [batch_size, seq_length, emb_dim].

        Retorna:
            Tensor: Tensor transformado após passar pelo encoder Transformer, mantendo a mesma dimensão de entrada.
        """
        return self.transformer_encoder(x)
