import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderWithCrossAttention(nn.Module):
    """
    Decodificador Transformer com atenção cruzada.
    Combinando informações de múltiplas fontes e projetando o resultado final,
    este componente é essencial para integrar efetivamente várias modalidades de entrada.

    Atributos:
        transformer_decoder (nn.TransformerDecoder): Decodificador Transformer que aplica atenção cruzada para combinar informações.
        projection (nn.Linear): Camada linear para projetar a saída do decodificador para a dimensão desejada.

    Métodos:
        forward(x, memory): Processa as entradas x com base na memória de contexto fornecida.
    """

    def __init__(self, emb_dim, num_heads, num_layers, ff_dim):
        """
        Inicializa o decodificador Transformer com atenção cruzada, configurado para integrar informações de múltiplas fontes.

        Parâmetros:
            emb_dim (int): Dimensão do embedding que serve como a dimensão do modelo no Transformer.
            num_heads (int): Número de cabeças de atenção multi-cabeça no Transformer.
            num_layers (int): Número de camadas do decodificador Transformer.
            ff_dim (int): Dimensão da camada feed-forward interna do Transformer.
        """
        super().__init__()
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                                               batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, memory):
        """
        Aplica o decodificador Transformer à entrada x, utilizando a memória de contexto fornecida.

        Parâmetros:
            x (Tensor): Tensor de entrada que representa embeddings concatenados.
            memory (Tensor): Tensor de memória que contém informações contextuais para aplicar atenção cruzada.

        Retorna:
            Tensor: Tensor transformado após passar pelo decodificador Transformer e pela projeção, mantendo a mesma dimensão de entrada.
        """
        output = self.transformer_decoder(x, memory)
        output = self.projection(output)
        return output
