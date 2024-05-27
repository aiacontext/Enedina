import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_embedding import SparseTextEmbedding
from .image_processor import GenericProcessor as ImageProcessor
from .equation_processor import GenericProcessor as EquationProcessor
from .transformer_expert import TransformerExpert
from .transformer_decoder import TransformerDecoderWithCrossAttention


class EnedinaModel(nn.Module):
    """
    Modelo principal: Enedina.
    Integra diferentes componentes especializados para processar múltiplos tipos de entrada
    e combina as informações para produzir uma saída unificada.

    Atributos:
        text_embedding (SparseTextEmbedding): Processa entrada de texto.
        image_processor (GenericProcessor): Processa todas as entradas de imagem, incluindo diagramas.
        equation_processor (GenericProcessor): Processa entrada de equações.
        experts (nn.ModuleList): Lista de especialistas transformadores para cada tipo de entrada.
        gate (nn.Linear): Camada que combina as saídas dos especialistas.
        transformer_decoder (TransformerDecoderWithCrossAttention): Decodifica e combina as saídas.

    Métodos:
        forward(text_input, image_input, equation_input): Processa e combina todas as entradas.
    """

    def __init__(self, text_num_tokens, image_input_dim, equation_input_dim, emb_dim=1024,
                 num_heads=16, num_layers=12, ff_dim=4096):
        """
        Inicializa o modelo Enedina com todas as configurações necessárias para cada componente.

        Parâmetros:
            text_num_tokens (int): Número de tokens únicos para o embedding de texto.
            image_input_dim (int): Dimensão da entrada de imagem.
            equation_input_dim (int): Dimensão da entrada de equação.
            emb_dim (int): Dimensão dos embeddings.
            num_heads (int): Número de cabeças de atenção para os transformadores.
            num_layers (int): Número de camadas nos transformadores.
            ff_dim (int): Dimensão interna feed-forward dos transformadores.
        """
        super().__init__()
        self.text_embedding = SparseTextEmbedding(text_num_tokens, emb_dim)
        self.image_processor = ImageProcessor(image_input_dim, emb_dim)
        self.equation_processor = EquationProcessor(equation_input_dim, emb_dim)
        self.experts = nn.ModuleList([
            TransformerExpert(emb_dim, num_heads, num_layers, ff_dim) for _ in range(3)  # Text, Image, Equation
        ])
        self.gate = nn.Linear(emb_dim * 3, 3)
        self.transformer_decoder = TransformerDecoderWithCrossAttention(emb_dim, num_heads, num_layers, ff_dim)

    def forward(self, text_input, image_input, equation_input):
        """
        Processa as entradas de texto, imagem e equação através de seus respectivos componentes,
        combina as saídas, e utiliza o decodificador Transformer para sintetizar a saída final.

        Parâmetros:
            text_input (Tensor): Entrada de texto.
            image_input (Tensor): Entrada de imagem.
            equation_input (Tensor): Entrada de equação.

        Retorna:
            Tensor: Saída combinada e transformada do modelo.
        """
        text_emb = self.text_embedding(text_input)
        image_emb = self.image_processor(image_input).unsqueeze(1)
        equation_emb = self.equation_processor(equation_input).unsqueeze(1)

        expert_inputs = [text_emb, image_emb, equation_emb]
        expert_outputs = []
        for expert, input in zip(self.experts, expert_inputs):
            output = expert(input)
            expert_outputs.append(output[:, -1, :])  # assume processing per batch

        combined_expert_outputs = torch.cat(expert_outputs, dim=1)
        gate_weights = F.softmax(self.gate(combined_expert_outputs), dim=1)
        expert_outputs_stack = torch.stack(expert_outputs, dim=1)
        combined_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs_stack, dim=1)

        final_output = self.transformer_decoder(combined_output.unsqueeze(0), text_emb)

        return final_output
