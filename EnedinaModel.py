import torch
import torch.nn as nn
import torch.nn.functional as F


# Define uma camada de embedding para processar entradas de texto
class TextEmbedding(nn.Module):
    def __init__(self, num_tokens, emb_dim):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_tokens, emb_dim)

    def forward(self, x):
        return self.embedding(x)


# Define um processador genérico para transformar entradas numéricas em embeddings
class GenericProcessor(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(GenericProcessor, self).__init__()
        self.fc = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        return F.relu(self.fc(x))


# Define um decodificador Transformer com atenção cruzada
class TransformerDecoderWithCrossAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers, ff_dim):
        super(TransformerDecoderWithCrossAttention, self).__init__()
        transformer_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_decoder = nn.TransformerDecoder(transformer_layer, num_layers=num_layers)
        self.projection = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, memory):
        # Ajusta a entrada para o formato esperado [seq_len, batch_size, emb_dim]
        # Aqui, assumimos que `x` e `memory` já estão com as dimensões corretas,
        # onde a dimensão seq_len de `x` é determinada pelo modelo anterior que processa a entrada.
        output = self.transformer_decoder(x, memory)
        # Aplica uma camada linear de projeção para mapear a saída do decodificador
        # de volta para o tamanho de embedding desejado.
        return self.projection(output)


# Define o modelo principal que incorpora os componentes acima
class EnedinaModel(nn.Module):
    def __init__(self, text_num_tokens, image_input_dim, equation_input_dim, diagram_input_dim,
                 emb_dim=512, num_heads=8, num_layers=6, ff_dim=2048):
        super(EnedinaModel, self).__init__()
        # Inicializa os componentes do modelo
        self.text_embedding = TextEmbedding(text_num_tokens, emb_dim)
        self.image_processor = GenericProcessor(image_input_dim, emb_dim)
        self.equation_processor = GenericProcessor(equation_input_dim, emb_dim)
        self.diagram_processor = GenericProcessor(diagram_input_dim, emb_dim)
        self.transformer_decoder = TransformerDecoderWithCrossAttention(emb_dim, num_heads, num_layers, ff_dim)

    def forward(self, text_input, image_input, equation_input, diagram_input):
        # Verifica as dimensões das entradas
        assert text_input.dim() == 2, "A entrada de texto deve ter dimensões (batch_size, seq_len)"
        assert image_input.dim() == 2, "A entrada de imagem deve ter dimensões (batch_size, image_input_dim)"
        assert equation_input.dim() == 2, "A entrada de equação deve ter dimensões (batch_size, equation_input_dim)"
        assert diagram_input.dim() == 2, "A entrada de diagrama deve ter dimensões (batch_size, diagram_input_dim)"

        # Processa as entradas através dos respectivos componentes
        text_emb = self.text_embedding(text_input)
        image_emb = self.image_processor(image_input)
        equation_emb = self.equation_processor(equation_input)
        diagram_emb = self.diagram_processor(diagram_input)

        # Ajusta as dimensões dos embeddings para permitir a concatenação
        # Importante: Ajusta para que todos os embeddings sejam 3D [batch_size, seq_len, emb_dim]
        # Texto já é [batch_size, seq_len, emb_dim] por padrão
        # Para imagem, equação e diagrama, adiciona-se uma dimensão seq_len fictícia
        image_emb = image_emb.unsqueeze(1)  # Agora é [batch_size, 1, emb_dim]
        equation_emb = equation_emb.unsqueeze(1)  # Agora é [batch_size, 1, emb_dim]
        diagram_emb = diagram_emb.unsqueeze(1)  # Agora é [batch_size, 1, emb_dim]

        # Concatena os embeddings
        combined = torch.cat([text_emb, image_emb, equation_emb, diagram_emb], dim=1)

        # Aplica o decodificador Transformer ao embedding combinado
        output = self.transformer_decoder(combined, combined)

        return output


# Configuração e simulação de entrada
text_num_tokens = 20000
image_input_dim = 512
equation_input_dim = 256
diagram_input_dim = 128
batch_size = 4
seq_len = 10

# Inicializa o modelo
model = EnedinaModel(text_num_tokens, image_input_dim, equation_input_dim, diagram_input_dim)

# Gera entradas simuladas
text_input = torch.randint(0, text_num_tokens, (batch_size, seq_len))
image_input = torch.randn(batch_size, image_input_dim)
equation_input = torch.randn(batch_size, equation_input_dim)
diagram_input = torch.randn(batch_size, diagram_input_dim)

# Executa o modelo com as entradas simuladas
output = model(text_input, image_input, equation_input, diagram_input)

# Imprime a forma da saída para verificação
print(output.shape)
