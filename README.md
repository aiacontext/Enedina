# Enedina: Grande Modelo de Linguagem Multimodal para Engenharia Brasileira

Este repositório hospeda a Prova de Conceito (PoC) do LLM multimodal Enedina, que é capaz de integrar entradas de texto, imagem e equações utilizando uma arquitetura avançada baseada em Transformers com atenção cruzada.

## Visão Geral

O Enedina é projetado para processar e integrar informações de múltiplas modalidades. Utilizando uma arquitetura modular, ele transforma entradas de texto, imagem e equações em embeddings, que são posteriormente processados para capturar interações complexas entre estas modalidades através de um decodificador Transformer com atenção cruzada.

## Arquitetura do Modelo

A arquitetura do Enedina foi significativamente refatorada para melhorar a modularidade e a robustez. Ela consiste nos seguintes componentes principais:

- `TextEmbedding`: Processa entradas de texto utilizando embeddings e atenção multi-cabeça.
- `ImageProcessor`: Transforma entradas de imagem em embeddings numéricos.
- `EquationProcessor`: Converte equações em formato simbólico para representações numéricas.
- `TransformerExpert`: Cada modalidade é processada por um expert especializado antes de ser combinada.
- `TransformerDecoderWithCrossAttention`: Integra as saídas dos diferentes processadores, utilizando atenção cruzada para entender as interações entre as modalidades.

### Estrutura de Diretórios

O projeto foi organizado em uma estrutura de diretórios modularizada que facilita a manutenção e a expansão.

## Uso

Para utilizar o modelo Enedina, siga estas etapas:

1. Clone o repositório:
   ```
   git clone https://github.com/aiacontext/Enedina.git
   ```

2. Instale as dependências necessárias:
   ```
   pip install -r requirements.txt
   ```

3. Importe a classe `EnedinaModel` e crie uma instância do modelo:
   ```python
   from enedina_model import EnedinaModel

   model = EnedinaModel(text_num_tokens, image_input_dim, equation_input_dim)
   ```

4. Prepare suas entradas (texto, imagem e equação) no formato adequado.

5. Passe as entradas para o modelo e obtenha a saída:
   ```python
   output = model(text_input, image_input, equation_input)
   ```

## Contribuição

Contribuições são bem-vindas! Se você encontrar algum problema, tiver sugestões de melhorias ou quiser adicionar novos recursos, fique à vontade para abrir uma issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

## Contato

Se você tiver alguma dúvida ou quiser entrar em contato, pode me encontrar no Hugging Face ou através do meu e-mail: contato@aiacontext.com.
Espero que este modelo seja útil para você! Sinta-se à vontade para personalizar e expandir de acordo com suas necessidades.
