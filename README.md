# Enedina: Combinando Diferentes Tipos de Entrada com Decodificador Transformer

Este repositório contém uma Prova de Conceito (PoC) de um modelo chamado Enedina, que combina diferentes tipos de entrada (texto, imagem, equação e diagrama) usando um decodificador Transformer com atenção cruzada.

## Visão Geral

O EnedinaModel é projetado para processar e integrar informações de várias modalidades, como texto, imagem, equação e diagrama. Ele usa componentes modulares para processar cada tipo de entrada e gerar embeddings correspondentes. Em seguida, os embeddings são concatenados e passados para um decodificador Transformer com atenção cruzada, que captura as dependências e interações entre as diferentes entradas.

## Arquitetura do Modelo

A arquitetura do EnedinaModel consiste nos seguintes componentes principais:

- `TextEmbedding`: Uma camada de embedding para processar entradas de texto.
- `GenericProcessor`: Um processador genérico para transformar entradas numéricas em embeddings.
- `TransformerDecoderWithCrossAttention`: Um decodificador Transformer com atenção cruzada para processar os embeddings concatenados.

## Uso

Para usar o EnedinaModel, siga estas etapas:

1. Clone este repositório:
   ```
   git clone https://github.com/seu-usuario/enedina-model.git
   ```

2. Instale as dependências necessárias:
   ```
   pip install -r requirements.txt
   ```

3. Importe a classe `EnedinaModel` e crie uma instância do modelo:
   ```python
   from enedina_model import EnedinaModel

   model = EnedinaModel(text_num_tokens, image_input_dim, equation_input_dim, diagram_input_dim)
   ```

4. Prepare suas entradas (texto, imagem, equação e diagrama) no formato adequado.

5. Passe as entradas para o modelo e obtenha a saída:
   ```python
   output = model(text_input, image_input, equation_input, diagram_input)
   ```

Para obter mais detalhes sobre os parâmetros e a configuração do modelo, consulte o código-fonte e a documentação neste repositório.

## Contribuição

Contribuições são bem-vindas! Se você encontrar algum problema, tiver sugestões de melhorias ou quiser adicionar novos recursos, fique à vontade para abrir uma issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

## Contato

Se você tiver alguma dúvida ou quiser entrar em contato, pode me encontrar no Hugging Face ou através do meu e-mail: contato@aiacontext.com.
Espero que este modelo seja útil para você! Sinta-se à vontade para personalizar e expandir de acordo com suas necessidades.
