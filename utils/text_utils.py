def load_text(file_path):
    """
    Carrega texto de um arquivo.

    Parâmetros:
        file_path (str): Caminho para o arquivo de texto.

    Retorna:
        str: Conteúdo do arquivo de texto.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def process_text(text):
    """
    Processa e normaliza o texto. Esta função pode incluir passos como tokenização,
    remoção de caracteres especiais, conversão para minúsculas, etc., dependendo das
    necessidades específicas do projeto e do modelo.

    Parâmetros:
        text (str): Texto bruto a ser processado.

    Retorna:
        str: Texto processado.
    """
    # Exemplo simples de processamento: remover caracteres especiais e converter para minúsculas
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())

    return text


def tokenize_text(text):
    """
    Tokeniza o texto processado em tokens individuais. Essa função pode utilizar
    ferramentas como spaCy, NLTK ou qualquer tokenizador personalizado conforme necessário.

    Parâmetros:
        text (str): Texto processado a ser tokenizado.

    Retorna:
        list: Lista de tokens.
    """
    # Exemplo de tokenização simples, usando espaços como delimitadores
    tokens = text.split()

    return tokens
