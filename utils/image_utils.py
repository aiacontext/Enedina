from PIL import Image
import torchvision.transforms as transforms


def load_image(file_path):
    """
    Carrega uma imagem do disco.

    Parâmetros:
        file_path (str): Caminho para o arquivo de imagem.

    Retorna:
        Image: Objeto Image do PIL carregado do arquivo.
    """
    return Image.open(file_path)


def transform_image(image):
    """
    Aplica transformações à imagem para prepará-la para o modelo. Isso pode incluir
    redimensionamento, normalização, conversão para tensor, e outras transformações necessárias.

    Parâmetros:
        image (Image): Objeto Image do PIL a ser transformado.

    Retorna:
        Tensor: Imagem transformada em tensor.
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize((640, 640)),  # Redimensiona a imagem para o tamanho desejado
        transforms.ToTensor(),  # Converte a imagem para tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normaliza a imagem
                             std=[0.229, 0.224, 0.225])
    ])
    return transform_pipeline(image)


def preprocess_image(file_path):
    """
    Combina o carregamento e a transformação de imagens em uma função única,
    simplificando a preparação de imagens para o modelo.

    Parâmetros:
        file_path (str): Caminho para o arquivo de imagem.

    Retorna:
        Tensor: Imagem carregada e transformada pronta para ser usada pelo modelo.
    """
    image = load_image(file_path)
    return transform_image(image)
