import os
import torch
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

from .text_utils import load_text, process_text
from .image_utils import transform_image
from .equation_utils import load_equations


def load_image_with_annotations(image_path, annotation_dir):
    """
    Carrega uma imagem e suas anotações correspondentes em XML e Markdown.

    Parâmetros:
        image_path (str): Caminho completo para o arquivo de imagem.
        annotation_dir (str): Diretório que contém os arquivos de anotações XML e Markdown.

    Retorna:
        dict: Um dicionário contendo a imagem e suas anotações processadas.
    """
    # Carrega a imagem
    image = Image.open(image_path)
    image = transform_image(image)  # Assume que transform_image converte a imagem para tensor e aplica transformações

    # Carrega as anotações XML e Markdown
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    xml_path = os.path.join(annotation_dir, base_filename + '.xml')
    md_path = os.path.join(annotation_dir, base_filename + '.md')

    # Processa as anotações
    with open(xml_path, 'r') as file:
        xml_content = file.read()
    annotations = ET.fromstring(xml_content)  # Simples processamento de XML

    text_annotation = load_text(md_path)  # Carrega e processa Markdown

    return {
        'image': image,
        'annotations': annotations,
        'text_annotation': text_annotation
    }


def load_dataset(dataset_dir):
    """
    Carrega e processa todos os dados de um diretório de conjunto de dados.

    Parâmetros:
        dataset_dir (str): Caminho do diretório do conjunto de dados que contém subdiretórios para treino, validação e teste.

    Retorna:
        dict: Dicionários para treino, validação e teste, cada um contendo listas de dados de imagem, texto e equação.
    """
    subsets = ['train', 'val', 'test']
    dataset = {}

    for subset in subsets:
        images = []
        texts = []
        equations = []

        # Caminhos para subdiretórios
        visual_path = os.path.join(dataset_dir, subset, 'visual', 'imagens')
        textual_path = os.path.join(dataset_dir, subset, 'textual')
        equations_path = os.path.join(dataset_dir, subset, 'equacoes')
        annotations_path = os.path.join(dataset_dir, subset, 'visual', 'anotacoes')

        # Carrega imagens e anotações
        for image_file in os.listdir(visual_path):
            image_path = os.path.join(visual_path, image_file)
            image_data = load_image_with_annotations(image_path, annotations_path)
            images.append(image_data)

        # Carrega textos soltos
        for text_file in os.listdir(textual_path):
            text_path = os.path.join(textual_path, text_file)
            text_content = load_text(text_path)
            texts.append(process_text(text_content))  # process_text pode envolver alguma normalização ou tokenização

        # Carrega equações
        for eq_file in os.listdir(equations_path):
            eq_path = os.path.join(equations_path, eq_file)
            eq_content = load_equations(eq_path)
            equations.append(eq_content)

        # Armazena dados no dicionário do dataset
        dataset[subset] = {
            'images': images,
            'texts': texts,
            'equations': equations
        }

    return dataset
