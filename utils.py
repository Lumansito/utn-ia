"""Funciones reutilizables para el pipeline de imágenes.

Movidas aquí para que puedan ser importadas por los workers del DataLoader
(no se puede picklear funciones definidas en el __main__ de un notebook Jupyter).
"""
from PIL import Image


def cargar_imagen_limpia(path):
    """Carga una imagen y la convierte a RGB estándar.

    Maneja formatos con transparencia (P, RGBA, LA) pegándolos sobre fondo blanco.
    """
    img = Image.open(path)
    if img.mode in ('P', 'RGBA', 'LA'):
        img = img.convert('RGBA')
        fondo = Image.new('RGB', img.size, (255, 255, 255))
        fondo.paste(img, mask=img.split()[3])
        return fondo
    return img.convert('RGB')