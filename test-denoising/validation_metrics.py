import torch


# Función para calcular PSNR
def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")  # Si no hay diferencia, PSNR es infinito
    max_pixel = 1.0  # Asumiendo que las imágenes están normalizadas en el rango [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


# Función para calcular MSE
def compute_mse(img1, img2):
    return torch.mean((img1 - img2) ** 2)
