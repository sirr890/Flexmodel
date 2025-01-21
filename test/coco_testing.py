from src.training_step.training_module import Generic_trainings
import torch
from coco_data import MyImage, RandomCrop
from validation_metrics import compute_psnr, compute_mse
import wandb
from torchvision import transforms
import onnxruntime as ort
import numpy as np
import pytorch_lightning as pl


# Configurar el logger de WandB
wandb.init(
    project="Coco project",
    name="plot testing",
    config={"batch_size": 1},  # Opcional: configura parámetros adicionales
)


def cargar_modelo_onnx(onnx_path):
    """Carga un modelo ONNX usando ONNX Runtime."""
    try:
        session = ort.InferenceSession(onnx_path)
        print("Modelo ONNX cargado correctamente.")
        return session
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo ONNX: {e}")


def ejecutar_inferencia(session, input_tensor):
    """
    Ejecuta la inferencia con el modelo ONNX cargado.
    """
    try:
        input_name = session.get_inputs()[0].name

        # Convertir input_tensor a numpy array si es un tensor de PyTorch
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.numpy()

        input_tensor = input_tensor.astype(
            np.float32
        )  # Asegurarse de que el tipo sea float32

        # Ejecutar la inferencia
        outputs = session.run(None, {input_name: input_tensor})
        print("Inferencia ejecutada correctamente.")
        return outputs
    except Exception as e:
        raise RuntimeError(f"Error durante la inferencia: {e}")


def main():
    conf_path = "./src/training_step/config_training.yaml"
    onnx_path = "./onnx_models/e_model_v2.onnx"  # Ruta al modelo ONNX
    # Cargar el modelo ONNX
    model = cargar_modelo_onnx(onnx_path)

    # Testar el modelo con n imagenes de teste y visualizar las metricas en un grafico
    test_data = MyImage(200, 300, transform=transforms.Compose([RandomCrop(256)]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

    # Evaluar las metricas de psnr y mse
    for batch_idx, batch in enumerate(test_loader):
        image, gt = batch
        out = ejecutar_inferencia(model, image)
        out = torch.tensor(out[0])

        psnr_value = compute_psnr(gt, out)
        mse_value = compute_mse(gt, out)

        # Log de métricas en WandB
        wandb.log({"test_mse": mse_value, "test_psnr": psnr_value})

        # Convertir tensores a imágenes
        image_np = transforms.ToPILImage()(
            image.squeeze(0)
        )  # Convertir tensor a imagen PIL
        gt_np = transforms.ToPILImage()(gt.squeeze(0))  # Ground Truth
        out_np = transforms.ToPILImage()(out.squeeze(0))  # Predicción del modelo

        # Loguear imágenes
        wandb.log(
            {
                "input_image": wandb.Image(image_np, caption=f"Input {batch_idx}"),
                "model_output": wandb.Image(
                    out_np, caption=f"Model Output {batch_idx}"
                ),
            }
        )

    wandb.finish()


if __name__ == "__main__":
    main()
