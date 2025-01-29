import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))
from export_step.export_module import ExportationGeneric

# Instancia de la clase
export2onnx = ExportationGeneric()

# Configuración de prueba
model_path = "./test-denoising/config_training.yaml"  # Ruta al archivo de configuración del modelo
ckpt_path = "checkpoints/best_model.ckpt"  # Ruta al checkpoint del mejor modelo segun la etapa de validacion
onnx_path = "./onnx_models"  # Directorio para guardar el modelo ONNX
nombre_base = "e_model"  # Nombre base del archivo ONNX
input_net_size = (1, 3, 256, 256)  # Tamaño del tensor de entrada

# Llamada a la función principal
try:
    onnx_file = export2onnx.from_ckpt_to_onnx(
        ckpt_path, model_path, onnx_path, input_net_size, nombre_base
    )
    print(f"Modelo ONNX creado: {onnx_file}")
except Exception as e:
    print(f"Error: {e}")
