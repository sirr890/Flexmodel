import tensorflow as tf
import yaml
import importlib
import os
import torch


class ExportationGeneric:
    def __init__(self):
        pass

    def load_model_from_config(self, conf_path):
        """
        Carga un modelo dinámicamente desde un archivo de configuración YAML.
        """
        try:
            with open(conf_path, "r") as file:
                config = yaml.safe_load(file)

            # Verificar que las claves necesarias estén presentes
            if "model" not in config:
                raise KeyError("La clave 'model' falta en el archivo de configuración.")
            if "module" not in config["model"] or "class" not in config["model"]:
                raise KeyError(
                    "El archivo de configuración debe contener 'module' y 'class'."
                )

            module_name = config["model"]["module"]
            class_name = config["model"]["class"]
            parameters = config["model"].get("parameters", {})

            # Importar dinámicamente el módulo y la clase
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            model = model_class(**parameters)
            return model
        except Exception as e:
            raise ValueError(
                f"Error al cargar el modelo desde el archivo de configuración: {e}"
            )

    def load_checkpoint(self, ckpt_path, model_path):
        """
        Carga un checkpoint y aplica los pesos al modelo.
        """
        try:
            # Cargar el modelo desde la configuración
            model = self.load_model_from_config(model_path)

            # Cargar el checkpoint
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
            if "state_dict" not in checkpoint:
                raise KeyError(
                    "La clave 'state_dict' no está presente en el checkpoint."
                )

            state_dict = checkpoint["state_dict"]
            new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(new_state_dict)
            model.eval()  # Poner el modelo en modo evaluación
            return model
        except Exception as e:
            raise ValueError(f"Error al cargar el checkpoint: {e}")

    def crear_version_incremental(self, path, nombre_base, extension=".onnx"):
        """
        Crea un nombre de archivo único incrementando versiones si ya existe.
        """
        os.makedirs(path, exist_ok=True)  # Asegurarse de que el directorio exista

        contador = 1
        archivo = os.path.join(path, f"{nombre_base}_v{contador}{extension}")

        while os.path.exists(archivo):
            archivo = os.path.join(path, f"{nombre_base}_v{contador}{extension}")
            contador += 1

        return archivo

    def export_to_onnx(self, model, input_net_size, onnx_path, nombre_base):
        """
        Exporta un modelo de PyTorch a formato ONNX.
        """
        try:
            dummy_input = torch.randn(input_net_size)
            onnx_path = self.crear_version_incremental(
                onnx_path, nombre_base, extension=".onnx"
            )

            # Exportar el modelo a ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=11,
                input_names=["input"],
                output_names=["output"],
            )
            print(f"Modelo exportado a ONNX en {onnx_path}")
            return onnx_path
        except Exception as e:
            raise RuntimeError(f"Error durante la exportación a ONNX: {e}")

    def from_ckpt_to_onnx(
        self, ckpt_path, model_path, onnx_path, input_net_size, nombre_base
    ):
        """
        Convierte un modelo de un checkpoint a ONNX.
        """
        try:
            model = self.load_checkpoint(ckpt_path, model_path)
            self.export_to_onnx(model, input_net_size, onnx_path, nombre_base)
            return onnx_path
        except Exception as e:
            raise RuntimeError(
                f"Error durante la conversión del checkpoint a ONNX: {e}"
            )
