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
        Dynamically loads a model from a YAML configuration file.

        Parameters:
        conf_path (str): The file path to the YAML configuration file.

        Returns:
        object: An instance of the model class specified in the configuration file.

        Raises:
        KeyError: If the required keys ('model', 'module', 'class') are missing in the configuration.
        ValueError: If there is an error loading the model from the configuration file.
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
        Loads a checkpoint and applies the weights to the model.

        Parameters:
        ckpt_path (str): The file path to the checkpoint file.
        model_path (str): The file path to the model configuration file.

        Returns:
        object: The model with weights loaded from the checkpoint, set to evaluation mode.

        Raises:
        ValueError: If there is an error loading the checkpoint or applying the weights.
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
        Creates a unique file name by incrementing version numbers if the file already exists.

        Parameters:
        path (str): The directory path where the file will be saved.
        nombre_base (str): The base name for the file.
        extension (str): The file extension, default is ".onnx".

        Returns:
        str: A unique file path with an incremented version number.
        """
        os.makedirs(path, exist_ok=True)  # Asegurarse de que el directorio exista.

        contador = 1
        archivo = os.path.join(path, f"{nombre_base}_v{contador}{extension}")

        while os.path.exists(archivo):
            archivo = os.path.join(path, f"{nombre_base}_v{contador}{extension}")
            contador += 1

        return archivo

    def export_to_onnx(self, model, input_net_size, onnx_path, nombre_base):
        """
        Exports a PyTorch model to ONNX format.

        Parameters:
            model (torch.nn.Module): The PyTorch model to export.
            input_net_size (tuple): The input dimensions for the model (e.g., (1, 3, 224, 224)).
            onnx_path (str): The path where the ONNX model will be saved.
            base_name (str): The base name for the ONNX file, used to create incremental versions.

        Returns:
            str: The full path of the exported ONNX model.

        Raises:
            RuntimeError: If an error occurs during the export process.
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
        Converts a model from a checkpoint to ONNX format.

        Parameters:
        ckpt_path (str): The file path to the checkpoint file.
        model_path (str): The file path to the model configuration file.
        onnx_path (str): The directory path where the ONNX file will be saved.
        input_net_size (tuple): The size of the input tensor for the model.
        nombre_base (str): The base name for the ONNX file.

        Returns:
        str: The file path to the exported ONNX model.
        """
        try:
            model = self.load_checkpoint(ckpt_path, model_path)
            onnx_path = self.export_to_onnx(
                model, input_net_size, onnx_path, nombre_base
            )
            return onnx_path
        except Exception as e:
            raise RuntimeError(
                f"Error durante la conversión del checkpoint a ONNX: {e}"
            )
