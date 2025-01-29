import unittest
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import onnxruntime

# Agregar el directorio raíz del proyecto dinámicamente al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training_step import initialize, Generic_trainings
from export_step import initialize, ExportationGeneric

# Llama a la función de inicialización del paquete
initialize()

# from export_step.export_module import ExportationGeneric
from test_generic_training.dummyclass import DummyClass


class TestExpOnnx(unittest.TestCase):
    def setUp(self):
        # Crear un dataset con datos ficticios
        self.data = [1, 2, 3, 4, 5]
        self.dataset = DummyClass(self.data)
        self.dataset_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=True
        )

        # Set hyperparameters
        self.n_epochs = 1
        self.conf_path = "./src/training_step/config_training.yaml"

        # Crear un checkpoint callback para guardar los mejores modelos
        self.checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best_model",
            verbose=True,
        )

        # Crear un modelo aleatorio
        self.model = Generic_trainings(self.conf_path)
        self.trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=self.n_epochs,
            limit_train_batches=100,
            callbacks=[self.checkpoint_callback],
        )

        self.onnx_path = "onnx_models"
        # Usa la clase ExportationGeneric
        self.exporter = ExportationGeneric()

    def tearDown(self):
        # Eliminar archivos generados durante las pruebas
        if os.path.exists(self.checkpoint_callback.dirpath):
            for file in os.listdir(self.checkpoint_callback.dirpath):
                os.remove(os.path.join(self.checkpoint_callback.dirpath, file))
            os.rmdir(self.checkpoint_callback.dirpath)

        if os.path.exists(self.onnx_path):
            for file in os.listdir(self.onnx_path):
                os.remove(os.path.join(self.onnx_path, file))
            os.rmdir(self.onnx_path)

    def run_training(self):
        self.trainer.fit(
            self.model,
            train_dataloaders=self.dataset_loader,
        )

    def test_training(self):
        """
        Verifica que el proceso de entrenamiento se complete sin errores.
        """
        try:
            self.run_training()
        except Exception as e:
            self.fail(f"El entrenamiento falló con una excepción: {e}")

    def test_export_to_onnx(self):
        """
        Verifica que el modelo entrenado pueda exportarse correctamente a ONNX.
        """
        # Ejecutar entrenamiento
        self.run_training()

        # Probar la exportación a ONNX
        try:
            onnx_file = self.exporter.from_ckpt_to_onnx(
                "checkpoints/best_model.ckpt",
                self.conf_path,
                self.onnx_path,
                1,
                "onnx_model",
            )

            # Verificar que el archivo ONNX fue creado
            self.assertTrue(
                os.path.exists(onnx_file), "El archivo ONNX no se generó correctamente."
            )

            # Validar el archivo ONNX cargándolo
            onnxruntime.InferenceSession(onnx_file)
        except Exception as e:
            self.fail(f"La conversión a ONNX falló con la excepción: {e}")


if __name__ == "__main__":
    unittest.main()
