from dummyclass import DummyClass
import torch
import pytorch_lightning as pl
from dummymodel import Random
import os
import sys
import unittest

# AAgregar el directorio raíz del proyecto dinámicamente al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training_step.training_module import Generic_trainings


class TestGenericTrainings(unittest.TestCase):
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

        # Crear un modelo aleatorio
        self.model = Generic_trainings(self.conf_path)
        self.trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=self.n_epochs,
            limit_train_batches=100,
        )

    def test_training_success(self):
        """
        Verifica que el proceso de entrenamiento se complete sin errores.
        """
        try:
            self.trainer.fit(self.model, train_dataloaders=self.dataset_loader)
            print("Entrenamiento completado con éxito.")
        except Exception as e:
            self.fail(f"El entrenamiento falló con una excepción: {e}")


if __name__ == "__main__":
    unittest.main()
