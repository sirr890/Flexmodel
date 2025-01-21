import torch
import pytorch_lightning as pl
import yaml
import importlib


class Generic_trainings(pl.LightningModule):
    def load_model_from_config(self, config):
        module_name = config["model"]["module"]
        class_name = config["model"]["class"]
        parameters = config["model"].get(
            "parameters", {}
        )  # Cargar parámetros si existen
        # Parámetros del modelo a crear a partir del YAML
        try:
            parameters = config["model"]["parameters"]
        except:
            parameters = {}

        # Importar dinámicamente el módulo y la clase
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        # Crear una instancia del modelo
        return model_class(**parameters)

    def get_loss_function(self, config):
        loss = config["training"]["loss"]
        module_loss = importlib.import_module(loss["module_loss_function"])
        class_loss = loss["class_loss_function"]
        parameters = loss.get("parameters", {})  # Cargar parámetros si existen
        return getattr(module_loss, class_loss)(**parameters)

    def get_optimizer(self, config):
        optimizer = config["training"]["optimizer"]
        module_optimizer = importlib.import_module(optimizer["module_optimizer"])
        class_optimizer = optimizer["class_optimizer"]
        parameters = optimizer.get("parameters", {})  # Cargar parámetros si existen
        return getattr(module_optimizer, class_optimizer)(
            self.model.parameters(), **parameters
        )

    def load_metrics_from_config(self, config):
        """Carga dinámicamente las métricas desde la configuración YAML."""
        metrics = []
        for metric_config in config["metrics"]:
            module_name = metric_config["module"]
            class_name = metric_config["class"]
            parameters = metric_config.get("parameters", {})  # Parámetros opcionales

            # Importar dinámicamente el módulo y la clase o función
            module = importlib.import_module(module_name)
            metric_function = getattr(module, class_name)

            # Agregar la métrica como un diccionario
            metrics.append(
                {
                    "name": metric_config["name"],
                    "func": metric_function,
                    "params": parameters,
                }
            )
        return metrics

    def __init__(self, conf_path):
        super(Generic_trainings, self).__init__()
        # Leer el archivo YAML
        with open(conf_path, "r") as file:
            config = yaml.safe_load(file)
        self.model = self.load_model_from_config(config)
        self.loss_function = self.get_loss_function(config)
        self.optimizer = self.get_optimizer(config)
        self.metrics = self.load_metrics_from_config(config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric in self.metrics:
            metric_value = metric["func"](logits, y, **metric["params"])
            self.log(f"val_{metric['name']}", metric_value)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        for metric in self.metrics:
            metric_value = metric["func"](logits, y, **metric["params"])
            self.log(f"test_{metric['name']}", metric_value)
        return loss

    def configure_optimizers(self):
        return self.optimizer
