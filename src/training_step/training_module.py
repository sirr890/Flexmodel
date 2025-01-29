import torch
import pytorch_lightning as pl
import yaml
import importlib


class Generic_trainings(pl.LightningModule):
    def load_model_from_config(self, config):
        """
        Loads a model instance dynamically based on the provided configuration.

        Parameters:
        - config (dict): A dictionary containing the configuration details for the model.
            - "model": A dictionary with the following keys:
                - "module" (str): The name of the module containing the model class.
                - "class" (str): The name of the model class to instantiate.
                - "parameters" (dict, optional): A dictionary of parameters to pass to the model's constructor. If not provided, an empty dictionary will be used.

        Returns:
        - model_class: An instance of the model class, created with the specified parameters.

        Raises:
        - ImportError: If the module cannot be imported.
        - AttributeError: If the class cannot be found within the module.
        - KeyError: If the "model" key or required sub-keys ("module", "class") are missing from the configuration.

        This function dynamically imports a model module, retrieves the model class by name,
        and creates an instance of the class using parameters from the configuration dictionary.
        """
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
        """
        Dynamically loads a loss function based on the provided configuration.

        Parameters:
        - config (dict): A dictionary containing the configuration details for the loss function.
            - "training": A dictionary with the following keys:
                - "loss": A dictionary containing the details of the loss function.
                    - "module_loss_function" (str): The name of the module that contains the loss function.
                    - "class_loss_function" (str): The name of the loss function class.
                    - "parameters" (dict, optional): A dictionary of parameters to pass to the loss function's constructor. If not provided, an empty dictionary will be used.

        Returns:
        - loss_function (object): An instance of the loss function class, created with the specified parameters.

        Raises:
        - KeyError: If required keys ("training", "loss", "module_loss_function", "class_loss_function") are missing from the configuration.
        - ImportError: If the specified module cannot be imported.
        - AttributeError: If the specified class cannot be found within the module.
        - TypeError: If the parameters are not correctly passed to the loss function constructor.

        This function dynamically imports the module containing the loss function, retrieves the loss function class by name,
        and creates an instance of the class using parameters from the configuration dictionary. It ensures that all necessary
        configuration keys are present and valid before proceeding.
        """
        try:
            # Extract the loss configuration from the provided config
            loss_config = config["training"]["loss"]
            module_name = loss_config["module_loss_function"]
            class_name = loss_config["class_loss_function"]
            parameters = loss_config.get("parameters", {})

            # Dynamically import the module and retrieve the loss function class
            module = importlib.import_module(module_name)
            loss_function_class = getattr(module, class_name)

            # Instantiate and return the loss function with the given parameters
            return loss_function_class(**parameters)

        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e.args[0]}") from e
        except ImportError as e:
            raise ImportError(f"Could not import module: {module_name}") from e
        except AttributeError as e:
            raise AttributeError(
                f"Could not find class {class_name} in module {module_name}"
            ) from e
        except TypeError as e:
            raise TypeError(
                f"Error when passing parameters to {class_name}: {e}"
            ) from e

    def get_optimizer(self, config):
        """
        Dynamically loads an optimizer based on the provided configuration.

        Parameters:
        - config (dict): A dictionary containing the configuration details for the optimizer.
            - "training": A dictionary with the following keys:
                - "optimizer": A dictionary containing the details of the optimizer.
                    - "module_optimizer" (str): The name of the module that contains the optimizer class.
                    - "class_optimizer" (str): The name of the optimizer class.
                    - "parameters" (dict, optional): A dictionary of parameters to pass to the optimizer's constructor. If not provided, an empty dictionary will be used.

        Returns:
        - optimizer (object): An instance of the optimizer class, created with the model's parameters and specified parameters.

        Raises:
        - KeyError: If required keys ("training", "optimizer", "module_optimizer", "class_optimizer") are missing from the configuration.
        - ImportError: If the specified module cannot be imported.
        - AttributeError: If the specified optimizer class cannot be found within the module.
        - TypeError: If the parameters are not correctly passed to the optimizer constructor.

        This function dynamically imports the module containing the optimizer, retrieves the optimizer class by name,
        and creates an instance of the class using the model's parameters and the configuration parameters.
        """

        try:
            # Extract the optimizer configuration from the provided config
            optimizer_config = config["training"]["optimizer"]
            module_name = optimizer_config["module_optimizer"]
            class_name = optimizer_config["class_optimizer"]
            parameters = optimizer_config.get("parameters", {})

            # Dynamically import the module and retrieve the optimizer class
            module = importlib.import_module(module_name)
            optimizer_class = getattr(module, class_name)

            # Instantiate and return the optimizer with the model's parameters and given parameters
            return optimizer_class(self.model.parameters(), **parameters)

        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e.args[0]}") from e
        except ImportError as e:
            raise ImportError(f"Could not import module: {module_name}") from e
        except AttributeError as e:
            raise AttributeError(
                f"Could not find class {class_name} in module {module_name}"
            ) from e
        except TypeError as e:
            raise TypeError(
                f"Error when passing parameters to {class_name}: {e}"
            ) from e

    def load_metrics_from_config(self, config):
        """
        Dynamically loads metrics from the provided configuration.

        Parameters:
        - config (dict): A dictionary containing the configuration details for the metrics.
            - "metrics" (list): A list of dictionaries, each containing the details of a metric.
                - "name" (str): The name of the metric.
                - "module" (str): The name of the module containing the metric function.
                - "class" (str): The name of the metric function.
                - "parameters" (dict, optional): A dictionary of parameters to pass to the metric function's constructor.

        Returns:
        - list: A list of dictionaries, each containing:
            - "name" (str): The name of the metric.
            - "func" (function): The loaded metric function.
            - "params" (dict): The parameters for the metric function.

        Raises:
        - KeyError: If the "metrics" key is missing or malformed in the configuration.
        - ImportError: If a module cannot be imported.
        - AttributeError: If the metric function cannot be found within the module.
        - TypeError: If parameters are not correctly passed to the metric function.

        This function dynamically imports the module containing each metric function,
        retrieves the function by name, and returns a list of metrics with their names,
        functions, and parameters.
        """
        metrics = []
        # Verificar si "metrics" está definido y no está vacío.
        if "metrics" not in config or not config["metrics"]:
            return metrics

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
        """
    Initializes the training configuration by loading model, loss function, optimizer, and metrics from a YAML configuration file.

    Parameters:
    - conf_path (str): Path to the YAML configuration file containing model, loss function, optimizer, and metrics settings.
        """
        with open(conf_path, "r") as file:
            config = yaml.safe_load(file)
        self.model = self.load_model_from_config(config)
        self.loss_function = self.get_loss_function(config)
        self.optimizer = self.get_optimizer(config)
        self.metrics = self.load_metrics_from_config(config)

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Parameters:
        - x (tensor): Input tensor to the model.

        Returns:
        - tensor: The output of the model after the forward pass.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step, computes the loss, and logs the result.

        Parameters:
        - batch (tuple): A tuple containing the input data (x) and the corresponding labels (y).
            - x (tensor): The input tensor.
            - y (tensor): The ground truth labels.
        - batch_idx (int): The index of the current batch in the training loop.

        Returns:
        - tensor: The computed loss for the current batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs the validation step, computes the loss, evaluates metrics, and logs the results.

        Parameters:
        - batch (tuple): A tuple containing the input data (x) and the corresponding labels (y).
            - x (tensor): The input tensor.
            - y (tensor): The ground truth labels.
        - batch_idx (int): The index of the current batch in the validation loop.

        Returns:
        - tensor: The computed loss for the current batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric in self.metrics:
            metric_value = metric["func"](logits, y, **metric["params"])
            self.log(f"val_{metric['name']}", metric_value)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Performs the testing step, computes the loss, evaluates metrics, and logs the results.

        Parameters:
        - batch (tuple): A tuple containing the input data (x) and the corresponding labels (y).
        - batch_idx (int): The index of the current batch in the testing loop.
        Returns:
        - tensor: The computed loss for the current batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        for metric in self.metrics:
            metric_value = metric["func"](logits, y, **metric["params"])
            self.log(f"test_{metric['name']}", metric_value)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        Returns:
        - optimizer (object): An instance of the optimizer class, created with the model's parameters.
        """
        return self.optimizer
