import torch
import numpy as np
from torchvision import transforms
from coco_data import MyImage, RandomCrop
from model_definition import UNet
import torch.nn as nn
import pytorch_lightning as pl
import os
import sys
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger
from validation_metrics import compute_psnr, compute_mse
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))
from training_step import Generic_trainings


# Configurar el logger de WandB
wandb_logger = WandbLogger(
    project="Coco project",  # Cambia esto por el nombre de tu proyecto en WandB
    name="plot training",  # Nombre del experimento
    log_model=True,  # Opcional: guarda los modelos entrenados en WandB
)

# Set hyperparameters
n_epochs = 100
batch_size = 2
cropsize = 256
conf_path = "./test-denoising/config_training.yaml"

# Create dataset with transformations
train_data = MyImage(0, 200, transform=transforms.Compose([RandomCrop(cropsize)]))
val_data = MyImage(200, 300, transform=transforms.Compose([RandomCrop(cropsize)]))

# Create DataLoader for batching
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)

# Crear DataLoader para validación
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Crear un checkpoint callback para guardar los mejores modelos
checkpoint_callback = ModelCheckpoint(
    monitor="val_mse",  # Monitorear MSE
    dirpath="./checkpoints",  # Carpeta donde se guardará el modelo
    save_top_k=1,  # Guarda el mejor modelo
    mode="min",  # Guarda el mejor modelo según el valor de la métrica especificada
    filename="best_model",  # Nombre del archivo de guardado del mejor modelo
    verbose=True,  # Muestra información durante la ejecución del entrenamiento
    enable_version_counter=False,  # Sobreescriir la version del archivo de guardado
)

lit_model = Generic_trainings(conf_path)

# Verificar si la GPU está disponible
print(torch.cuda.is_available())  # Esto debe devolver True si tu GPU está disponible.

# Si CUDA está disponible, mover el modelo a la GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
lit_model.to(device)

# Train with PyTorch Lightning
trainer = pl.Trainer(
    accelerator="gpu",  # Use GPU
    devices=1,
    max_epochs=n_epochs,
    limit_train_batches=100,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
)

trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
