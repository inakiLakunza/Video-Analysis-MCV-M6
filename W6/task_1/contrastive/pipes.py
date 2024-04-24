
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score


from typing import *



from utils import model_analysis
from utils import statistics


device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def get_all_embeddings(dataset, model, transforms):
    
    model.eval()
    embeddings = None
    labels = None
    for i in tqdm(range(len(dataset))):
        clips, lab, path = dataset[i]
        clips = torch.cat([transforms(clip).unsqueeze(0).permute(0, 2, 1, 3, 4) for clip in clips], dim=0).to(device)

        feat = model(clips)
        feat = torch.mean(feat, dim=0).unsqueeze(0)
        
        if embeddings is None:
            embeddings = feat
            labels = torch.tensor([lab[0]])
        
        else:
            embeddings = torch.cat((embeddings, feat), dim=0)
            labels = torch.cat((labels, torch.tensor([lab[0]])), dim=0)
                  
    return embeddings, labels.to(device)



def create_optimizer(optimizer_name: str, parameters: Iterator[nn.Parameter], lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Creates an optimizer for the given parameters.
    
    Args:
        optimizer_name (str): Name of the optimizer (supported: "adam" and "sgd" for now).
        parameters (Iterator[nn.Parameter]): Iterator over model parameters.
        lr (float, optional): Learning rate. Defaults to 1e-4.

    Returns:
        torch.optim.Optimizer: The optimizer for the model parameters.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")




def create_datasets(
        dataset,
        frames_dir: str,
        annotations_dir: str,        
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        n_segments: int
) -> Dict[str, type[Dataset]]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (HMDB51Dataset.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.

    Returns:
        Dict[str, HMDB51Dataset]: A dictionary containing the datasets for training, validation, and testing.
    """
    
    REGIME = dataset.Regime
    SPLIT = dataset.Split.TEST_ON_SPLIT_1
    
    datasets = {}
    for regime in REGIME:
        datasets[regime.name.lower()] = dataset(
            frames_dir,
            annotations_dir,
            SPLIT,
            regime,
            clip_length,
            crop_size,
            temporal_stride,
            n_segments
        )
    
    return datasets



def print_model_summary(
        model: nn.Module,
        clip_length: int,
        crop_size: int,
        print_model: bool = True,
        print_params: bool = True,
        print_FLOPs: bool = True,
        params: list = None
    ) -> None:
    """
    Prints a summary of the given model.

    Args:
        model (nn.Module): The model for which to print the summary.
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        print_model (bool, optional): Whether to print the model architecture. Defaults to True.
        print_params (bool, optional): Whether to print the number of parameters. Defaults to True.
        print_FLOPs (bool, optional): Whether to print the number of FLOPs. Defaults to True.

    Returns:
        None
    """
    if print_model:
        print(model)

    if print_params:
        if params is None:
            num_params = sum(p.numel() for p in model.parameters())
        else:
            num_params = sum(p.numel() for p in params)

        #num_params = model_analysis.calculate_parameters(model) # should be equivalent
        print(f"Number of parameters (M): {round(num_params / 10e6, 2)}")

    if print_FLOPs:
        num_FLOPs = model_analysis.calculate_operations(model, clip_length, crop_size, crop_size)
        print(f"Number of FLOPs (G): {round(num_FLOPs / 10e9, 2)}")


def create_dataloaders(
        datasets: Dict[str, Type[Dataset]],
        batch_size: int,
        batch_size_eval: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True
    ) -> Dict[str, DataLoader]:
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        datasets (Dict[str, HMDB51Dataset]): A dictionary containing datasets for training, validation, and testing.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 8.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory in DataLoader for faster GPU transfer. Defaults to True.

    Returns:
        Dict[str, DataLoader]: A dictionary containing data loaders for training, validation, and testing datasets.
    """
    dataloaders = {}
    for key, dataset in datasets.items():
        dataloaders[key] = DataLoader(
            dataset,
            batch_size=(batch_size if key == 'training' else batch_size_eval),
            shuffle=(key == 'training'),  # Shuffle only for training dataset
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
            
    return dataloaders



def accuracies_por_clases(modelo, X_test, y_test):
    """
    Calcula los accuracies por clases para un modelo de regresión logística.

    Args:
    - modelo: El modelo de regresión logística entrenado.
    - X_test: Los datos de prueba.
    - y_test: Las etiquetas de clase correspondientes a los datos de prueba.

    Returns:
    - accuracies: Un diccionario que contiene los accuracies por clases.
    """
    # Predecir las clases para los datos de prueba
    y_pred = modelo.predict(X_test)
    
    # Obtener las clases únicas en los datos de prueba
    clases = set(y_test)
    
    # Inicializar el diccionario de accuracies por clases
    accuracies = {}
    
    # Calcular el accuracy para cada clase
    for clase in clases:
        # Obtener las muestras pertenecientes a la clase actual
        indices_clase = (y_test == clase)
        # Calcular el accuracy para la clase actual
        accuracy = accuracy_score(y_test[indices_clase], y_pred[indices_clase])
        accuracies[clase] = accuracy
    
    return accuracies