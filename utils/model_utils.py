from .logger_utils import get_logger
from .neural_network import NeuralNetwork
import torch

logger = get_logger()

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    return device


def get_model(device):
    model = NeuralNetwork().to(device)
    logger.info(f"Model: {model}")
    return model