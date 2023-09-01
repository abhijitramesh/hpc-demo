from .logger_utils import get_logger
import torch

logger = get_logger()
def get_optimizer(*, optimizer, model, lr):
    """Get optimizer.

    Args:
        optimizer (str): Optimizer name.

    Returns:
        torch.optim.Optimizer: Optimizer.
    """
    logger.debug(f"Getting optimizer: {optimizer}")
    if optimizer == "sgd":
        logger.info("Using SGD.")
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "adam":
        logger.info("Using Adam.")
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        logger.error(f"Optimizer {optimizer} not implemented.")
        raise NotImplementedError(f"Optimizer {optimizer} not implemented.")