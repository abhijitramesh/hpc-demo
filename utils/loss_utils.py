import torch.nn
from .logger_utils import get_logger
logger = get_logger()


def get_loss_fn(loss_fn):
    """Get loss function.

    Args:
        loss_fn (str): Loss function name.

    Returns:
        torch.nn.modules.loss._Loss: Loss function.
    """
    logger.debug(f"Getting loss function: {loss_fn}")
    if loss_fn == "cross_entropy":
        logger.info("Using CrossEntropyLoss.")
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == "nll_loss":
        logger.info("Using NLLLoss.")
        return torch.nn.NLLLoss()
    else:
        logger.error(f"Loss function {loss_fn} not implemented.")
        raise NotImplementedError(f"Loss function {loss_fn} not implemented.")