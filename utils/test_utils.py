import torch
from .logger_utils import get_logger

logger = get_logger()

def test(dataloader, model, device, loss_fn):
    """
    Test the model.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader for the test set.
        model (torch.nn.Module): Model to be tested.
        device (str): Device (cpu or gpu or mps) on which to run the test.
        loss_fn (torch.nn.modules.loss._Loss): Loss function to evaluate the model.

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")