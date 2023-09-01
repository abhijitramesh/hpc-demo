from .logger_utils import get_logger
logger = get_logger()

def train(*, dataloader, model, device, loss_fn, optimizer):
    """Train the model.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader.
        model (torch.nn.Module): Model.
        device (str): Device (cpu or gpu or mps)
        loss_fn (torch.nn.modules.loss._Loss): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")