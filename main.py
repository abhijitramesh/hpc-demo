from utils.data_utils import get_fashion_mnist_dataset, get_dataloader
from utils.logger_utils import get_logger
from utils.model_utils import get_device, get_model
from utils.loss_utils import get_loss_fn
from utils.optimizer_utils import get_optimizer
from utils.train_utils import train
from utils.test_utils import test
from utils.prediction_utils import prediction
import torch

logger = get_logger()

def run_train(config):
    ############ Setting up configuration ############
    logger.info("Setting up configuration...")
    batch_size = config["batch_size"]
    loss_fn = config["loss_fn"]
    lr = config["lr"]
    optimizer = config["optimizer"]
    epochs = config["epochs"]
    save_path = config["save_path"]
    logger.info("Configuration set.")
    ################################################

    ############ Fetching Dataset ################
    logger.info("Featching Fashion-MNIST dataset...")
    train_dataset = get_fashion_mnist_dataset(train=True)
    test_dataset = get_fashion_mnist_dataset(train=False)
    logger.info("Featched Fashion-MNIST dataset.")
    logger.info(f"Length of training dataset: {len(train_dataset)}")
    logger.info(f"Length of test dataset: {len(test_dataset)}")

    ############ Creating Dataloader ################
    logger.info("Creating dataloader...")
    train_dataloader = get_dataloader(dataset=train_dataset, batch_size=batch_size)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size)
    logger.info("Dataloader created.")
    ################################################

    ############ Creating Model ################
    logger.info("Fetching device...")
    device = get_device()
    logger.info("Device fetched.")

    logger.info("Creating model...")
    model = get_model(device)
    logger.info("Model created.")
    ################################################

    ############ Loss and Optimizer ################
    loss_fn = get_loss_fn(loss_fn=loss_fn)
    optimizer = get_optimizer(optimizer=optimizer, model=model, lr=lr)
    ################################################

    ############ Training ################
    logger.info("Training...")
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train(dataloader=train_dataloader, model=model, device=device, loss_fn=loss_fn, optimizer=optimizer)
        test(dataloader=test_dataloader, model=model, device=device, loss_fn=loss_fn)
    logger.info("Training completed.")
    logger.info("Saving model...")
    torch.save(model.state_dict(), save_path)
    logger.info("Model saved.")

def run_prediction(config):
    save_path = config["save_path"]
    
    logger.info("Fetching device...")
    device = get_device()
    logger.info("Device fetched.")
    
    logger.info("Creating model...")
    model = get_model(device)
    logger.info("Model created.")
    
    logger.info("Loading model...")
    model.load_state_dict(torch.load(save_path))
    logger.info("Model loaded.")
    
    logger.info("Predicting...")
    model.eval()
    test_data = get_fashion_mnist_dataset(train=False)
    prediction(test_data=test_data, model=model, device=device)
    logger.info("Prediction completed.")

if __name__ == "__main__":
    config = {
        "batch_size": 64,
        "loss_fn": "cross_entropy",
        "optimizer": "adam",
        "lr": 1e-3,
        "epochs": 5,
        "save_path": "model.pth"
    }
    # run_train(config)
    run_prediction(config)
