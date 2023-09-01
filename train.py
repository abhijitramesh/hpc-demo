from utils.data_utils import get_fashion_mnist_dataset, get_dataloader
from utils.logger_utils import get_logger
from utils.model_utils import get_device, get_model
from utils.loss_utils import get_loss_fn
from utils.optimizer_utils import get_optimizer

logger = get_logger()

def run_train(config):
    ############ Setting up configuration ############
    logger.info("Setting up configuration...")
    batch_size = config["batch_size"]
    loss_fn = config["loss_fn"]
    lr = config["lr"]
    optimizer = config["optimizer"]
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
if __name__ == "__main__":
    config = {
        "batch_size": 64,
        "loss_fn": "cross_entropy",
        "optimizer": "adam",
        "lr": 1e-3,
    }
    run_train(config)