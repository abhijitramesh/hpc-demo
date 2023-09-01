from utils.data_utils import get_fashion_mnist_dataset, get_dataloader
from utils.logger_utils import get_logger
from utils.model_utils import get_device, get_model


logger = get_logger()

def run_train(config):
    ############ Setting up configuration ############
    logger.info("Setting up configuration...")
    batch_size = config["batch_size"]
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

if __name__ == "__main__":
    config = {
        "batch_size": 64,
    }
    run_train(config)