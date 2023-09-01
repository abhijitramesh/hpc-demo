import logging

def get_logger():
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('FashionMNIST_Training')
    return logger