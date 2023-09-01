import torch
from .logger_utils import get_logger
import matplotlib.pyplot as plt
import numpy as np
import itertools

logger = get_logger()
def prediction(*, test_data, model, device):
    """Predict on test data and generate confusion matrix.

    Args:
        test_data (torch.utils.data.Dataset): Test dataset.
        model (torch.nn.Module): Model.
        device (str): Device (cpu or gpu or mps)
    """
    num_classes = 10
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for X, y in test_data:
            X = X.unsqueeze(0).to(device)
            y = torch.tensor([y]).to(device)
            
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            
            for t, p in zip(y.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    plt.figure(figsize=(10,10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    logger.info('Confusion Matrix saved as confusion_matrix.png')
