from options.test_options import TestOptions
from models import create_model
from data import create_dataset
import time


if __name__ == "__main__":
    opt = TestOptions().parse()
    test_dataset = create_dataset(opt, mode="test")
    test_dataset_size = len(test_dataset)
    print('The number of test images = %d' % test_dataset_size)

    model = create_model(opt)
    model.setup()
    model.net.eval()

    correct = 0
    for data, label, idx, rot, zoom_factor in test_dataset:
        model.set_input(data, label)
        correct += 1 if model.test(test_dataset,
                                   idx, rot, zoom_factor) else 0

    accuracy = correct/test_dataset_size
    print(f'Accuracy: {accuracy:.5f}')
