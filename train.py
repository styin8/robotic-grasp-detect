from options.train_options import TrainOptions
from models import create_model
from data import create_dataset
import time

if __name__ == "__main__":
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup()

    for epoch in range(0, opt.epochs):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            total_loss = 0
            iter_start_time = time.time()
            model.set_input(data)
            model.optimize_parameters()
            total_loss += model.loss
            print(
                f'Epoch: {epoch}, Batch: {i}, Loss:{model.loss},Time: {time.time() - iter_start_time}')

        print(
            f'End of epoch {epoch} / {opt.epochs} \t Loss:{total_loss/(i+1):.2f},Time Taken: {time.time() - epoch_start_time} sec')

        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}')
            model.save_networks(epoch)
