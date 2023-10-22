from options.train_options import TrainOptions
from models import create_model
from data import create_dataset
import time
import tensorboardX
import os


if __name__ == "__main__":
    opt = TrainOptions().parse()
    tb = tensorboardX.SummaryWriter(
        os.path.join(opt.checkpoints_dir, opt.name, "tensorboard"))

    train_dataset = create_dataset(opt, mode="train")
    train_dataset_size = len(train_dataset)
    print('The number of training images = %d' % train_dataset_size)

    val_dataset = create_dataset(opt, mode="val")
    val_dataset_size = len(val_dataset)
    print('The number of val images = %d' % val_dataset_size)

    model = create_model(opt)
    model.setup()

    best_accuracy = 0
    for epoch in range(0, opt.epochs):
        model.net.train()
        epoch_start_time = time.time()
        i = 0
        total_loss = 0
        for data, label, _, _, _ in train_dataset:
            i += 1
            iter_start_time = time.time()
            model.set_input(data, label)
            model.optimize_parameters()

            total_loss += model.loss
            print(
                f'Epoch: {epoch}, Batch: {i}, Loss:{model.loss:.5f},Time: {time.time() - iter_start_time:.2f}')
        tb.add_scalar('train_loss', total_loss/i, epoch)
        print(
            f'End of epoch {epoch} / {opt.epochs} \t Loss:{total_loss/i:.5f},Time Taken: {time.time() - epoch_start_time:.2f} sec')

        print("Evaluating the model on validation data")
        model.net.eval()
        correct = 0
        for data, label, idx, rot, zoom_factor in val_dataset:
            model.set_input(data, label)
            correct += 1 if model.test(val_dataset,
                                       idx, rot, zoom_factor) else 0
        accuracy = correct/val_dataset_size
        print(f'Accuracy: {accuracy:.5f}')
        tb.add_scalar('Accuracy', accuracy, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f'Saving the best model at the end of epoch {epoch}')
            model.save_network(epoch, accuracy)
            continue

        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}')
            model.save_network(epoch, accuracy)
