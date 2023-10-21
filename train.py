from options.train_options import TrainOptions
from models import create_model

if __name__ == "__main__":
    opt = TrainOptions().parse()
    model = create_model(opt)
    model.setup()
