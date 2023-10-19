from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument("--lr", type=float, default=0.001,
                            help="initial learning rate")
        parser.add_argument("--dropout", type=float, default=0.5,
                            help="drop out rate")
        parser.add_argument("--epochs", type=int,
                            default=1000, help="number of epochs")
        parser.add_argument("--batch_size", type=int,
                            default=8, help="input batch size")
        parser.add_argument("--init_weight", type=bool,
                            default=True, help="initial weight")

        self.isTrain = True
        return parser
