from .base_options import BaseOptions


class VisualOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument("--data", type=str, help="path to visual data")
        parser.add_argument("--model_path", type=str, help="path to model")
        self.isTrain = False
        return parser
