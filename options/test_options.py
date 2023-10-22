from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument("--results_dir", type=str,
                            default="./results/", help="path to results")
        parser.add_argument("--model_path", type=str, help="path to model")
        self.isTrain = False
        return parser
