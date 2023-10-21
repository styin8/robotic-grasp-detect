from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument("--results_dir", type=str,
                            default="./results/", help="path to results")

        self.isTrain = False
        return parser
