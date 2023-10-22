import argparse
import os


class BaseOptions():
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument("--dataroot", required=False, help="path to data")
        parser.add_argument("--dataset_name", type=str,
                            default="cornell", help="name of dataset")
        parser.add_argument("--shuffle", type=int,
                            default=1, help="shuffle dataset")
        parser.add_argument("--start", type=float,
                            default=0.0, help="start of dataset")
        parser.add_argument("--end", type=float, default=0.9,
                            help="end of dataset")
        parser.add_argument("--train_test_split", type=float,
                            default=0.9, help="train test split")
        parser.add_argument("--train_val_split", type=float,
                            default=0.9, help="train val split")
        parser.add_argument("--ds_rotate", type=float,
                            default=0.0, help="weather dataset rotation")
        parser.add_argument("--random_rotate", type=int,
                            default=1, help="weather random rotate")
        parser.add_argument("--random_zoom", type=int,
                            default=1, help="weather random zoom")
        parser.add_argument("--output_size", type=int,
                            default=300, help="output size")

        parser.add_argument("--num_threads", type=int, default=4,
                            help="number of threads for data loader")
        parser.add_argument("--checkpoints_dir", type=str,
                            default="./checkpoints", help="models are saved here")
        parser.add_argument(
            "--name", type=str, default="copyright@1st.", help="the name of experience")
        parser.add_argument("--enable_depth", type=bool,
                            default=1, help="weather enable depth image")
        parser.add_argument("--enable_rgb", type=bool,
                            default=0, help="weather enable rgb image")
        parser.add_argument("--gpu_ids", default="-1", help="use -1 for cpu")
        parser.add_argument("--model", type=str,
                            default="ggcnn", choices=["ggcnn", "ggcnn2", "grcnn", "grcnn2", "grcnn3", "grcnn4", "tcnn"], help="the name of model")

        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:<25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if opt.isTrain:
            save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = os.path.join(save_dir, f'{opt.name}_opt.txt')
            with open(file_name, 'wt') as f:
                f.write(message)
                f.write('\n')

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        opt = parser.parse_args()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        self.opt = opt
        return self.opt
