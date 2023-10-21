from data.base_dataset import BaseDataset
import os
import glob
from utils import grasp, image


class CornellDataset(BaseDataset):

    def __init__(self, opt, mode="train"):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        # file_path, start=0.0, end=1.0, ds_rotate=0,
        BaseDataset.__init__(self, opt)
        graspf = glob.glob(os.path.join(self.opt.root, '*', 'pcd*cpos.txt'))
        graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError(
                'No dataset files found. Check path: {}'.format(self.opt.root))

        if self.opt.ds_rotate:
            graspf = graspf[int(l*self.opt.ds_rotate):] + \
                graspf[:int(l*self.opt.ds_rotate)]

        depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
        rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]

        if mode == "train":
            self.grasp_files = graspf[int(
                l*self.opt.start):int(l*self.opt.end*0.8)]
            self.depth_files = depthf[int(
                l*self.opt.start):int(l*self.opt.end*0.8)]
            self.rgb_files = rgbf[int(l*self.opt.start)
                                      :int(l*self.opt.end*0.8)]
        elif mode == "val":
            self.grasp_files = graspf[int(
                l*self.opt.end*0.8):int(l*self.opt.end)]
            self.depth_files = depthf[int(
                l*self.opt.end*0.8):int(l*self.opt.end)]
            self.rgb_files = rgbf[int(l*self.opt.end*0.8):int(l*self.opt.end)]

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(
            self.grasp_files[idx])
        center = gtbbs.center
        left = max(
            0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size //
                  2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(
            self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(
            480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size),
                     min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def __len__(self):
        return len(self.grasp_files)
