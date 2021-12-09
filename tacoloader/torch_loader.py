"""
Author: Adrish Dey (adrish@wandb.com)
Lightweight Dataset Class for TACO Trash Annotation Dataset

Based on: https://github.com/pedropro/TACO/
"""
import collections
import functools
import json
import pathlib

import PIL
import torch
import torchvision
from pycocotools import coco


def dummy_wrapper(function):
    """
    Dummy decorator function
    """
    return function


BatchedResult = collections.namedtuple(
    "BatchedResult",
    ["images", "labels", "bboxes", "masks", "backgrounds"],
)
Result = collections.namedtuple(
    "Result", ["image", "label", "bbox", "mask", "backgrounds"]
)


class TacoDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, transform_fn=None, cache_fn=None, ann_file=None):
        """
        Initiating Trash Context Annotation for Dataset Class
        Args:
            datadir (`str`): path to data directory.

            transform_fn (`callable` or `None`): function accepting
            image tensor and returning image tensor.

            cache_fn (`callable` or `None`): decorator type wrapper function
            for caching IO operations.

            ann_file (`str`): path to `annotations.json` file in COCO format.
        """
        datadir = pathlib.Path(datadir)
        _cache_fn = cache_fn or dummy_wrapper

        if not ann_file:
            ann_file = datadir / "annotations.json"
        with open(ann_file, "r") as f:
            manifest = json.load(f)

        self.datadir = datadir
        self._cocoobj = coco.COCO()
        self._cocoobj.dataset = manifest
        self.load_image_fn = _cache_fn(TacoDataset.load_image)
        self._cocoobj.createIndex()

        self.imgid2bgids = dict()
        for x in manifest["scene_annotations"]:
            self.imgid2bgids[x["image_id"]] = x["background_ids"]
        self._bgs = manifest["scene_categories"]

        self.len_categories = max(manifest["categories"], key=lambda x: x["id"])["id"]
        self.len_categories += 1
        self.len_background = max(self._bgs, key=lambda x: x["id"])["id"]
        self.len_background += 1
        self._load_fn = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
            (0.495, 0.468, 0.424), # mean of image channels in TACO
            (0.2513, 0.2389, 0.2390) # std of image channels in TACO
            )])
        self._transform_fn = transform_fn
        self._idx2key = self._cocoobj.getImgIds()
        self._catkey2idx = {k: i for i, k in enumerate(self._cocoobj.getCatIds())}

    def __len__(self):
        return len(self._idx2key)

    @staticmethod
    def load_image(path, transform_fn, to_tensor_fn):
        """
        Loads Image and apply transforms Given the path.
        Args:
            path (`str` or `pathlib.Path`): path to image file.

            transform_fn (`callable`): takes an image tensor
            and apply preprocessing steps.

            to_tensor_fn (`callable`): take a pillow image
            and converts it to `torch.Tensor`.
        """
        image = PIL.Image.open(path)
        exif = image._getexif()
        if exif:
            exif = dict(exif.items())
            # Rotate portrait images if necessary (274 is the orientation tag code)
            if 274 in exif:
                if exif[274] == 3:
                    image = image.rotate(180, expand=True)
                if exif[274] == 6:
                    image = image.rotate(270, expand=True)
                if exif[274] == 8:
                    image = image.rotate(90, expand=True)
        width, height = image.size
        rotmask = False
        if height > width:
            image = image.rotate(90, expand=True)
            rotmask = True
        image = transform_fn(to_tensor_fn(image))
        image.rotmask = rotmask
        return image

    def get_categories(self, cat_ids=[], with_background=True):
        """
        Return Category Names given Category IDs.
        Args:
            cat_ids (`iterable`): collection of all category ids.

            with_background (`boolean`): consider `0` in the segmentation mask.
            as generic background pixels.
        """
        cat_names = []
        print(with_background)
        for cat_id in cat_ids:
            cat_id -= int(with_background)
            if cat_id == -1:
                name = "background"
            else:
                name = self._cocoobj.loadCats(cat_id)[0]["name"]
            cat_names.append(name)
        return cat_names

    def get_backgrounds(self, bgids=[], imgids=[], names=True):
        """
        Returns background information given background ID or Image ID.
        Args:
            bgids (`list`): List of background IDs.

            imgids (`list`): List of Image IDs.

            names (`boolean`): Return only the names instead of background
            dictionary.
        """
        if not (bgids or imgids):
            return self._bgs

        def get_names_fn(bg):
            return bg["name"] if names else bg

        bgids = set(bgids.extend([self.imgids2bgids[i] for i in imgids]))
        return [get_names_fn(bg) for bg in self._bgs if bg["id"] in bgids]

    @staticmethod
    def collate_fn(batch):
        """
        Collate multiple data item into
        an easily accessible batch
        Args:
            batch (`list`): list of items to
            collect in a batch
        """
        images = []
        categories = []
        bboxes = []
        segmentations = []
        backgrounds = []
        for elem in batch:
            images.append(elem.image)
            categories.append(elem.label)
            bboxes.append(elem.bbox)
            segmentations.append(elem.mask)
            backgrounds.append(elem.backgrounds)
        return BatchedResult(
            images=torch.stack(images),
            labels=categories,
            bboxes=bboxes,
            masks=torch.stack(segmentations),
            backgrounds=backgrounds,
        )

    def __getitem__(self, idx):
        """
        Returns a data item given an image index.
        Args:
            idx (`int`): index of Image to load.
        """
        idx = self._idx2key[idx]
        filename = self._cocoobj.loadImgs(idx)[0]["file_name"]
        path = self.datadir / filename
        img = self.load_image_fn(path, self._transform_fn, self._load_fn)
        bgs = self.imgid2bgids.get(idx, [])
        ann_ids = self._cocoobj.getAnnIds(imgIds=idx)

        cat_ids = []
        bboxes = []
        segms = []

        segms = torch.zeros((*img.shape[1:],))
        for i in ann_ids:
            ann = self._cocoobj.anns[i]
            cat_id = ann["category_id"]
            cat_ids.append(cat_id)

            segm = self._cocoobj.annToMask(ann)
            segm = torch.from_numpy(segm).unsqueeze(0)
            rotmask = getattr(img, "rotmask", None)

            if rotmask:
                segm = torch.rot90(segm, 1, [1, 2])

            if self._transform_fn and callable(self._transform_fn):
                segm = self._transform_fn(segm)

            segms = torch.maximum(segm * (self._catkey2idx[cat_id]), segms)

            bboxes.append(torch.tensor(ann["bbox"]))

        return Result(
            image=img,
            label=torch.tensor(cat_ids),
            bbox=torch.stack(bboxes),
            mask=segms.to(torch.uint8).squeeze(),
            backgrounds=torch.tensor(bgs),
        )
