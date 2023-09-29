from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
from typing import Optional
import os
import torch

def _get_cutout_holes(
    height,
    width,
    min_holes=8,
    max_holes=32,
    min_height=48,
    max_height=128,
    min_width=48,
    max_width=128,
):
    holes = []
    for _n in range(random.randint(min_holes, max_holes)):
        hole_height = random.randint(min_height, max_height)
        hole_width = random.randint(min_width, max_width)
        y1 = random.randint(0, height - hole_height)
        x1 = random.randint(0, width - hole_width)
        y2 = y1 + hole_height
        x2 = x1 + hole_width
        holes.append((x1, y1, x2, y2))
    return holes


def _generate_random_mask(image):
    mask = torch.zeros_like(image[:1])
    print(mask.shape)
    holes = _get_cutout_holes(mask.shape[1], mask.shape[2])
    for (x1, y1, x2, y2) in holes:
        mask[:, y1:y2, x1:x2] = 1.0
    masked_image = image * (mask < 0.5)
    return mask, masked_image



class SFTInpaintDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        global_caption: Optional[str] = None,
        min_size=512,
        max_size=768,
        resize=True,
        normalize=True,
        noise_offset: float = 0.1,
        train_inpainting: bool = True,
    ):  
        
        self.resize = resize
        self.min_size = min_size
        self.max_size = max_size
        
        self.tokenizer = tokenizer
        self.train_inpainting = train_inpainting
        instance_data_root = Path(instance_data_root)

        if not Path(instance_data_root).exists():
            raise ValueError("Instance images root doesn't exists.")
        img_path = os.path.join(instance_data_root, "images")
        captions_path = os.path.join(instance_data_root, "captions")

        if not Path(img_path).exists():
            raise ValueError("Instance images root doesn't exists.")
        if not Path(captions_path).exists():
            raise ValueError("Instance captions root doesn't exists.")

        self.images_path = []
        self.captions_path = []

        for root, dirs, files in os.walk(img_path):
            for file in files:
                self.images_path.append(os.path.join(root, file))
                self.captions_path.append(os.path.join(captions_path, file.split('.')[0] + '.txt'))
        
        self.images_path.sort()
        self.captions_path.sort()

        self.num_instance_images = len(self.images_path)

        self.global_caption = global_caption 
        self._length = len(self.images_path)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size=self.min_size, 
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    max_size=self.max_size, 
                ) # the image size is as maximum 768x512
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                if normalize
                else transforms.Lambda(lambda x: x),
            ]
        )

        self.noise_offset = noise_offset

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        """ 
        Returns a dictionary with the following keys:
            instance_images: The instance image
            instance_masks: The mask for the instance image
            instance_masked_images: The masked instance image (image with the mask applied)
            instance_masked_values: The masked instance image values (values of the mask)
            instance_captions: The instance captions
            instance_prompt_ids: The tokenized instance captions
        """
        example = {}
        instance_image = Image.open(
            self.images_path[index % self.num_instance_images]
        )

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        
        example["instance_images"] = self.image_transforms(instance_image)
        
        (
            example["instance_masks"],
            example["instance_masked_images"],
        ) = _generate_random_mask(example["instance_images"])

        # instance_masked_values: The values of the masked instance image
        example["instance_masked_values"] = (
            example["instance_images"] * (1 - example["instance_masks"])
        )

        with open(self.captions_path[index % self.num_instance_images], "r") as f:
            caption = f.read().strip()
        text = caption
        if self.global_caption:
            text = caption + ", " + self.global_caption.strip()
        
        example["instance_captions"] = text

        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example