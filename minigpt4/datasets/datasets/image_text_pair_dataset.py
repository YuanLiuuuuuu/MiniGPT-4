import cv2
from PIL import Image
import numpy as np
import os
from .base_dataset import BaseDataset


class CC3MDataset(BaseDataset):
    def __getitem__(self, index) -> dict:
        ann = self.annotation[index]
        image_path = ann['image']
        image_path = image_path.split("/")[-2:]
        image_path = os.path.join(self.vis_root, *image_path)
        caption = ann['caption']

        # process image and caption
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        caption = self.text_processor(caption)

        return {"image": image, "text_input": caption}

    def _add_instance_ids(self, key="instance_id"):
        if isinstance(self.annotation[0], dict):
            for idx, ann in enumerate(self.annotation):
                ann[key] = str(idx)
        elif isinstance(self.annotation[0], list):
            for idx, ann in enumerate(self.annotation):
                cur_ann = dict()
                cur_ann["instance_id"] = str(idx)
                cur_ann["image"] = ann[0]
                cur_ann["caption"] = ann[1]
                self.annotation[idx] = cur_ann


class CC12MDataset(BaseDataset):
    from petrel_client.client import Client
    client = Client()

    def __getitem__(self, index) -> dict:
        ann = self.annotation[index]
        image_path = ann['image']
        caption = ann['caption']
        try:
            img_bytes = self.client.get(image_path)
            img_mem_view = memoryview(img_bytes)
            img_array = np.frombuffer(img_mem_view, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            image = Image.fromarray(img)
        except Exception as e:  # noqa
            print(f"Meet corrupted image: {image_path}")
            random_image = np.random.randint(0, 255, (224, 224, 3))
            random_image = random_image.astype(np.uint8)
            image = Image.fromarray(random_image)

        image = self.vis_processor(image)
        caption = self.text_processor(caption)

        return {"image": image, "text_input": caption}

    def _add_instance_ids(self, key="instance_id"):
        if isinstance(self.annotation[0], dict):
            for idx, ann in enumerate(self.annotation):
                ann[key] = str(idx)
        elif isinstance(self.annotation[0], list):
            for idx, ann in enumerate(self.annotation):
                cur_ann = dict()
                cur_ann["instance_id"] = str(idx)
                cur_ann["image"] = ann[0]
                cur_ann["caption"] = ann[1]
                self.annotation[idx] = cur_ann