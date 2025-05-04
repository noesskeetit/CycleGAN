from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import io

class VangoghPhotoDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Читаем байты и превращаем в PIL.Image
        vangogh_img = Image.open(io.BytesIO(sample["imageA"]["bytes"])).convert("RGB")
        photo_img = Image.open(io.BytesIO(sample["imageB"]["bytes"])).convert("RGB")

        # Переводим в np.array для albumentations
        vangogh_img = np.array(vangogh_img)
        photo_img = np.array(photo_img)

        if self.transform:
            augmented = self.transform(image=vangogh_img, image0=photo_img)
            vangogh_img = augmented["image"]
            photo_img = augmented["image0"]

        return vangogh_img, photo_img