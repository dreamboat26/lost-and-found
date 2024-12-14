import os
import glob
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DehazingImageDataset(Dataset):
    def __init__(self, dataset_dir, sample_size=None, random_flip=False, is_train=False):
        """
        Args:
            dataset_dir (string): Directory with all the images, structured as subfolders per class.
            is_train (bool, optional): Optional boolean flag to indicate whether
                to generate data for training or testing, Defaults to False.
        """
        self.random_flip = random_flip
        self.sample_size = sample_size
        self.root_dir = dataset_dir
        assert os.path.isdir(self.root_dir)
        self.hazy_dir = os.path.join(self.root_dir, "hazy")
        assert os.path.isdir(self.hazy_dir)
        self.gt_dir = os.path.join(self.root_dir, "GT")
        assert os.path.isdir(self.gt_dir)

        self.haze_images = glob.glob(self.hazy_dir + "/*.png")
        self.gt_images = glob.glob(self.gt_dir + "/*.png")
        
        self.img_paths = []  # List of tuples (img_path, label)
        for gt_img_path in self.gt_images:
            hazy_img_name = os.path.basename(gt_img_path).replace("GT", "hazy")
            hazy_img_path = os.path.join(self.hazy_dir, hazy_img_name)
            self.img_paths.append([gt_img_path, hazy_img_path])
        random.shuffle(self.img_paths)
        
        self.is_train = is_train
        self.classes = [0, 1]       # 0 : Fake, 1 : Real
        self.class_to_idx = {"Fake": 0, "Real": 1}

    def __len__(self):
        return len(self.img_paths)

    def random_crop_coords(self, image):
        """Generates random crop coordinates for an image."""
        height, width = image.shape[1], image.shape[2]
        crop_width, crop_height = self.sample_size
        if width < crop_width or height < crop_height:
            raise ValueError("Crop size must be smaller than the image size.")
        
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)
        return x, y, x + crop_width, y + crop_height

    def get_split_imgs(self, image):
        # image : [C, H, W]    
        h, w = image.shape[1], image.shape[2]
        return image[:, :, :w // 2], image[:, :, w // 2:]
    
    def __getitem__(self, idx):
        gt_img_path, hazy_img_path = self.img_paths[idx]
        gt_img = Image.open(gt_img_path).convert("RGB")
        hazy_img = Image.open(hazy_img_path).convert("RGB")

        gt_img = transforms.ToTensor()(gt_img)  # converts [H, W, C] into [C, H, W] and divides image by 255.
        gt_img = (gt_img * 2) - 1               # converts [0-1] into [-1, 1] 
        gt_img_A, gt_img_B = self.get_split_imgs(gt_img)
        
        x1_A, y1_A, x2_A, y2_A = self.random_crop_coords(gt_img_A)
        x1_B, y1_B, x2_B, y2_B = self.random_crop_coords(gt_img_B)

        gt_img_A = gt_img_A[:, y1_A:y2_A, x1_A:x2_A]
        gt_img_B = gt_img_B[:, y1_B:y2_B, x1_B:x2_B]
        
        hazy_img = transforms.ToTensor()(hazy_img)  # converts [H, W, C] into [C, H, W] and divides image by 255.
        hazy_img = (hazy_img * 2) - 1               # converts [0-1] into [-1, 1]
        hazy_img_A, hazy_img_B = self.get_split_imgs(hazy_img)

        hazy_img_A = hazy_img_A[:, y1_A:y2_A, x1_A:x2_A]
        hazy_img_B = hazy_img_B[:, y1_B:y2_B, x1_B:x2_B]
        
        if self.random_flip and random.random() > 0.5:
            gt_img_A = transforms.functional.hflip(gt_img_A)
            gt_img_B = transforms.functional.hflip(gt_img_B)
            hazy_img_A = transforms.functional.hflip(hazy_img_A)
            hazy_img_B = transforms.functional.hflip(hazy_img_B)
        
        return gt_img_A, gt_img_B, hazy_img_A, hazy_img_B


if __name__ == "__main__":
    import config
    dataset = DehazingImageDataset(config.dataset_path, sample_size=config.sample_size, random_flip=config.random_flip)
    
    def deprocess(image):
        image = (((image + 1) / 2) * 255).to(torch.uint8)
        return transforms.ToPILImage()(image)
    
    print("Dataset len: ", len(dataset))
    for data in dataset:
        gt_img_A, gt_img_B, hazy_img_A, hazy_img_B = data
        print("gt_img_A shape: ", gt_img_A.shape)
        print("gt_img_B shape: ", gt_img_B.shape)
        print("hazy_img_A shape: ", hazy_img_A.shape)
        print("hazy_img_B shape: ", hazy_img_B.shape)
        gt_img_A = deprocess(gt_img_A)
        gt_img_B = deprocess(gt_img_B)
        hazy_img_A = deprocess(hazy_img_A)
        hazy_img_B = deprocess(hazy_img_B)
        # gt_img_A.save("gt_img_A.png")
        # gt_img_B.save("gt_img_B.png")
        # hazy_img_A.save("hazy_img_A.png")
        # hazy_img_B.save("hazy_img_B.png")
        break
