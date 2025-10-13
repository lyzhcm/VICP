import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import glob
import os
import pandas as pd
from PIL import Image
import torch

class PetFaceDataset(Dataset):
    def __init__(self, root_dir, class_names=None, transform=None):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform

        image_list = []
        for name in class_names:
            image_list.extend(pd.read_csv(os.path.join(root_dir, "split", name, "train.csv"))['filename'])
        labels = []
        d = {}
        for f in image_list:
            n = f.split('/')[-2]
            if n not in d:
                d[n] = len(d)
            labels.append(d[n])
        self.image_list = image_list
        self.labels = labels
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path, label = os.path.join(self.root_dir, 'images', self.image_list[idx]), self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class PetFaceEvalDataset(Dataset):
    def __init__(self, root_dir, class_name, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(os.path.join(root_dir, "split", class_name, "verification.csv"))
        self.data = [(row[0], row[1], int(row[2])) for row in df.values]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path1, img_path2, label = self.data[idx]
        image1 = Image.open(os.path.join(self.root_dir, 'images', img_path1)).convert("RGB")
        image2 = Image.open(os.path.join(self.root_dir, 'images', img_path2)).convert("RGB")
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, label

class PetFaceReIDEvalDataset(Dataset):
    def __init__(self, root_dir, class_name, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        lines = open(os.path.join(root_dir, "split", class_name, "test.txt")).readlines()
        self.data = []
        pid = {}
        for f in lines:
            f = f.strip()
            i = f.split('/')[-2]
            if i not in pid:
                pid[i] = len(pid)
            self.data.append((f, pid[i]))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path1, label = self.data[idx]
        image1 = Image.open(os.path.join(self.root_dir, 'images', img_path1)).convert("RGB")
        
        if self.transform:
            image1 = self.transform(image1)
        
        return image1, label

class PetFaceTrainDataset(Dataset):
    def __init__(self, image_path, class_names=None, transform=None, sample_by_id=False):
        self.class_names = class_names
        self.sample_by_id = sample_by_id
        print(class_names)

        image_list = []
        for name in class_names:
            filenames = pd.read_csv(os.path.join(image_path, "split", name, "train.csv"))['filename']
            image_list.extend([os.path.join(image_path, 'images', f) for f in filenames])
        print(len(image_list))
        d = {}
        self.labels = []
        self.label2images = {}
        self.categorytoindex = {c: i for i, c in enumerate(class_names)}
        self.category_labels = []
        self.pid2label = {}
        for idx, f in enumerate(image_list):
            n = '/'.join(f.split('/')[-3:-1])
            if n not in d:
                d[n] = len(d)
                self.pid2label[n] = f.split('/')[-3]
            self.labels.append(d[n])
            if n not in self.label2images:
                self.label2images[n] = []
            # self.label2images[n].append(idx)
            self.label2images[n].append(f)
            self.category_labels.append(self.categorytoindex[n.split('/')[0]])

        self.image_list = image_list
        from torchvision import transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                # transforms.RandomResizedCrop(size=224, scale=(0.08, 1.)),
                # transforms.RandomCrop(224, padding=8, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([
                #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], 0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True),
            ])
        else:
            self.transform = transform

    def __len__(self):
        if self.sample_by_id:
            return len(self.label2images)
        else:
            return len(self.image_list)
    
    def __getitem__(self, idx):
        if self.sample_by_id:
            pid = list(self.label2images.keys())[idx]
            import random
            # print(self.label2images[pid])
            # exit(0)
            img_path1, img_path2 = random.choices(self.label2images[pid], k=2)
            image1 = self.transform(Image.open(img_path1).convert("RGB"))
            image2 = self.transform(Image.open(img_path2).convert("RGB"))
            return {'image_crops': torch.stack([image1, image2]), 'labels': torch.tensor(idx, dtype=torch.long)}
        else:
            img_path1 = self.image_list[idx]
            image1 = self.transform(Image.open(img_path1).convert("RGB"))
            return {'image_crops': image1.unsqueeze(0), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}


import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
from PIL import Image, ImageOps
def pad_to_square(img, fill=0):
	"""Pads an image to make it square by adding equal padding to shorter sides."""
	width, height = img.size
	max_side = max(width, height)
	
	# Compute padding
	left = (max_side - width) // 2
	right = max_side - width - left
	top = (max_side - height) // 2
	bottom = max_side - height - top
	
	# Apply padding
	img_padded = ImageOps.expand(img, (left, top, right, bottom), fill=fill)
	return img_padded