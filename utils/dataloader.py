import cv2 
import torch
import glob 
from torch.utils.data import Dataset, DataLoader

class DataSampler(Dataset):
    def __init__(self, path, batch_size, threads, resolution):
        self.input_images = glob.glob(f'{path}/*') 
        self.bs = batch_size 
        self.threads = threads 
        self.size = resolution
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image_name = self.input_images[idx]
        image = cv2.imread(image_name)
        image = cv2.resize(image, (size, size))
        image = torch.from_numpy(image, dtype=torch.float).transpose(2,0,1)
        image = (image - 127.5) / 127.5

        return [image, image.clone()]
    
    def build(path, batch_size, threads, resolution):
        trainset = DataSampler(path, batch_size, threads, resolution)
        return Dataloader(trainset, batch_size=self.bs, shuffle=True, num_workers=self.threads)
        