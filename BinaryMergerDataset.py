
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import glob
from astropy.visualization import simple_norm, SqrtStretch
from skimage.transform import resize
from SetRandomSeed import set_random_seeds




set_random_seeds(626)

path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/NormalizedDataset/'
#stretch = AsinhStretch()
pad_val = 0
BATCH_SIZE = 64
NUM_EPOCHS = 40
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



torch.cuda.is_available()




class BinaryMergerDataset(Dataset): 
    def __init__(self, data_path, dataset, mergers = True, transform=None, codetest=True):
        self.dataset = dataset
        self.mergers = mergers
        self.codetest=codetest
        if self.dataset == 'train':
            if mergers == True:
                self.images = glob.glob(data_path + 'training/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'training/anymergers/mergerlabel.npy')
                self.image_filenames = self.images
                #print('length of file list', len(self.images))
            else:
                self.images = glob.glob(data_path + 'training/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'training/nonmergers/mergerlabel.npy')
                self.image_filenames = self.images
                #print('length of file list', len(self.images))
        elif self.dataset == 'validation':
            if mergers == True:
                self.images = glob.glob(data_path + 'validation/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'validation/anymergers/mergerlabel.npy')
                self.image_filenames = self.images
            else:
                self.images = glob.glob(data_path + 'validation/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'validation/nonmergers/mergerlabel.npy')
                self.image_filenames = self.images
        elif self.dataset == 'test':
            if mergers == True:
                self.images = glob.glob(data_path + 'test/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'test/anymergers/mergerlabel.npy')
                self.image_filenames = self.images
            else:
                self.images = glob.glob(data_path + 'test/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'test/nonmergers/mergerlabel.npy')
                self.image_filenames = self.images
        
        self.transform = transform
        

    def __len__(self):
        if self.codetest:
            return len(self.img_labels[0:200])
        else:   
            return len(self.img_labels)

    def __getitem__(self, idx):
        #print(np.shape(self.images))
        img_path = self.images[idx]
        img_name = self.image_filenames[idx]
        #print(idx)
        image = np.load(img_path) #keep as np array to normalize

        label = self.img_labels[idx]

        # split word
        w = 'allfilters'
        _, _, name = img_name.partition(w)
        name = name[:-4]
 
        if self.transform is not None:
            image = self.transform(image)

        return image, int(label), name


# In[7]:


def get_transforms(aug=True):
    transforms = []
    transforms.append(T.ToTensor())
    if aug == True:
        transforms.extend([
        #transforms.append(torch.nn.Sequential(
        T.RandomRotation(180), 
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.85, 1.1))
        ])
    else: 
        transforms.append(T.Pad(pad_val))

    return T.Compose(transforms)



