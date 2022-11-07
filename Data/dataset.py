import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class RHSDataSet(Dataset):
    def __init__(self, h5file, case='train', transform=None, target_transform=None):
        """
        case = 'train' or 'test'
        """
        h5 = h5py.File(h5file,'r')
        self.data = np.array(h5[case], dtype=np.float32)
        self.totensor = ToTensor()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        rhs_tensor = self.totensor(self.data[idx])
        if self.transform:
            rhs_tensor = self.transform(rhs_tensor)
        return rhs_tensor

class IsoPoissonDataSet(Dataset):
    '''Dataset stores u, f, bc_value, bc_index'''
    def __init__(self, h5file, transform=None, target_transform=None):
        h5 = h5py.File(h5file, 'r')
        self.bc_index = np.array(h5['boundary_index'], dtype=np.float32)
        self.bc_value = np.array(h5['boundary_value'], dtype=np.float32)
        self.f = np.array(h5['rhs'], dtype=np.float32)
        self.u = np.array(h5['u'], dtype=np.float32)
        self.totensor = ToTensor()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, idx):
        f_tensor = self.totensor(self.f[idx])
        u_tensor = self.totensor(self.u[idx])
        bc_index_tensor = self.totensor(self.bc_index[idx])
        bc_value_tensor = self.totensor(self.bc_value[idx])
        if self.transform:
            f_tensor = self.transform(f_tensor)
            u_tensor = self.transform(u_tensor)
            bc_index_tensor = self.transform(bc_index_tensor)
            bc_value_tensor = self.transform(bc_value_tensor)
        return u_tensor, f_tensor, bc_value_tensor, bc_index_tensor