import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import math



class ProcessNet(nn.Module):
    
    """
    Data preprocessing: calculate different densities on 3D grid accerlerated by GPU using pytorch
    """
    
    def __init__(self, pos_grid):
        
        """
        pos_grid: 3D grid with x, y, z positions and dummy dimensions
        z, a, b, c are parameters for slater and form factor based densities
        IMPORTANT: b is already scaled with the factor 4*pi^2: 39.44
        """
        
        super(ProcessNet, self).__init__()
        self.pos_grid = torch.from_numpy(pos_grid).float().cuda()
        self.z = torch.tensor([1, 3.14, 3.83, 4.45]).float().cuda()
        self.a1 = torch.tensor([0.489918, 2.31, 12.2126, 3.0485]).float().cuda().view(4,1) # normalize
        self.b1 = 1 / torch.tensor([20.6593, 20.8439, 0.0057, 13.2771]).float().cuda().view(4,1) * 39.44
        self.a2 = torch.tensor([0.262003, 1.02, 3.1322, 2.2868]).float().cuda().view(4,1)
        self.b2 = 1 / torch.tensor([7.74039, 10.2075, 9.8933, 5.7011]).float().cuda().view(4,1) * 39.44
        self.a3 = torch.tensor([0.196767, 1.5886, 2.0125, 1.5463]).float().cuda().view(4,1)
        self.b3 = 1 / torch.tensor([49.5519, 0.5687, 28.9975, 0.3239]).float().cuda().view(4,1) * 39.44
        self.a4 = torch.tensor([0.049879, 0.865, 1.1663, 0.867]).float().cuda().view(4,1)
        self.b4 = 1 / torch.tensor([2.20159, 51.6512, 0.5826, 32.9089]).float().cuda().view(4,1) * 39.44
        
        
    def _gaussian(self, x, feature, sigma=1/3):
       
        """
        Gaussian density: D(x)=exp(-(x-x_a)^2/sigma) without normalizing factor
        """
        
        diff = self.pos_grid - x.permute(2,0,1)
        norm = torch.norm(diff, dim=-3)
        gaussian = torch.exp(- norm * norm / sigma)
        gaussian = gaussian * feature.permute(2,0,1)
        gaussian = torch.sum(gaussian, dim=-1).permute(4,0,1,2,3)
        return gaussian
    
    
    def _slater(self, x, feature):
        
        """
        Slater density: D(x)=r^(n-1)exp(-\zeta*r) without normalizing factor
        """
        
        diff = self.pos_grid - x.permute(2,0,1)
        norm = torch.norm(diff, dim=-3)
        r = norm
        r[:,:,:,:,:,0] = 1
        zeta = self.z * feature
        slater = r * torch.exp(- zeta.permute(2,0,1) * norm)
        slater = slater * feature.permute(2,0,1)
        slater = torch.sum(slater, dim=-1).permute(4,0,1,2,3)
        return slater
    
    
    def _form_factor(self, x, feature, norm_factor=100):
        
        """
        Density calculated from Form Factor:
        D(x)=\sum_{i=1}^4 \sqrt{b_i}*exp(-b_i*norm^2)
        IMPORTANT: b_i is scaled, please refer __init__ function
        Normalized with 100 in denominator, can be tuned.
        """
        
        diff = self.pos_grid - x.permute(2,0,1)
        norm = torch.norm(diff, dim=-3)
        ff = self.a1 * torch.sqrt(self.b1) * torch.exp(- self.b1 * norm * norm) \
             + self.a2 * torch.sqrt(self.b2) * torch.exp(- self.b2 * norm * norm) \
             + self.a3 * torch.sqrt(self.b3) * torch.exp(- self.b3 * norm * norm) \
             + self.a4 * torch.sqrt(self.b4) * torch.exp(- self.b4 * norm * norm)
        ff = ff * feature.permute(2,0,1) / norm_factor
        ff = torch.sum(ff, dim=-1).permute(4,0,1,2,3)
        return ff
    
    
    def forward(self, x, feature, density_type="Gaussian", hyperparameter=1/3):
        
        """
        Calculate different densities
        x: torch cuda tensor x, y, z coordinates
        feature: torch cuda tensor one-hot atom type
        density_type: only suppotr "Gaussian", "Slater" and "Form_Factor"
        hyperparameter: for Gaussian, it's sigma, default 1/3; for Form_Fator, it's normalizing factor
        This normalizing factor can be tuned to help the convergence during training session
        """
        
        if density_type == "Gaussian":
            return self._gaussian(x, feature, hyperparameter)
        if density_type == "Slater":
            return self._slater(x, feature)
        if density_type == "Form_Factor":
            return self._form_factor(x, feature, hyperparameter)
        else:
            raise NotImplementedError("Density Type Not Implemented!")
            
            
            
# numpy version
def density_calc(x, feature, pos_grid, density_type="Gaussian", hyperparameter=1/3):
    
    """
    numpy version to calculate the density, only implemented the Gaussian Density
    """
    
    def _gaussian(x, feature, pos_grid, sigma=1/3):
               
        """
        Gaussian density: D(x)=exp(-(x-x_a)^2/sigma) without normalizing factor
        """
        
        diff = pos_grid - np.transpose(x,(2,0,1))
        norm = np.linalg.norm(diff, axis=-3)
        gaussian = np.exp(- * norm * norm / sigma)
        gaussian = gaussian * np.transpose(feature, (2,0,1))
        gaussian = np.transpose(np.sum(gaussian, axis=-1, dtype=np.float16, keepdims = False), (4,0,1,2,3))
        return gaussian
    
    
    def _slater(x, feature, pos_grid):
        
        """
        Slater density: D(x)=r^(n-1)exp(-\zeta*r) without normalizing factor
        """
        
        z = np.array([1, 3.14, 3.83, 4.45])
        diff = pos_grid - np.transpose(x,(2,0,1))
        norm = np.linalg.norm(diff, axis=-3)
        r = norm
        r[:,:,:,:,:,0] = 1
        zeta = z * feature
        slater = r * np.exp(- np.transpose(zeta, (2,0,1)) * norm)
        slater = slater * np.transpose(feature, (2,0,1))
        slater = np.sum(slater, dim=-1)
        slater = np.transpose(slater, (4,0,1,2,3))
        return slater
    
    
    def _form_factor(self, x, feature, norm_factor=100):
        
        """
        Density calculated from Form Factor:
        D(x)=\sum_{i=1}^4 \sqrt{b_i}*exp(-b_i*norm^2)
        IMPORTANT: b_i is scaled, please refer __init__ function
        Normalized with 100 in denominator, can be tuned.
        """
        
        a1 = np.array([0.489918, 2.31, 12.2126, 3.0485])
        b1 = 1 / np.array([20.6593, 20.8439, 0.0057, 13.2771]).reshape((4,1)) * 39.44
        a2 = np.array([0.262003, 1.02, 3.1322, 2.2868]).reshape((4,1))
        b2 = 1 / np.array([7.74039, 10.2075, 9.8933, 5.7011]).reshape((4,1)) * 39.44
        a3 = np.array([0.196767, 1.5886, 2.0125, 1.5463]).reshape((4,1))
        b3 = 1 / np.array([49.5519, 0.5687, 28.9975, 0.3239]).reshape((4,1)) * 39.44
        a4 = np.array([0.049879, 0.865, 1.1663, 0.867]).reshape((4,1))
        b4 = 1 / np.array([2.20159, 51.6512, 0.5826, 32.9089]).reshape((4,1)) * 39.44
        diff = pos_grid - np.transpose(x,(2,0,1))
        norm = np.linalg.norm(diff, axis=-3)
        ff = a1 * np.sqrt(self.b1) * np.exp(- self.b1 * norm * norm)\
             + a2 * np.sqrt(self.b2) * np.exp(- self.b2 * norm * norm)\
             + a3 * np.sqrt(self.b3) * np.exp(- self.b3 * norm * norm)\
            + self.a4 * np.sqrt(self.b4) * np.exp(- self.b4 * norm * norm)
        ff = ff * np.transpose(feature, (2,0,1)) / norm_factor
        ff = torch.sum(ff, dim=-1)
        ff = np.transpose(ff, (4,0,1,2,3))
        return ff
        
            
        """
        Calculate different densities
        x: torch cuda tensor x, y, z coordinates
        feature: torch cuda tensor one-hot atom type
        density_type: only suppotr "Gaussian", "Slater" and "Form_Factor"
        hyperparameter: for Gaussian, it's sigma, default 1/3; for Form_Fator, it's normalizing factor
        This normalizing factor can be tuned to help the convergence during training session
        """
        
    if density_type == "Gaussian":
        return self._gaussian(x, feature, hyperparameter)
    if density_type == "Slater":
        return self._slater(x, feature)
    if density_type == "Form_Factor":
        return self._form_factor(x, feature, hyperparameter)
    else:
        raise NotImplementedError("Density Type Not Implemented!")




def generate_density(prefix, batch_size, density_type="Gaussian", hyperparameter=1/3):
    
    """
    Generate the density
    prefix: the prefix for y, points and xyz files
    batch_size: depends on the GPU. Tested batch_size=32 for Tesla P100/V100
    density_type: only suppotr "Gaussian", "Slater" and "Form_Factor"
        hyperparameter: for Gaussian, it's sigma, default 1/3; for Form_Fator, it's normalizing factor
        This normalizing factor can be tuned to help the convergence during training session
    """
    
    train_y = np.load(pre + "_y.npy")
    train_points = np.load(pre + "_points.npy")[:,:,:4]
    train_points = torch.tensor(train_points).float()
    num_train_batches = train_y.shape[0] // batch_size

    for l in [2, 3, 4, 5, 7]:
        pos_grid = np.zeros((16, 16, 16, 3))
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    pos_grid[i][j][k] = np.array([(i-7.5)/7.5*l, (j-7.5)/7.5*l, (k-7.5)/7.5*l])
        pos_grid = np.array([pos_grid for _ in range(batch_size*320*4)])
        pos_grid = pos_grid.reshape((batch_size,320,4,16,16,16,3))
        pos_grid = pos_grid.transpose(3,4,5,2,6,0,1)
        model = ProcessNet(pos_grid).cuda()
        for idx in range(8):
            train_xyz = np.load(pre + "_aug_xyz_" + str(idx) + ".npy")
            print(idx)
            train_xyz = torch.tensor(train_xyz).float()
            gaussian = []
            for i in range(num_train_batches):
                print(i)
                index = list(range(i * batch_size, min((i+1) * batch_size, train_y.shape[0])))
                points, xyz = train_points[index].cuda(), train_xyz[index].cuda()
                a = model(xyz, points)
                gaussian.append(np.array(a.cpu().numpy(), dtype=np.float16))
            if train_y.shape[0] % batch_size != 0:
                gaussian.append(density_calc(train_xyz[num_train_batches*batch_size:train_y.shape[0]].numpy(), train_points[num_train_batches*batch_size:train_y.shape[0]].numpy(), pos_grid[:,:,:,:,:,0:train_y.shape[0]-num_train_batches*batch_size]))
            gaussian = np.concatenate(gaussian)
            np.save(pre + "_x_" + str(l) + "A_" + str(idx), gaussian)
            
            
            
if __name__ == "__main__":
    for dataset in ["train_", "test_"]:
        for atom_type in ["H", "C", "N", "O"]:
            generate_density(dataset + atom_type, 32)
    