import torch
import lightning.pytorch as pl
import data as dt
import einops

class TCDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        
        self.data_cfg = config.dataset
        self.normalizer = dt.get_normalizer(self.data_cfg.normalize)
        self.batch_size = config.optim.batch_size
        self.num_workers = config.optim.num_workers
        

    def prepare_data(self):
        # load data 
        self.dataset = dt.load_data(num_channels=self.data_cfg.num_channels, combine=self.data_cfg.combine_data, normalize=self.normalizer)
        

    def setup(self, stage=None):

        t_in = self.data_cfg.t_in
        t_out = self.data_cfg.t_out
        Xs, Ys = [], []
        for data in self.dataset:
            X, Y = dt.temporal_data(t_in, t_out, data, task=self.data_cfg.task)
            X = X[:,:,:, :self.data_cfg.img_height, :self.data_cfg.img_width]
            Y = Y[:,:,:, :self.data_cfg.img_height, :self.data_cfg.img_width]
            Xs.append(X)
            Ys.append(Y)
        Xs = [einops.rearrange(X, 'b t c h w -> b t h w c') for X in Xs]
        Ys = [einops.rearrange(Y, 'b t c h w -> b t h w c') for Y in Ys]
        self.eval_datasets = [dt.CustomDataset(X, Y, indices=torch.arange(i * 856 + t_in, i * 856 + t_in + X.shape[0])) for i, (X, Y) in enumerate(zip(Xs, Ys))]
        self.trainset = torch.utils.data.ConcatDataset(self.eval_datasets[:self.data_cfg.num_train_chunks])
        self.testset = torch.utils.data.ConcatDataset(self.eval_datasets[self.data_cfg.num_train_chunks:])

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(self.trainset,  batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return trainloader

    @property
    def num_train_samples(self):
        return len(self.trainset)
    
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return testloader