from typing import Any, Callable, Optional, Tuple
import torch
from torch import nn, optim
import lightning.pytorch as pl
import torchvision.models.video as tvmv
import sklearn.metrics as skm

class SyntaxLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr: float,
        weight_decay: float = 0,
        max_epochs: int = None,
        weight_path: str = None,
        save_path: str = None,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.num_classes = num_classes
        self.save_path = save_path

        # Video ResNet
        self.model = tvmv.r3d_18(weights=tvmv.R3D_18_Weights.DEFAULT)
        # self.model = tvmv.mc3_18(weights=tvmv.MC3_18_Weights)
        # self.model = tvmv.r2plus1d_18(weights=tvmv.R2Plus1D_18_Weights)
        
        # Video S3D
        # self.model = tvmv.s3d(weights=tvmv.S3D_Weights)

        # Video SwinTransformer
        # self.model = tvmv.swin3d_t(weights=tvmv.Swin3D_T_Weights)


        self.lr = lr
        # self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        # self.loss_func = nn.MSELoss()
        # self.loss_func = nn.L1Loss()

        # Video ResNet
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # Video S3D
        # self.model.classifier = nn.Conv3d(1024, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        # Video SwinTransformer
        # in_features = self.model.head.in_features
        # self.model.head = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        self.weight_path = weight_path
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))

        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

        self.y_val = []
        self.p_val = []
        self.r_val = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, sample_weight, path = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)
        loss = loss * sample_weight
        loss = loss.mean()

        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, sample_weight, path = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)
        loss = loss * sample_weight
        loss = loss.mean()

        y_pred = torch.sigmoid(y_hat)

        self.y_val.append(int(y[...,0].cpu()))
        self.p_val.append(float(y_pred[...,0].cpu()))
        self.r_val.append(round(float(y_pred[...,0].cpu())))

        return loss

    def on_validation_epoch_end(self):
        try:
            self.log("val_roc_auc_art", skm.roc_auc_score(self.y_val, self.p_val), prog_bar=True)
            self.log("val_f1_score_art", skm.f1_score(self.y_val, self.r_val), prog_bar=True)
            self.log("val_accuracy_art", skm.accuracy_score(self.y_val, self.r_val), prog_bar=True)
        except ValueError as err:
            print(err)
            print("Y_VAL", self.y_val)
            print("P_VAL", self.p_val)
        self.y_val.clear()
        self.p_val.clear()
        self.r_val.clear()
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path)

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path+".train")

    def configure_optimizers(self):
        if not self.weight_path: # pretrain mode
            params = self.model.fc.parameters()
        else: # full train mode
            params = self.model.parameters()
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs is not None:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=self.max_epochs
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y, sample_weight, path = batch
        y_hat = self(x)
        y_pred = torch.sigmoid(y_hat)

        return {"y": y, "y_pred": torch.round(y_pred), "y_prob": y_pred}
