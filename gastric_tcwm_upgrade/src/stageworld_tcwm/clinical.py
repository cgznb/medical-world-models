"""Frozen clinical logits fitted only on the training patients."""
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ClinicalAnchor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(32))
        self.register_buffer("scale", torch.ones(32))
        self.register_buffer("weight", torch.zeros(32))
        self.register_buffer("bias", torch.zeros(()))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, clinical, labels, valid):
        x = clinical[valid].detach().cpu().double().numpy()
        y = labels[valid].detach().cpu().numpy()
        self.weight.zero_()
        self.bias.zero_()
        self.mean.zero_()
        self.scale.fill_(1)
        if len(y):
            scaler = StandardScaler().fit(x)
            self.mean.copy_(torch.as_tensor(scaler.mean_))
            self.scale.copy_(torch.as_tensor(scaler.scale_))
            if len(np.unique(y)) == 2:
                classifier = LogisticRegression(C=1., solver="lbfgs", max_iter=2000,
                                                tol=1e-8, random_state=17)
                classifier.fit(scaler.transform(x), y)
                self.weight.copy_(torch.as_tensor(classifier.coef_[0]))
                self.bias.copy_(torch.as_tensor(classifier.intercept_[0]))
            else:
                prevalence = np.clip(y.mean(), 1e-4, 1-1e-4)
                self.bias.fill_(float(np.log(prevalence/(1-prevalence))))
        self.fitted.fill_(True)

    def forward(self, clinical):
        if not bool(self.fitted):
            raise RuntimeError("Clinical anchor must be fitted on training patients before use")
        with torch.autocast(clinical.device.type, enabled=False):
            return ((clinical.float()-self.mean)/self.scale) @ self.weight + self.bias
