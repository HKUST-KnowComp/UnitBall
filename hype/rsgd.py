#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Note: this file is modified in the UnitBall repository compared with the original codes in poincare-embeddings

from torch.optim.optimizer import Optimizer, required
import sys
from torch.nn import Embedding
from torch import cat


def norm(re_u, im_u):
    if isinstance(re_u, Embedding):
        re_u = re_u.weight
    if isinstance(im_u, Embedding):
        im_u = im_u.weight
    norm = (re_u.pow(2).sum(dim=-1)+im_u.pow(2).sum(dim=-1)).sqrt()
    return norm


def normalize(re_z, im_z):
    d = re_z.size(-1)
    cat_z = cat((re_z.data, im_z.data), -1)
    cat_z.view(-1, 2 * d).renorm_(2, 0, 1 - 1e-5)
    re_z.data = cat_z[..., 0: d]
    im_z.data = cat_z[..., d:]
    return re_z, im_z


class RiemannianSGD(Optimizer):
    """Riemannian stochastic gradient descent.

    Args:
        rgrad (Function): Function to compute the Riemannian gradient
           from the Euclidean gradient
        retraction (Function): Function to update the retraction
           of the Riemannian gradient
    """

    def __init__(
            self,
            params,
            lr=required,
            rgrad=required,
            expm=required,
            complex_tensor=required,
    ):
        defaults = {
            'lr': lr,
            'rgrad': rgrad,
            'expm': expm,
            'complex_tensor': complex_tensor,
        }
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None, counts=None, complex_tensor=False, **kwargs):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None
        if complex_tensor:
            for group in self.param_groups:
                re_p = group['params'][0]
                im_p = group['params'][1]
                lr = lr or group['lr']
                rgrad = group['rgrad']
                expm = group['expm']
                if re_p.grad is None:
                    continue
                re_d_p = re_p.grad.data
                im_d_p = im_p.grad.data
                if re_d_p.is_sparse:
                    re_d_p = re_d_p.coalesce()
                    im_d_p = im_d_p.coalesce()
                re_d_p, im_d_p = rgrad(re_p.data, im_p.data, re_d_p, im_d_p)
                re_d_p.mul_(-lr)
                im_d_p.mul_(-lr)
                re_p.data, im_p.data = expm(re_p.data, im_p.data, re_d_p, im_d_p, normalize=True)
        else:
            for group in self.param_groups:
                for p in group['params']:
                    lr = lr or group['lr']
                    rgrad = group['rgrad']
                    expm = group['expm']
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if d_p.is_sparse:
                        d_p = d_p.coalesce()
                    d_p = rgrad(p.data, d_p)
                    d_p.mul_(-lr)
                    expm(p.data, d_p)

        return loss
