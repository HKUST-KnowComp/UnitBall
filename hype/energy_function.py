#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Note: this file is modified in the UnitBall repository compared with the original codes in poincare-embeddings

import torch
import torch.nn.functional as F
import torch as th
import sys
from itertools import chain


class EnergyFunction(torch.nn.Module):
    def __init__(self, manifold, dim, size, sparse=False, complex=False, **kwargs):
        super().__init__()
        self.manifold = manifold
        if complex:
            self.lt = manifold.allocate_lt(size, dim, sparse)
            self.re_lt, self.im_lt = self.lt
        else:
            self.lt = manifold.allocate_lt(size, dim, sparse)
        self.nobjects = size
        self.manifold.init_weights(self.lt)

    def forward(self, inputs):
        if hasattr(self.lt, "parameters"):
            e = self.lt(inputs)
            with torch.no_grad():
                e = self.manifold.normalize(e)
            o = e.narrow(1, 1, e.size(1) - 1)
            s = e.narrow(1, 0, 1).expand_as(o)
            return self.energy(s, o).squeeze(-1)
        else:
            # re_e = (self.lt)[0](inputs)
            # im_e = (self.lt)[1](inputs)
            re_e = self.re_lt(inputs)
            im_e = self.im_lt(inputs)
            with torch.no_grad():
                re_e, im_e = self.manifold.normalize(re_e, im_e)
            re_o = re_e.narrow(1, 1, re_e.size(1) - 1)
            im_o = im_e.narrow(1, 1, im_e.size(1) - 1)
            re_s = re_e.narrow(1, 0, 1).expand_as(re_o)
            im_s = im_e.narrow(1, 0, 1).expand_as(im_o)
            return self.complex_energy(re_s, im_s, re_o, im_o).squeeze(-1)

    def optim_params(self):
        if (hasattr(self.lt, "parameters")):
            params = self.lt.parameters()
            return [{
                'params': params,
                'rgrad': self.manifold.rgrad,
                'expm': self.manifold.expm,
                'logm': self.manifold.logm,
                'ptransp': self.manifold.ptransp,
            }]
        else:
            params = chain((self.lt)[0].parameters(), (self.lt)[1].parameters())
            return [{
                'params': params,
                'rgrad': self.manifold.rgrad,
                'expm': self.manifold.expm,
                'logm': self.manifold.logm,
                'ptransp': self.manifold.ptransp,
            }]

    def loss_function(self, inp, target, **kwargs):
        raise NotImplementedError


class DistanceEnergyFunction(EnergyFunction):
    def energy(self, s, o):
        return self.manifold.distance(s, o)

    def complex_energy(self, re_s, im_s, re_o, im_o):
        return self.manifold.distance(re_s, im_s, re_o, im_o)

    def loss(self, inp, target, **kwargs):
        # print('The negative input is: {}'.format(inp.neg()))
        # print('The target is: {}'.format(target))
        # sys.exit()
        return F.cross_entropy(inp.neg(), target)


class EntailmentConeEnergyFunction(EnergyFunction):
    def __init__(self, *args, margin=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.manifold.K is not None, (
            "K cannot be none for EntailmentConeEnergyFunction"
        )
        assert hasattr(self.manifold, 'angle_at_u'), 'Missing `angle_at_u` method'
        self.margin = margin

    def energy(self, s, o):
        energy = self.manifold.angle_at_u(o, s) - self.manifold.half_aperture(o)
        return energy.clamp(min=0)

    def loss(self, inp, target, **kwargs):
        loss = inp[:, 0].clamp_(min=0).sum()  # positive
        loss += (self.margin - inp[:, 1:]).clamp_(min=0).sum()  # negative
        return loss / inp.numel()
