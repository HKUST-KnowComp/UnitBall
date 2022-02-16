#!/usr/bin/env python3
# Copyright (c) 2021-present, Huiru Xiao, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch as th
from torch.autograd import Function
from .complexmanifold import ComplexManifold
import numpy as np
import sys
from torch.nn import Embedding


def norm(re_u, im_u):
    if isinstance(re_u, Embedding):
        re_u = re_u.weight
    if isinstance(im_u, Embedding):
        im_u = im_u.weight
    norm = (re_u.pow(2).sum(dim=-1)+im_u.pow(2).sum(dim=-1)).sqrt()
    return norm


class UnitBallModel(ComplexManifold):
    def __init__(self, eps=1e-5, K=None, **kwargs):
        self.eps = eps
        self.max_norm = 1 - self.eps
        self.K = K
        if K is not None:
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * self.K))

    def normalize(self, re_z, im_z):
        
        d = re_z.size(-1)
        if self.max_norm:
            cat_z = th.cat((re_z.data, im_z.data), -1)
            cat_z.view(-1, 2 * d).renorm_(2, 0, self.max_norm)
        re_z.data = cat_z[..., 0: d]
        im_z.data = cat_z[..., d:]
        return re_z, im_z

    def distance(self, re_z, im_z, re_w, im_w):
        return Distance.apply(re_z, im_z, re_w, im_w, self.eps)

    def rgrad(self, re_p, im_p, re_d_p, im_d_p):
        if re_d_p.is_sparse:
            p_sqnorm = th.sum(
                re_p[re_d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
            ).expand_as(re_d_p._values()) + th.sum(
                im_p[im_d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
            ).expand_as(im_d_p._values())
            re_n_vals = re_d_p._values() * ((1 - p_sqnorm) ** 2) / 4
            re_n_vals.renorm_(2, 0, 5)
            im_n_vals = im_d_p._values() * ((1 - p_sqnorm) ** 2) / 4
            im_n_vals.renorm_(2, 0, 5)
            re_d_p = th.sparse.DoubleTensor(re_d_p._indices(), re_n_vals, re_d_p.size())
            im_d_p = th.sparse.DoubleTensor(im_d_p._indices(), im_n_vals, im_d_p.size())
        else:
            p_sqnorm = th.sum(re_p ** 2, dim=-1, keepdim=True) + th.sum(im_p ** 2, dim=-1, keepdim=True)
            re_d_p = re_d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(re_d_p)
            im_d_p = im_d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(im_d_p)
        return re_d_p, im_d_p

    # TODO: to be modified to complex version
    def half_aperture(self, u):
        eps = self.eps
        sqnu = u.pow(2).sum(dim=-1)
        sqnu.clamp_(min=0, max=1 - eps)
        return th.asin((self.inner_radius * (1 - sqnu) / th.sqrt(sqnu))
            .clamp(min=-1 + eps, max=1 - eps))

    # TODO: to be modified to complex version
    def angle_at_u(self, u, v):
        norm_u = u.norm(2, dim=-1)
        norm_v = v.norm(2, dim=-1)
        dot_prod = (u * v).sum(dim=-1)
        edist = (u - v).norm(2, dim=-1)  # euclidean distance
        num = (dot_prod * (1 + norm_v ** 2) - norm_v ** 2 * (1 + norm_u ** 2))
        denom = (norm_v * edist * (1 + norm_v**2 * norm_u**2 - 2 * dot_prod).sqrt())
        return (num / denom).clamp_(min=-1 + self.eps, max=1 - self.eps).acos()

    # TODO: to be modified to complex hyperbolic version (current is complex + poincare)
    def expm(self, re_p, im_p, re_d_p, im_d_p, normalize=False, lr=None, re_out=None, im_out=None):
        if lr is not None:
            re_d_p.mul_(-lr)
            im_d_p.mul_(-lr)
        if re_out is None:
            re_out = re_p
        if im_out is None:
            im_out = im_p
        # re_out.add_(re_d_p)
        # im_out.add_(im_d_p)
        re_out = th.add(re_out, re_d_p)
        im_out = th.add(im_out, im_d_p)
        if normalize:
            self.normalize(re_out, im_out)
        return re_out, im_out

    # TODO: to be modified to complex version
    def logm(self, p, d_p, out=None):
        return p - d_p

    # TODO: to be modified to complex version
    def ptransp(self, p, x, y, v):
        ix, v_ = v._indices().squeeze(), v._values()
        return p.index_copy_(0, ix, v_)


class Distance(Function):
    @staticmethod
    def grad(z, w, znorm, wnorm, zw, wz, x, eps):
        # The gradient with regard to the real part and the imag part
        p = th.sqrt(th.pow(x, 2) - 1)
        p = th.clamp(p * th.pow(znorm, 2) * wnorm, max=-eps).unsqueeze(-1)
        a = znorm.unsqueeze(-1).expand_as(z) * (zw.unsqueeze(-1).expand_as(z) * w).real - (zw * wz).real.unsqueeze(-1).expand_as(z) * z.real
        b = znorm.unsqueeze(-1).expand_as(z) * (zw.unsqueeze(-1).expand_as(z) * w).imag - (zw * wz).real.unsqueeze(-1).expand_as(z) * z.imag
        grad_real = 4 * a / p.expand_as(z)
        grad_imag = 4 * b / p.expand_as(z)
        return grad_real, grad_imag

    @staticmethod
    def forward(ctx, re_z, im_z, re_w, im_w, eps):
        z = re_z + im_z * 1j
        w = re_w + im_w * 1j
        zw = Distance.HermitianSig(z, w)           # size=[batchsize, negsample]
        wz = Distance.HermitianSig(w, z)           # size=[batchsize, negsample]
        znorm = th.clamp(Distance.HermitianSig(z, z).real, min=-1, max=-eps)   # size=[batchsize, negsample]
        wnorm = th.clamp(Distance.HermitianSig(w, w).real, min=-1, max=-eps)   # size=[batchsize, negsample]
        x = th.add(2 * (zw * wz).real / (znorm * wnorm), -1)
        ctx.eps = eps
        ctx.save_for_backward(z, w, znorm, wnorm, zw, wz, x)
        return th.acosh(x)

    @staticmethod
    def backward(ctx, g):
        z, w, znorm, wnorm, zw, wz, x = ctx.saved_tensors
        g = g.unsqueeze(-1)
        re_gz, im_gz = Distance.grad(z, w, znorm, wnorm, zw, wz, x, ctx.eps)
        re_gw, im_gw = Distance.grad(w, z, wnorm, znorm, wz, zw, x, ctx.eps)
        re_z_grad = g.expand_as(re_gz) * re_gz
        im_z_grad = g.expand_as(im_gz) * im_gz
        re_w_grad = g.expand_as(re_gw) * re_gw
        im_w_grad = g.expand_as(im_gw) * im_gw
        return re_z_grad, im_z_grad, re_w_grad, im_w_grad, None

    @staticmethod
    def HermitianSig(z, w):
        return th.add(th.sum(z * Distance.conjugate(w), dim=-1), -1)

    @staticmethod
    def conjugate(z):
        return th.conj(z)
