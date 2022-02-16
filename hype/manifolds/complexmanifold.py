#!/usr/bin/env python3
# Copyright (c) 2021-present, Huiru Xiao, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from torch.nn import Embedding
import sys


class ComplexManifold(object):
    def allocate_lt(self, N, dim, sparse):
        re_embedding = Embedding(N, dim, sparse=sparse)
        im_embedding = Embedding(N, dim, sparse=sparse)
        return re_embedding, im_embedding
        # return Embedding(N, 2 * dim, sparse=sparse)     # The concatenation version

    def normalize(self, u):
        raise NotImplementedError

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    def init_weights(self, w, scale=1e-4):
        w[0].weight.data.uniform_(-scale, scale)
        w[1].weight.data.uniform_(-scale, scale)

    @abstractmethod
    def expm(self, re_p, im_p, re_d_p, im_d_p, normalize=False, lr=None, re_out=None, im_out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport
        """
        raise NotImplementedError

    def norm(self, re_u, im_u, **kwargs):
        if isinstance(re_u, Embedding):
            re_u = re_u.weight
        if isinstance(im_u, Embedding):
            im_u = im_u.weight
        norm = (re_u.pow(2).sum(dim=-1)+im_u.pow(2).sum(dim=-1)).sqrt()
        # print(norm.max().item())
        # sys.exit()
        return norm

    @abstractmethod
    def half_aperture(self, u):
        """
        Compute the half aperture of an entailment cone.
        As in: https://arxiv.org/pdf/1804.01882.pdf
        """
        raise NotImplementedError

    @abstractmethod
    def angle_at_u(self, u, v):
        """
        Compute the angle between the two half lines (0u and uv
        """
        raise NotImplementedError
