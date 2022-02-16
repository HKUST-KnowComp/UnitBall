#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Note: this file is modified in the UnitBall repository compared with the original codes in poincare-embeddings

import torch as th
import numpy as np
import logging
import argparse
from hype.adjacency_matrix_dataset import AdjacencyDataset
from hype import train
from hype.graph import load_adjacency_matrix, load_edge_list, eval_reconstruction
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
import sys
import json
import torch.multiprocessing as mp
import shutil
from hype.graph_dataset import BatchedDataset
from hype import MANIFOLDS, MODELS, build_model
import pandas

th.manual_seed(42)
np.random.seed(42)


def reconstruction_eval(adj, opt, epoch, elapsed, loss, pth, best=None, complex=False):
    chkpnt = th.load(pth, map_location='cpu')
    hits_conf = [1, 3, 10]
    if not complex:
        model = build_model(opt, chkpnt['embeddings'].size(0))
        model.load_state_dict(chkpnt['model'])

        sqnorms = model.manifold.norm(model.lt)
        lmsg = {
            'epoch': epoch,
            'elapsed': elapsed,
            'loss': loss,
            'sqnorm_min': sqnorms.min().item(),
            'sqnorm_avg': sqnorms.mean().item(),
            'sqnorm_max': sqnorms.max().item(),
        }
        meanrank, maprank, mrr, roc, hits = eval_reconstruction(adj, model)
        lmsg['mr'] = meanrank
        lmsg['map'] = maprank
        lmsg['mrr'] = mrr
        lmsg['roc'] = roc
        for j, hj in enumerate(hits_conf):
            lmsg[f'hits{hj}'] = hits[j]
    else:
        model = build_model(opt, chkpnt['embeddings'][0].size(0))
        model.load_state_dict(chkpnt['model'])

        embeddings = chkpnt['embeddings']

        sqnorms = model.manifold.norm(embeddings[0], embeddings[1])
        lmsg = {
            'epoch': epoch,
            'elapsed': elapsed,
            'loss': loss,
            'sqnorm_min': sqnorms.min().item(),
            'sqnorm_avg': sqnorms.mean().item(),
            'sqnorm_max': sqnorms.max().item(),
        }
        meanrank, maprank, mrr, roc, hits = eval_reconstruction(adj, model, embeddings=embeddings, complex=True)
        lmsg['mr'] = meanrank
        lmsg['map'] = maprank
        lmsg['mrr'] = mrr
        lmsg['roc'] = roc
        for j, hj in enumerate(hits_conf):
            lmsg[f'hits{hj}'] = hits[j]
    return lmsg, pth


def hypernymy_eval(epoch, elapsed, loss, pth, best):
    _, summary = hype_eval(pth, cpu=True)
    return {
        'epoch': epoch,
        'elapsed': elapsed,
        'loss': loss,
        'best': bool(
            best is None or summary['eval_hypernymy_avg'] > best['eval_hypernymy_avg'])
        ,
        **summary
    }


def async_eval(adj, q, logQ, opt):
    best = None
    while True:
        temp = q.get()
        if temp is None:
            return

        if not q.empty():
            continue

        epoch, elapsed, loss, pth = temp
        if opt.eval == 'reconstruction':
            lmsg = reconstruction_eval(adj, opt, epoch, elapsed, loss, pth, best)
        elif opt.eval == 'hypernymy':
            lmsg = hypernymy_eval(epoch, elapsed, loss, pth, best)
        else:
            raise ValueError(f'Unrecognized evaluation: {opt.eval}')
        best = lmsg if lmsg['best'] else best
        logQ.put((lmsg, pth))


# Adapated from:
# https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
class Unsettable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(Unsettable, self).__init__(option_strings, dest, nargs='?', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        val = None if option_string.startswith('-no') else values
        setattr(namespace, self.dest, val)


def main():
    parser = argparse.ArgumentParser(description='Train Complex and Real Hyperbolic Embeddings')
    parser.add_argument('-checkpoint', help='Where to store the model checkpoint')
    parser.add_argument('-trainset', type=str, default='./data/ICD10/train_taxonomy.csv',
                        help='Training data path')
    parser.add_argument('-testset', type=str, default='./data/ICD10/test_taxonomy.csv',
                        help='Test data path')
    parser.add_argument('-dim', type=int,
                        help='Embedding dimension')
    parser.add_argument('-manifold', type=str, default='unitball',
                        choices=MANIFOLDS.keys())
    parser.add_argument('-model', type=str, default='distance',
                        choices=MODELS.keys(), help='Energy function model')
    parser.add_argument('-lr', type=float, default=1000,
                        help='Learning rate')
    parser.add_argument('-eps', type=float, default=1e-5, help='Eps to avoid numerical instabilities')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-batchsize', type=int, default=12800,
                        help='Batchsize')
    parser.add_argument('-negs', type=int, default=50,
                        help='Number of negatives')
    parser.add_argument('-burnin', type=int, default=20,
                        help='Epochs of burn in')
    parser.add_argument('-dampening', type=float, default=0.75,
                        help='Sample dampening during burnin')
    parser.add_argument('-ndproc', type=int, default=8,
                        help='Number of data loading processes')
    parser.add_argument('-fresh', action='store_true', default=False,
                        help='Override checkpoint')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Print debuggin output')
    parser.add_argument('-gpu', default=0, type=int,
                        help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-sym', action='store_true', default=False,
                        help='Symmetrize dataset')
    parser.add_argument('-maxnorm', '-no-maxnorm', default='500000',
                        action=Unsettable, type=int)
    parser.add_argument('-sparse', default=False, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-burnin_multiplier', default=0.01, type=float)
    parser.add_argument('-neg_multiplier', default=1.0, type=float)
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument('-lr_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-train_threads', type=int, default=1,
                        help='Number of threads to use in training')
    parser.add_argument('-margin', type=float, default=0.1, help='Hinge margin')
    parser.add_argument('-eval', choices=['reconstruction', 'hypernymy'],
                        default='reconstruction', help='Which type of eval to perform')
    opt = parser.parse_args()

    opt.complex = True if opt.manifold == 'unitball' else False

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('lorentz')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    if opt.gpu >= 0 and opt.train_threads > 1:
        opt.gpu = -1
        log.warning(f'Specified hogwild training with GPU, defaulting to CPU...')

    # set default tensor type
    th.set_default_tensor_type('torch.DoubleTensor')
    # set device
    device = th.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 else 'cpu')

    if 'csv' in opt.trainset:
        log.info('Using edge list dataloader')
        idx, objects, weights = load_edge_list(opt.trainset, opt.sym)
        data = BatchedDataset(idx, objects, weights, opt.negs, opt.batchsize,
            opt.ndproc, opt.burnin > 0, opt.dampening)
    else:
        log.info('Using adjacency matrix dataloader')
        dset = load_adjacency_matrix(opt.trainset, 'hdf5')
        log.info('Setting up dataset...')
        data = AdjacencyDataset(dset, opt.negs, opt.batchsize, opt.ndproc,
            opt.burnin > 0, sample_dampening=opt.dampening)
        objects = dset['objects']

    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}

    log.info('Test taxonomy: using edge list dataloader')
    adj_test = {}
    test_taxonomy = pandas.read_csv(opt.testset)
    for i, row in test_taxonomy.iterrows():
        x_idx = objects.index(row['id1'])
        y_idx = objects.index(row['id2'])
        if x_idx in adj_test:
            adj_test[x_idx].add(y_idx)
        else:
            adj_test[x_idx] = {y_idx}

    model = build_model(opt, len(objects))
    # log.info(f'The initial model is {model}')

    # set burnin parameters
    data.neg_multiplier = opt.neg_multiplier
    train._lr_multiplier = opt.burnin_multiplier

    # Build config string for log
    log.info(f'json_conf: {json.dumps(vars(opt))}')

    if opt.lr_type == 'scale':
        opt.lr = opt.lr * opt.batchsize

    # setup optimizer
    optimizer = RiemannianSGD(model.optim_params(), lr=opt.lr, complex_tensor=opt.complex)

    # setup checkpoint
    checkpoint = LocalCheckpoint(
        opt.checkpoint,
        include_in_all={'conf': vars(opt), 'objects': objects},
        start_fresh=opt.fresh
    )

    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']


    controlQ, logQ = mp.Queue(), mp.Queue()

    # control closure
    def control(model, epoch, elapsed, loss):
        """
        Control thread to evaluate embedding
        """
        if hasattr(model, 'w_avg'):
            lt = model.w_avg
            model.manifold.normalize(lt)
        else:
            if not opt.complex:
                lt = model.lt.weight.data
                model.manifold.normalize(lt)
            else:
                lt = (model.lt[0].weight.data, model.lt[1].weight.data)
                model.manifold.normalize(model.lt[0].weight.data, model.lt[1].weight.data)
        checkpoint.path = f'{opt.checkpoint}.{epoch}'
        checkpoint.save({
            'model': model.state_dict(),
            'embeddings': lt,
            'epoch': epoch,
            'model_type': opt.model,
        })

        controlQ.put((epoch, elapsed, loss, checkpoint.path))

        lmsg, pth = reconstruction_eval(adj_test, opt, epoch, elapsed, loss, checkpoint.path, complex=opt.complex)
        shutil.move(pth, opt.checkpoint)
        log.info(f'json_stats: {json.dumps(lmsg)}')

    control.checkpoint = True
    model = model.to(device)
    if hasattr(model, 'w_avg'):
        model.w_avg = model.w_avg.to(device)
    if opt.train_threads > 1:
        threads = []
        model = model.share_memory()
        args = (device, model, data, optimizer, opt, log, opt.complex)
        kwargs = {'ctrl': control, 'progress': not opt.quiet}
        for i in range(opt.train_threads):
            kwargs['rank'] = i
            threads.append(mp.Process(target=train.train, args=args, kwargs=kwargs))
            threads[-1].start()
        [t.join() for t in threads]
    else:
        train.train(device, model, data, optimizer, opt, log, opt.complex, ctrl=control,
            progress=not opt.quiet)
    control(model, opt.epochs, 0, 0)
    while not logQ.empty():
        lmsg, pth = logQ.get()
        shutil.move(pth, opt.checkpoint)
        log.info(f'json_stats: {json.dumps(lmsg)}')


if __name__ == '__main__':
    main()
