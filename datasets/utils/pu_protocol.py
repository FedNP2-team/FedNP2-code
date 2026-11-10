"""Isolated data protocol for the FedNearPU v2 implementation.

Training and teacher-selection views deliberately omit hidden labels.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class ImageOnlyIndexView(Dataset):
    def __init__(self, parent, ids):
        self.parent = parent
        self.ids = np.asarray(ids, dtype=np.int64)
    def __len__(self):
        return int(self.ids.size)
    def __getitem__(self, item):
        image, _ = self.parent[int(self.ids[item])]
        return (image,)


class PUIndexView(Dataset):
    """View a parent dataset by original IDs and expose observed PU labels only."""
    def __init__(self, parent, ids, observed_by_id):
        self.parent = parent
        self.ids = np.asarray(ids, dtype=np.int64)
        self.observed_by_id = observed_by_id

    def __len__(self):
        return int(self.ids.size)

    def __getitem__(self, item):
        idx = int(self.ids[item])
        image, _ = self.parent[idx]
        return image, int(self.observed_by_id[idx]), idx


@dataclass
class ClientPUData:
    p_fit: Dataset
    u_fit: Dataset
    d_fit: Dataset
    p_score: Dataset
    u_score: Dataset
    u_cache: Dataset
    p_fit_ids: np.ndarray
    u_fit_ids: np.ndarray
    p_score_ids: np.ndarray
    u_score_ids: np.ndarray
    n_fit: int
    alpha_u: Optional[float]
    pi_dfit: Optional[float]
    prior_valid: bool


@dataclass
class PUDataBundle:
    clients: list
    probe_dataset: Dataset
    probe_pool_ids: np.ndarray
    manifests: list

    def make_probe_loader(self, ids, batch_size, workers=0):
        return DataLoader(ImageOnlyIndexView(self.probe_dataset, ids),
                          batch_size=batch_size, shuffle=False, num_workers=workers)


def _get_targets(dataset):
    if hasattr(dataset, 'targets'):
        return np.asarray(dataset.targets)
    if hasattr(dataset, 'labels'):
        return np.asarray(dataset.labels)
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        return np.asarray(dataset.dataset.targets)
    raise AttributeError('Dataset must expose targets or labels for PU partitioning.')


def _dirichlet_partition(ids, y, n_clients, n_classes, beta, rng, min_size=10):
    if beta < 0:
        raise ValueError('beta must be non-negative')
    for _ in range(10000):
        batches = [[] for _ in range(n_clients)]
        for cls in range(n_classes):
            cls_ids = ids[y[ids] == cls].copy()
            rng.shuffle(cls_ids)
            if beta == 0:
                chunks = np.array_split(cls_ids, n_clients)
            else:
                proportions = rng.dirichlet(np.full(n_clients, beta))
                proportions *= np.array([len(b) < len(ids) / n_clients for b in batches])
                if proportions.sum() == 0:
                    proportions = np.ones(n_clients, dtype=float)
                proportions /= proportions.sum()
                cuts = (np.cumsum(proportions) * len(cls_ids)).astype(int)[:-1]
                chunks = np.split(cls_ids, cuts)
            for client_id, chunk in enumerate(chunks):
                batches[client_id].extend(map(int, chunk))
        if min(map(len, batches)) >= min_size:
            return [np.asarray(sorted(x), dtype=np.int64) for x in batches]
    raise RuntimeError('Could not form client partitions meeting min_size after 10000 attempts.')


def _estimate_priors(n_p, n_u, c, n_p_fit=None, n_u_fit=None):
    n_d = n_p + n_u
    if n_p <= 0 or n_d <= 0 or n_u < 0 or not (0 < c <= 1):
        return None, None, None, False
    raw_pi = n_p / (c * n_d)
    pi_d0 = float(np.clip(raw_pi, n_p / n_d, 1.0))
    if n_u == 0:
        alpha_u = None
    else:
        alpha_u = float(np.clip((n_d * pi_d0 - n_p) / n_u, 0.0, 1.0))
    if n_p_fit is None or n_u_fit is None:
        return pi_d0, alpha_u, None, alpha_u is not None
    n_fit = n_p_fit + n_u_fit
    if n_fit == 0 or (n_u_fit > 0 and alpha_u is None):
        return pi_d0, alpha_u, None, False
    pi_dfit = n_p_fit / n_fit
    if n_u_fit:
        pi_dfit += n_u_fit / n_fit * alpha_u
    return pi_d0, alpha_u, float(np.clip(pi_dfit, 0.0, 1.0)), alpha_u is not None


def build_pu_data_bundle(train_dataset, stable_dataset, test_dataset, setting):
    """Partition into a public pool, clients, and fixed PU scoring folds."""
    args = setting.args
    seed = int(getattr(args, 'seed', 0))
    seed_seq = np.random.SeedSequence(seed)
    rng_public, rng_partition, *client_seeds = [np.random.default_rng(s) for s in seed_seq.spawn(args.parti_num + 2)]
    y = _get_targets(train_dataset)
    ids = np.arange(len(y), dtype=np.int64)
    pool_size = int(args.public_pool_size)
    if pool_size <= 0 or pool_size >= len(ids):
        raise ValueError(f'public_pool_size must be in [1, {len(ids)-1}], got {pool_size}')
    probe_pool_ids = np.sort(rng_public.choice(ids, size=pool_size, replace=False))
    public_mask = np.ones(len(ids), dtype=bool)
    public_mask[probe_pool_ids] = False
    private_ids = ids[public_mask]
    client_ids = _dirichlet_partition(private_ids, y, args.parti_num, setting.N_CLASS,
                                      float(args.beta), rng_partition)
    pos_classes = np.asarray(args.pos_class_list)
    observed_by_id = {}
    pu_masks = []
    for cid, cids in enumerate(client_ids):
        labels = np.isin(y[cids], pos_classes).astype(np.int64)
        pos_ids = cids[labels == 1]
        labeled = client_seeds[cid].random(pos_ids.size) < float(args.label_freq)
        observed_ids = pos_ids[labeled]
        obs = {int(i): 0 for i in cids}
        obs.update({int(i): 1 for i in observed_ids})
        observed_by_id.update(obs)
        pu_masks.append((cids, obs, int(pos_ids.size), int(observed_ids.size)))

    fraction = float(args.score_fraction)
    if not 0 <= fraction < 0.5:
        raise ValueError('score_fraction must be in [0, 0.5)')
    clients, manifests = [], []
    for cid, (cids, obs, _n_hidden_pos, _n_observed) in enumerate(pu_masks):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1701, cid]))
        p0 = np.asarray([i for i in cids if obs[int(i)] == 1], dtype=np.int64)
        u0 = np.asarray([i for i in cids if obs[int(i)] == 0], dtype=np.int64)
        rng.shuffle(p0); rng.shuffle(u0)
        def split(arr):
            n_score = int(np.floor(fraction * len(arr)))
            if len(arr) > 0:
                n_score = min(n_score, len(arr) - 1)
            return np.sort(arr[n_score:]), np.sort(arr[:n_score])
        p_fit_ids, p_score_ids = split(p0)
        u_fit_ids, u_score_ids = split(u0)
        pi_d0, alpha_u, pi_dfit, prior_valid = _estimate_priors(
            len(p0), len(u0), float(args.label_freq), len(p_fit_ids), len(u_fit_ids))
        label_view = dict(obs)
        clients.append(ClientPUData(
            p_fit=PUIndexView(train_dataset, p_fit_ids, label_view),
            u_fit=PUIndexView(train_dataset, u_fit_ids, label_view),
            d_fit=PUIndexView(train_dataset, np.sort(np.concatenate([p_fit_ids, u_fit_ids])), label_view),
            p_score=PUIndexView(stable_dataset, p_score_ids, label_view),
            u_score=PUIndexView(stable_dataset, u_score_ids, label_view),
            u_cache=PUIndexView(stable_dataset, u_fit_ids, label_view),
            p_fit_ids=p_fit_ids, u_fit_ids=u_fit_ids,
            p_score_ids=p_score_ids, u_score_ids=u_score_ids,
            n_fit=int(len(p_fit_ids) + len(u_fit_ids)), alpha_u=alpha_u,
            pi_dfit=pi_dfit, prior_valid=bool(prior_valid)))
        manifests.append({
            'client_id': cid, 'client_ids': cids.tolist(),
            'p_fit_ids': p_fit_ids.tolist(), 'u_fit_ids': u_fit_ids.tolist(),
            'p_score_ids': p_score_ids.tolist(), 'u_score_ids': u_score_ids.tolist(),
            'n_p0': len(p0), 'n_u0': len(u0), 'n_p_fit': len(p_fit_ids),
            'n_u_fit': len(u_fit_ids), 'n_p_score': len(p_score_ids),
            'n_u_score': len(u_score_ids), 'pi_d0': pi_d0,
            'alpha_u': alpha_u, 'pi_dfit': pi_dfit, 'prior_valid': prior_valid,
        })
    return PUDataBundle(clients, stable_dataset, probe_pool_ids, manifests)


def make_loader(dataset, batch_size, shuffle=False, workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, drop_last=False, pin_memory=torch.cuda.is_available())


def configure_fednearpu_args(args):
    if getattr(args, 'model', None) != 'fednearpu_v2':
        return args
    expected = {
        'fedpu_cifar10': (500, 700, 300, 200, 50),
        'fedpu_imagenette': (100, 140, 60, 40, 10),
        'fedpu_usps': (100, 140, 60, 40, 10),
        'fedpu_letter': (260, 364, 156, 104, 26),
    }
    if args.dataset not in expected:
        raise ValueError(f'FedNearPU v2 has no probe specification for {args.dataset}')
    active, pool, anchor, rotating, refresh = expected[args.dataset]
    # The new protocol freezes these dataset-level probe counts. Existing `public_size`
    # remains the active probe count used in old command lines.
    if int(args.public_size) not in (1000, active):
        raise ValueError(f'FedNearPU v2 expects public_size={active} for {args.dataset}')
    args.public_size = active
    args.public_pool_size = int(getattr(args, 'public_pool_size', None) or pool)
    args.fednearpu_anchor_size = int(getattr(args, 'fednearpu_anchor_size', None) or anchor)
    args.fednearpu_rotate_size = int(getattr(args, 'fednearpu_rotate_size', None) or rotating)
    args.fednearpu_refresh_size = int(getattr(args, 'fednearpu_refresh_size', None) or refresh)
    if args.public_pool_size != pool:
        raise ValueError(f'FedNearPU v2 expects pool size {pool} for {args.dataset}')
    if int(args.parti_num) != 50:
        raise ValueError('FedNearPU v2 initial protocol fixes 50 clients')
    if not np.isclose(float(args.label_freq), 0.2):
        raise ValueError('FedNearPU v2 initial protocol fixes known SCAR labeling rate c=0.2')
    args.communication_epoch = 50
    args.label_freq = 0.2
    args.local_lr = 0.01
    args.reg = 1e-5
    args.eval_threshold = 0.5
    args.pseudo_refresh_gap = 10
    args.fednearpu_warmup_rounds = 20
    args.fednearpu_local_epochs = 5
    if args.Twarm == 10:
        args.Twarm = 20
    elif args.Twarm != 20:
        raise ValueError('FedNearPU v2 initial protocol fixes Twarm=20')
    args.online_ratio = 1.0
    args.fednearpu_top_m = int(getattr(args, 'fednearpu_top_m', 10))
    return args


class BinaryTestView(Dataset):
    def __init__(self, parent, targets):
        self.parent = parent
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, _ = self.parent[idx]
        return image, int(self.targets[idx])


def partition_pu_loaders_v2(train_dataset, test_dataset, setting, stable_dataset):
    if stable_dataset is None:
        raise ValueError('FedNearPU v2 requires a stable-transform dataset for scoring/probes')
    args = setting.args
    bundle = build_pu_data_bundle(train_dataset, stable_dataset, test_dataset, setting)
    setting.pu_v2_bundle = bundle
    setting.pu_v2_clients = bundle.clients
    setting.pu_v2_manifests = bundle.manifests
    train_loaders = [make_loader(c.d_fit, args.local_batch_size, shuffle=True, workers=0)
                     for c in bundle.clients]
    pool_loader = bundle.make_probe_loader(bundle.probe_pool_ids, args.local_batch_size)
    y_test = _get_targets(test_dataset)
    binary_test = np.isin(y_test, np.asarray(args.pos_class_list)).astype(np.int64)
    test_loader = DataLoader(BinaryTestView(test_dataset, binary_test),
                             batch_size=args.local_batch_size, shuffle=False, num_workers=0)
    priors = {cid: data.pi_dfit for cid, data in enumerate(bundle.clients)}
    return train_loaders, test_loader, pool_loader, priors
