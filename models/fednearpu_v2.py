"""Integrated repair: full-time nnPU + ramped neighbor distillation + weak global fallback."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.utils.federated_model import FederatedModel
from models.utils.pu_objectives import nnpu_marginal_loss, cached_distillation_loss, solve_pu_brier_weights


class RotatingProbeManager:
    def __init__(self, pool_ids, anchor_size, rotating_size, refresh_size, seed):
        ids = np.asarray(pool_ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Public probe pool contains duplicate IDs')
        rng = np.random.default_rng(seed)
        self.ids = rng.permutation(ids)
        self.anchor_size = int(anchor_size)
        self.rotating_size = int(rotating_size)
        self.refresh_size = int(refresh_size)
        self.anchor_ids = self.ids[:self.anchor_size].copy()
        self.rotation_pool = self.ids[self.anchor_size:].copy()
        if self.rotating_size > len(self.rotation_pool):
            raise ValueError('Rotating probe set is larger than the non-anchor pool')
        if self.refresh_size > self.rotating_size and self.rotating_size:
            raise ValueError('One probe refresh cannot replace more samples than the rotating set')
        self.cursor = 0
        self.active_rotating_ids = self.rotation_pool[:self.rotating_size].copy()

    @property
    def active_ids(self):
        return np.concatenate([self.anchor_ids, self.active_rotating_ids])

    def rotate(self):
        if self.rotating_size == 0 or self.refresh_size == 0:
            return self.active_ids
        n = len(self.rotation_pool)
        old = self.cursor
        self.cursor = (self.cursor + self.refresh_size) % n
        positions = (np.arange(self.rotating_size) + self.cursor) % n
        next_ids = self.rotation_pool[positions]
        removed = set(map(int, self.active_rotating_ids[:self.refresh_size]))
        if removed.intersection(map(int, next_ids)):
            raise RuntimeError('Probe rotation immediately reinserted a removed sample')
        self.active_rotating_ids = next_ids.copy()
        return self.active_ids


class FedNearPUV2(FederatedModel):
    NAME = 'fednearpu_v2'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)
        if float(args.online_ratio) != 1.0:
            raise ValueError('FedNearPUV2 requires full client participation (online_ratio=1)')
        if args.parti_num < 2:
            raise ValueError('FedNearPUV2 requires at least two clients')
        for net in self.nets_list:
            net.cpu()
        self.global_net = copy.deepcopy(self.nets_list[0]).cpu()
        self.pu_v2_clients = None
        self.pu_v2_manifests = None
        self.probe_bundle = None
        self.probe_manager = None
        self.client_cache = {}
        self.client_teacher_weights = {}
        self.client_refresh_metadata = {}
        self.top_m = min(int(args.fednearpu_top_m), int(args.parti_num) - 1)
        self.tau = float(args.teacher_tau)
        self.experiment_mode = str(args.v2_experiment_mode)
        self.distill_lambda = float(args.v2_distill_lambda)
        self.distill_ramp_rounds = max(1, int(args.v2_distill_ramp_rounds))
        self.sample_gamma = float(args.sample_gamma)
        self.Twarm = int(args.fednearpu_warmup_rounds)
        self.local_epoch = int(args.fednearpu_local_epochs)
        self.score_min_p = int(args.score_min_p)
        self.score_min_u = int(args.score_min_u)
        self.batch_size_p = int(args.pu_batch_size_p)
        self.batch_size_d = int(args.pu_batch_size_d)
        self.batch_size_u = int(args.pu_batch_size_u)
        self.refresh_rounds = {self.Twarm}
        gap = int(args.pseudo_refresh_gap)
        if gap <= 0:
            raise ValueError('pseudo_refresh_gap must be positive')
        round_id = self.Twarm + gap
        while round_id < int(args.communication_epoch):
            self.refresh_rounds.add(round_id)
            round_id += gap
        self.local_rngs = [np.random.default_rng(np.random.SeedSequence([int(args.seed), 2203, i]))
                           for i in range(int(args.parti_num))]
        self.refresh_log = []

    def attach_pu_data(self, bundle):
        self.probe_bundle = bundle
        self.pu_v2_clients = bundle.clients
        self.pu_v2_manifests = bundle.manifests
        self.probe_manager = RotatingProbeManager(
            bundle.probe_pool_ids,
            self.args.fednearpu_anchor_size,
            self.args.fednearpu_rotate_size,
            self.args.fednearpu_refresh_size,
            int(self.args.seed) + 991)
        active = self.probe_manager.active_ids
        expected = int(self.args.public_size)
        if len(active) != expected:
            raise ValueError(f'Probe config active size {len(active)} != public_size {expected}')
        return self

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0]).cpu()
        state = self.global_net.state_dict()
        for net in self.nets_list:
            net.load_state_dict(state)
            net.cpu()

    def loc_update(self, priloader_list):
        if self.pu_v2_clients is None or self.probe_bundle is None:
            raise RuntimeError('PU v2 data bundle was not attached before training')
        self.trainloaders = priloader_list
        online_clients = list(range(int(self.args.parti_num)))
        self.online_clients = online_clients
        total_loss = 0.0
        self.current_pu_diagnostics = []
        for cid in online_clients:
            loss = self._train_client(cid)
            total_loss += loss
        mean_loss = total_loss / max(1, len(online_clients))
        diagnostic_path = Path(self.args.run_output_dir) / 'pu_round_diagnostics.jsonl'
        with diagnostic_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'round': int(self.epoch_index) + 1,
                                     'clients': self.current_pu_diagnostics}) + '\n')
        completed_round = int(self.epoch_index) + 1
        if completed_round in self.refresh_rounds:
            if completed_round >= int(self.args.communication_epoch):
                raise RuntimeError('Attempted an unused final-round teacher refresh')
            self._refresh_teachers(completed_round)
        self._aggregate_fit_weighted()
        return mean_loss

    @staticmethod
    def _next_batch(dataset, batch_size, rng):
        if len(dataset) == 0:
            return None, None, None
        indices = rng.integers(0, len(dataset), size=int(batch_size))
        rows = [dataset[int(i)] for i in indices]
        images = torch.stack([row[0] for row in rows])
        ids = np.asarray([int(row[2]) for row in rows], dtype=np.int64)
        return images, ids, indices

    def _train_client(self, cid):
        data = self.pu_v2_clients[cid]
        net = self.nets_list[cid].to(self.device)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9,
                                    weight_decay=float(self.args.reg))
        if data.n_fit == 0:
            net.cpu()
            return 0.0
        steps_per_epoch = int(np.ceil(data.n_fit / 32.0))
        steps = int(self.local_epoch) * steps_per_epoch
        rng = self.local_rngs[cid]
        loss_sum = 0.0
        risk_sum = {}
        risk_steps = 0
        for step in range(steps):
            x_p, _, _ = self._next_batch(data.p_fit, self.batch_size_p, rng)
            if x_p is None or data.pi_dfit is None:
                # Zero-P clients have no nnPU term and no teacher cache until warm-up ends.
                if self.epoch_index < self.Twarm or self.experiment_mode == 'nnpu_only':
                    continue
                x_u, ids_u, _ = self._next_batch(data.u_fit, self.batch_size_u, rng)
                if x_u is None:
                    continue
                targets = self._cached_targets(cid, ids_u)
                q = torch.as_tensor(targets[:, 0], device=self.device, dtype=torch.float32)
                r = torch.as_tensor(targets[:, 1], device=self.device, dtype=torch.float32)
                logits_u = net(x_u.to(self.device, non_blocking=True)).reshape(-1)
                loss = self._distill_weight(cid) * cached_distillation_loss(logits_u, q, r)
            else:
                x_d, _, _ = self._next_batch(data.d_fit, self.batch_size_d, rng)
                logits_p = net(x_p.to(self.device, non_blocking=True)).reshape(-1)
                logits_d = net(x_d.to(self.device, non_blocking=True)).reshape(-1)
                loss, parts = nnpu_marginal_loss(logits_p, logits_d, data.pi_dfit)
                parts['train_D_pred_positive_fraction'] = float((torch.sigmoid(logits_d.detach()) >= 0.5).float().mean())
                parts['train_D_count'] = int(logits_d.numel())
                parts['train_D_score_mean'] = float(torch.sigmoid(logits_d.detach()).mean())
                risk_steps += 1
                for key, value in parts.items():
                    risk_sum[key] = risk_sum.get(key, 0.0) + float(value)
                if self.epoch_index >= self.Twarm and self.experiment_mode != 'nnpu_only':
                    x_u, ids_u, _ = self._next_batch(data.u_fit, self.batch_size_u, rng)
                    if x_u is not None:
                        targets = self._cached_targets(cid, ids_u)
                        q = torch.as_tensor(targets[:, 0], device=self.device, dtype=torch.float32)
                        r = torch.as_tensor(targets[:, 1], device=self.device, dtype=torch.float32)
                        logits_u = net(x_u.to(self.device, non_blocking=True)).reshape(-1)
                        loss = loss + self._distill_weight(cid) * cached_distillation_loss(logits_u, q, r)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f'non-finite loss at round={self.epoch_index}, client={cid}, step={step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        self.current_pu_diagnostics.append({
            'client': cid, 'prior': data.pi_dfit, 'steps': risk_steps,
            **{key: value / max(1, risk_steps) for key, value in risk_sum.items()}})
        net.cpu()
        return loss_sum / max(1, steps)

    def _distill_weight(self, cid):
        elapsed = max(0, int(self.epoch_index) - self.Twarm + 1)
        ramp = min(1.0, elapsed / float(self.distill_ramp_rounds))
        fallback_scale = 1.0
        meta = self.client_refresh_metadata.get(cid, {})
        if self.experiment_mode in ('nnpu_global_fallback', 'nnpu_combined') and meta.get('fallback'):
            fallback_scale = 0.25
        return self.distill_lambda * ramp * fallback_scale

    def _cached_targets(self, cid, ids):
        cache = self.client_cache.get(cid)
        if cache is None:
            raise RuntimeError(f'Missing PP cache for client {cid}, round {self.epoch_index}')
        values = []
        for idx in ids:
            pair = cache.get(int(idx))
            if pair is None:
                raise RuntimeError(f'Cache misses client={cid}, sample_id={idx}')
            values.append(pair)
        return np.asarray(values, dtype=np.float32)

    def _snapshot_local_models(self):
        snapshots = []
        for net in self.nets_list:
            snapshots.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        return snapshots

    def _make_scratch(self, state):
        scratch = copy.deepcopy(self.nets_list[0]).to(self.device)
        scratch.load_state_dict(state)
        scratch.eval()
        return scratch

    def _features(self, net, images):
        if hasattr(net, 'probe_features'):
            h = net.probe_features(images, layer='penultimate')
        elif hasattr(net, 'features'):
            h = net.features(images)
        else:
            raise RuntimeError(f'{net.__class__.__name__} lacks features/probe_features API')
        return h.flatten(1) if h.ndim > 2 else h

    @torch.no_grad()
    def _compute_prototypes(self, snapshots, probe_loader):
        prototypes = []
        scratch = self._make_scratch(snapshots[0])
        for state in snapshots:
            scratch.load_state_dict(state)
            scratch.eval()
            total, count = None, 0
            for batch in probe_loader:
                images = batch[0].to(self.device, non_blocking=True)
                features = self._features(scratch, images)
                batch_sum = features.sum(0)
                total = batch_sum if total is None else total + batch_sum
                count += int(features.shape[0])
            if not count:
                raise RuntimeError('Active public probe set is empty')
            prototypes.append((total / count).detach().cpu().numpy().astype(np.float64))
        scratch.cpu()
        return prototypes

    @staticmethod
    def _loader_predictions(predictions, loader, cid, device):
        ids_out, chunks = [], []
        for images, _observed, ids in loader:
            images = images.to(device, non_blocking=True)
            probs = torch.sigmoid(predictions(images).reshape(-1)).detach().cpu().numpy()
            ids_out.extend(map(int, ids.numpy().tolist()))
            chunks.extend(map(float, probs.tolist()))
        return np.asarray(ids_out, dtype=np.int64), np.asarray(chunks, dtype=np.float64)

    @torch.no_grad()
    def _predict_matrix(self, snapshots, candidate_ids, loader, scratch):
        all_ids, columns = None, []
        for teacher_id in candidate_ids:
            scratch.load_state_dict(snapshots[teacher_id])
            scratch.eval()
            ids, probs = self._loader_predictions(scratch, loader, teacher_id, self.device)
            if all_ids is None:
                all_ids = ids
            elif not np.array_equal(all_ids, ids):
                raise RuntimeError('Teacher prediction sample IDs do not align')
            columns.append(probs)
        n = 0 if all_ids is None else len(all_ids)
        matrix = np.stack(columns, axis=1) if columns else np.empty((n, 0), dtype=np.float64)
        return all_ids if all_ids is not None else np.empty(0, dtype=np.int64), matrix

    def _select_neighbors(self, cid, prototypes):
        target = prototypes[cid]
        distances = []
        for other_id, proto in enumerate(prototypes):
            if other_id == cid:
                continue
            distances.append((float(np.linalg.norm(target - proto)), other_id))
        distances.sort(key=lambda pair: (pair[0], pair[1]))
        return [other_id for _, other_id in distances[:self.top_m]]

    def _refresh_teachers(self, completed_round):
        if self.probe_manager is None:
            raise RuntimeError('Rotating probe manager was not attached')
        if completed_round > self.Twarm:
            self.probe_manager.rotate()
        active_ids = self.probe_manager.active_ids
        batch_size = max(1, min(128, int(self.args.local_batch_size)))
        probe_loader = self.probe_bundle.make_probe_loader(active_ids, batch_size)
        snapshots = self._snapshot_local_models()
        prototypes = self._compute_prototypes(snapshots, probe_loader)
        pending_cache, pending_weights, pending_meta = {}, {}, {}
        scratch = self._make_scratch(snapshots[0])
        for cid, data in enumerate(self.pu_v2_clients):
            candidates = self._select_neighbors(cid, prototypes)
            p_loader = DataLoader(data.p_score, batch_size=batch_size, shuffle=False, num_workers=0)
            u_loader = DataLoader(data.u_score, batch_size=batch_size, shuffle=False, num_workers=0)
            _, pred_p = self._predict_matrix(snapshots, candidates, p_loader, scratch)
            _, pred_u = self._predict_matrix(snapshots, candidates, u_loader, scratch)
            weights, solve = solve_pu_brier_weights(
                pred_u, pred_p, data.alpha_u, tau=self.tau,
                min_p=self.score_min_p, min_u=self.score_min_u)
            fit_loader = DataLoader(data.u_cache, batch_size=batch_size, shuffle=False, num_workers=0)
            fit_ids, pred_fit = self._predict_matrix(snapshots, candidates, fit_loader, scratch)
            q = pred_fit @ weights if pred_fit.shape[0] else np.empty(0, dtype=np.float64)
            r = np.abs(2.0 * q - 1.0) ** self.sample_gamma
            fallback_teacher = None
            if solve['fallback'] and self.experiment_mode in ('nnpu_global_fallback', 'nnpu_combined') and len(fit_ids):
                scratch.load_state_dict(self.global_net.state_dict())
                scratch.eval()
                _, global_probs = self._loader_predictions(scratch, fit_loader, -1, self.device)
                q = global_probs
                r = np.maximum(np.abs(2.0 * q - 1.0) ** self.sample_gamma, 0.25)
                fallback_teacher = 'previous_global_model'
            if len(fit_ids) != len(data.u_fit_ids) or not np.array_equal(fit_ids, data.u_fit_ids):
                raise RuntimeError(f'Cache IDs do not exactly cover U_fit for client {cid}')
            pending_cache[cid] = {int(idx): (float(qv), float(rv))
                                  for idx, qv, rv in zip(fit_ids, q, r)}
            pending_weights[cid] = {int(j): float(w) for j, w in zip(candidates, weights)}
            pending_meta[cid] = {
                'round': completed_round, 'candidate_ids': candidates,
                'weights': weights.tolist(), 'probe_ids': list(map(int, active_ids)),
                'n_p_score': len(data.p_score), 'n_u_score': len(data.u_score),
                'solver': solve, 'fallback': solve['fallback'], 'reason': solve['reason'],
                'fallback_teacher': fallback_teacher,
                'q_mean': float(q.mean()) if q.size else None,
                'q_std': float(q.std()) if q.size else None,
                'r_mean': float(r.mean()) if r.size else None,
            }
        scratch.cpu()
        self.client_cache = pending_cache
        self.client_teacher_weights = pending_weights
        self.client_refresh_metadata = pending_meta
        self.refresh_log.append({'round': completed_round, 'active_probe_ids': list(map(int, active_ids)),
                                 'clients': pending_meta})
        print(f'[FedNearPUV2] round={completed_round} refreshed {len(pending_cache)} PP caches; '
              f'fallbacks={sum(v["fallback"] for v in pending_meta.values())}')

    def _aggregate_fit_weighted(self):
        counts = np.asarray([max(0, d.n_fit) for d in self.pu_v2_clients], dtype=np.float64)
        if counts.sum() <= 0:
            raise RuntimeError('All clients have empty fit partitions')
        weights = counts / counts.sum()
        source_states = [net.state_dict() for net in self.nets_list]
        global_state = self.global_net.state_dict()
        for key in global_state:
            values = [state[key].detach().cpu() for state in source_states]
            if values[0].is_floating_point():
                acc = torch.zeros_like(values[0], dtype=torch.float64)
                for weight, value in zip(weights, values):
                    acc.add_(value.to(torch.float64), alpha=float(weight))
                global_state[key] = acc.to(dtype=values[0].dtype)
            else:
                winner = int(np.argmax(counts))
                global_state[key] = values[winner].clone()
        self.global_net.load_state_dict(global_state)
        global_copy = self.global_net.state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_copy)
            net.cpu()

    def final_artifacts(self, output_dir, round_metrics):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_net.state_dict(), output / 'final_global_model.pt')
        (output / 'round_metrics.json').write_text(json.dumps(round_metrics, ensure_ascii=False, indent=2))
        (output / 'split_manifest.json').write_text(json.dumps(self.pu_v2_manifests, ensure_ascii=False, indent=2))
        config = {k: v for k, v in vars(self.args).items()
                  if isinstance(v, (str, int, float, bool, list, tuple, type(None)))}
        config['v2_experiment_mode'] = self.experiment_mode
        config['v2_distill_lambda'] = self.distill_lambda
        config['v2_distill_ramp_rounds'] = self.distill_ramp_rounds
        config['refresh_log'] = self.refresh_log
        (output / 'run_configuration.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
