"""Read-only loss, actual key/ref trim activity, and frozen-bank audit."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmcv.parallel import is_module_wrapper
from ics_v2_runtime import file_sha


@HOOKS.register_module()
class ICSv2AuditHook(Hook):
    def __init__(self, root):
        self.root = Path(root)
        self.rows = []
        self.bank_update_calls = 0
        self.first_init_iter = None
        self.trim_calls = []
        self.active_iteration = 0

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        self.head = model.track_head
        self.rank = get_dist_info()[0]
        h = self.head
        assert not h.use_roi_proto_memory and h.roi_proto_memory is None
        assert not h.roi_proto_couple_ics and h.causal_min_samples == 32
        self.bank = h.causal_bank
        self.initial_pcs = self.bank.pcs.detach().clone() if self.bank is not None else None
        self.initial_updates = int(self.bank.num_updates.item()) if self.bank is not None else None
        self.fixed = h.causal_fixed_basis_path is not None
        if self.fixed:
            E = torch.from_numpy(np.load(h.causal_fixed_basis_path)).T.to(self.initial_pcs)
            assert torch.equal(E, self.initial_pcs) and bool(self.bank.initialized.item())
        if h.causal_reset_bank_on_resume:
            assert self.initial_updates == 0 and not bool(self.bank.initialized.item())
        if self.bank is not None:
            original_trim = self.bank.trim
            def trim(feats, *args, **kwargs):
                result = original_trim(feats, *args, **kwargs)
                self.trim_calls.append(dict(rois=int(feats.shape[0]), effective=not torch.equal(result, feats)))
                return result
            self.bank.trim = trim
            original_update = self.bank.update
            def update(*args, **kwargs):
                self.bank_update_calls += 1
                assert not self.fixed, 'A fixed bank update was called'
                before = bool(self.bank.initialized.item())
                result = original_update(*args, **kwargs)
                if not before and bool(self.bank.initialized.item()) and self.first_init_iter is None:
                    self.first_init_iter = self.active_iteration
                return result
            self.bank.update = update
            original_cov_update = self.bank.update_from_covariance
            def update_cov(*args, **kwargs):
                assert not self.fixed, 'A fixed covariance update was called'
                return original_cov_update(*args, **kwargs)
            self.bank.update_from_covariance = update_cov
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / ('iterations_rank%d.csv' % self.rank)
        self.path.write_text('')
        original = h.forward_train
        self.latest = None
        def call(*args, **kwargs):
            losses = original(*args, **kwargs)
            assert 'loss_roi_proto' not in losses
            self.latest = {k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v)}
            return losses
        h.forward_train = call
        self.save('start', runner)

    def before_train_iter(self, runner):
        self.active_iteration = runner.iter + 1
        self.trim_calls = []

    def save(self, tag, runner):
        h = self.head
        same = self.bank is None or torch.equal(self.bank.pcs, self.initial_pcs)
        if self.fixed:
            assert same and int(self.bank.num_updates.item()) == self.initial_updates
            assert self.bank_update_calls == 0
        n = len(self.rows)
        key = sum(int(r['key_trim_effective']) for r in self.rows)
        ref = sum(int(r['ref_trim_effective']) for r in self.rows)
        both = sum(int(r['key_trim_effective'] and r['ref_trim_effective']) for r in self.rows)
        losses = [r['trim_loss'] for r in self.rows if r['trim_loss'] is not None]
        record = dict(tag=tag, rank=self.rank, epoch=runner.epoch, iteration=runner.iter,
                      total_batches=n, APR_enabled=False, APR_history_loaded=False,
                      history_domains=0, coupling=False, ICS_enabled=self.bank is not None,
                      pca_mode=h.causal_pca_mode, alpha=h.causal_trim_alpha,
                      loss_weight=h.causal_loss_weight, min_samples=h.causal_min_samples,
                      num_pcs=h.causal_num_pcs, fixed_basis=self.fixed, fixed_pcs_bitwise_unchanged=same,
                      basis_path=h.causal_fixed_basis_path,
                      basis_sha256=file_sha(h.causal_fixed_basis_path) if self.fixed else None,
                      num_updates_start=self.initial_updates,
                      num_updates=int(self.bank.num_updates.item()) if self.bank is not None else None,
                      bank_update_calls=self.bank_update_calls, first_initialized_iteration=self.first_init_iter,
                      trim_active_batches=sum(int(r['trim_present']) for r in self.rows),
                      key_effective_batches=key, ref_effective_batches=ref, both_effective_batches=both,
                      key_effective_ratio=key / n if n else None,
                      ref_effective_ratio=ref / n if n else None,
                      both_effective_ratio=both / n if n else None,
                      loss_track_trim_mean=sum(losses) / len(losses) if losses else None)
        record['activation_below80_at50'] = n == 50 and self.bank is not None and min(key / n, ref / n) < .8
        (self.root / ('v2_%s_rank%d.json' % (tag, self.rank))).write_text(json.dumps(record, indent=2))
        runner.logger.info('[ICS_V2_AUDIT] %s', record)
        if record['activation_below80_at50']:
            runner.logger.warning('[ICS_V2_ALERT] key/ref activation below80%% in first50: %s', record)

    def after_train_iter(self, runner):
        assert self.latest is not None and all(np.isfinite(v) for v in self.latest.values())
        if self.bank is None:
            assert 'loss_track_trim' not in self.latest and not self.trim_calls
        if self.fixed:
            assert torch.equal(self.bank.pcs, self.initial_pcs)
            assert int(self.bank.num_updates.item()) == self.initial_updates and self.bank_update_calls == 0
        loss = float(runner.outputs['loss'].detach())
        assert np.isfinite(loss)
        calls = self.trim_calls
        assert len(calls) in [0, 2], calls
        row = dict(iteration=runner.iter + 1, epoch=runner.epoch + 1, total_loss=loss,
                   trim_loss=self.latest.get('loss_track_trim'), trim_aux_loss=self.latest.get('loss_track_aux_trim'),
                   trim_present='loss_track_trim' in self.latest,
                   key_rois=calls[0]['rois'] if calls else 0, ref_rois=calls[1]['rois'] if calls else 0,
                   key_trim_effective=calls[0]['effective'] if calls else False,
                   ref_trim_effective=calls[1]['effective'] if calls else False,
                   history_domains=0, APR_enabled=False,
                   num_updates=int(self.bank.num_updates.item()) if self.bank is not None else None)
        self.rows.append(row)
        with self.path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if len(self.rows) == 1:
                writer.writeheader()
            writer.writerow(row)
        if (runner.iter + 1) % 10 == 0:
            self.save('progress', runner)
        if runner.iter + 1 == 50:
            self.save('iteration50', runner)

    def after_train_epoch(self, runner):
        self.save('epoch%d' % (runner.epoch + 1), runner)

    def after_run(self, runner):
        if self.bank is not None:
            assert any(r['trim_present'] for r in self.rows)
        if self.bank is not None and not self.fixed:
            assert self.initial_updates == 0 and int(self.bank.num_updates.item()) > 0
            assert self.first_init_iter is not None
        self.save('end', runner)
