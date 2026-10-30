"""Read-only loss/basis audits, plus the explicitly requested 50-step smoke stop."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook, get_dist_info


class SmokeComplete(Exception):
    pass


@HOOKS.register_module()
class APRSpaceAuditHook(Hook):
    def __init__(self, root, require_nonzero=False, expected_mode=None, expected_coupling=None):
        self.root = Path(root)
        self.require_nonzero = bool(require_nonzero)
        self.expected_mode = expected_mode
        self.expected_coupling = expected_coupling
        self.rows = []

    def before_run(self, runner):
        self.model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        self.head = self.model.track_head
        self.rank = get_dist_info()[0]
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / ('apr_iterations_rank%d.csv' % self.rank)
        self.path.write_text('')
        self.initial_updates = int(self.head.causal_bank.num_updates.item())
        self.initial_basis = (self.head.roi_proto_space_basis.detach().clone()
                              if hasattr(self.head, 'roi_proto_space_basis') else None)
        assert len(self.head.roi_proto_memory._history) == 3
        if self.expected_mode is not None:
            assert self.head.roi_proto_space_mode == self.expected_mode
        if self.expected_coupling is not None:
            assert self.head.roi_proto_couple_ics == self.expected_coupling
        assert not any('roi_proto_space_basis' in key for key in self.model.state_dict())
        original = self.head.forward_train
        self.latest = None
        def call(*args, **kwargs):
            losses = original(*args, **kwargs)
            value = losses.get('loss_roi_proto')
            value = None if value is None else float(value.detach())
            self.latest = dict(weighted_APR_loss=value,
                               trim_loss_present='loss_track_trim' in losses,
                               stats=dict(self.head.roi_proto_memory.last_stats))
            return losses
        self.head.forward_train = call
        self.save('start', runner)

    def save(self, tag, runner):
        h = self.head
        unchanged = (self.initial_basis is None or
                     torch.equal(h.roi_proto_space_basis.detach(), self.initial_basis))
        assert unchanged, 'The fixed released basis changed'
        record = dict(tag=tag, rank=self.rank, epoch=runner.epoch,
                      iteration=runner.iter, total_batches=len(self.rows),
                      history_domains=len(h.roi_proto_memory._history),
                      history_names=list(h.roi_proto_memory._history),
                      mode=h.roi_proto_space_mode, coupling=h.roi_proto_couple_ics,
                      coupling_beta=h.roi_proto_couple_beta,
                      coupling_basis=h.roi_proto_couple_basis,
                      coupling_metric=h.roi_proto_couple_metric,
                      basis_sha256=h.roi_proto_space_basis_sha256,
                      fixed_basis_unchanged=unchanged,
                      basis_not_persisted_in_checkpoint=True,
                      num_updates_start=self.initial_updates,
                      num_updates=int(h.causal_bank.num_updates.item()))
        path = self.root / ('apr_audit_%s_rank%d.json' % (tag, self.rank))
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(path)
        runner.logger.info('[APR_SPACE_AUDIT] %s', record)

    def after_train_iter(self, runner):
        assert self.latest is not None
        h = self.head
        value = self.latest['weighted_APR_loss']
        if self.require_nonzero:
            assert value is not None and np.isfinite(value) and value != 0.0, self.latest
        if value is not None:
            assert np.isfinite(value)
        st = self.latest['stats']
        reference = st.get('full_reference_loss') if h.roi_proto_space_mode == 'release' else None
        row = dict(iteration=runner.iter + 1, epoch=runner.epoch + 1,
                   weighted_APR_loss=value, raw_APR_loss=st.get('loss') if value is not None else None,
                   paired_full_reference_weighted=(float(reference) * h.roi_proto_loss_weight
                                                   if reference is not None and value is not None else None),
                   mean_in_share=st.get('mean_in_share'),
                   selected_rois=st.get('selected_rois'), prototypes=st.get('prototypes'),
                   history_domains=len(h.roi_proto_memory._history),
                   total_loss=float(runner.outputs['loss'].detach()),
                   trim_loss_present=self.latest['trim_loss_present'],
                   num_updates=int(h.causal_bank.num_updates.item()))
        self.rows.append(row)
        with self.path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if len(self.rows) == 1:
                writer.writeheader()
            writer.writerow(row)
        if (runner.iter + 1) % 20 == 0:
            self.save('progress', runner)

    def after_train_epoch(self, runner):
        self.save('epoch%d' % (runner.epoch + 1), runner)

    def after_run(self, runner):
        self.save('end', runner)


@HOOKS.register_module()
class StopSmokeAfterIterationsHook(Hook):
    def __init__(self, iterations=50):
        self.iterations = int(iterations)

    def after_train_iter(self, runner):
        if runner.iter + 1 >= self.iterations:
            for hook in runner.hooks:
                if isinstance(hook, APRSpaceAuditHook):
                    hook.save('smoke_end', runner)
            raise SmokeComplete('Requested smoke run completed %d updates' % self.iterations)
