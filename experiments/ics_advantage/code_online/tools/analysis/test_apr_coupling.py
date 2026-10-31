#!/usr/bin/env python3
"""CPU-only mathematical and persistence checks; no mmdet import required."""
import importlib.util
import io
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def direct_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proto = direct_module('audit_roi_proto', 'darth/models/utils/roi_proto_memory.py')
bank_module = direct_module('audit_roi_bank', 'darth/models/utils/roi_causal_bank.py')


def main():
    torch.manual_seed(721)
    x = torch.randn(19, 16, dtype=torch.float64, requires_grad=True)
    p = torch.randn(37, 16, dtype=torch.float64, requires_grad=True)
    e, _ = torch.linalg.qr(torch.randn(16, 2, dtype=torch.float64))
    sim = F.normalize(x, dim=1) @ F.normalize(p, dim=1).detach().t()
    old = (1.0 - sim.topk(3, dim=1).values.mean(dim=1)).mean()
    for metric in ('projected_l2', 'projected_cosine'):
        new, _, _ = proto.roi_proto_loss_with_gate(
            x, p, basis=e, beta=0.0, coupled_metric=metric)
        assert torch.equal(new, old), metric
    unit_x, unit_p = F.normalize(x, dim=1), F.normalize(p, dim=1)
    delta = unit_x[:, None, :] - unit_p[None, :, :]
    par = (delta @ e) @ e.t()
    beta = 0.35
    z = unit_x - beta * ((unit_x @ e) @ e.t())
    y = unit_p - beta * ((unit_p @ e) @ e.t())
    direct = (z[:, None, :] - y[None, :, :]).square().sum(-1)
    decomposition = ((1-beta)**2 * par.square().sum(-1)
                     + (delta-par).square().sum(-1))
    assert torch.allclose(direct, decomposition, atol=1e-12, rtol=1e-12)
    expected = 0.5 * direct.topk(3, dim=1, largest=False).values.mean()
    observed, _, _ = proto.roi_proto_loss_with_gate(
        x, p, basis=e, beta=beta, coupled_metric='projected_l2')
    assert torch.allclose(observed, expected, atol=1e-12, rtol=1e-12)
    loss, _, _ = proto.roi_proto_loss_with_gate(
        x, p, basis=e, beta=1.0, coupled_metric='projected_cosine')
    grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
    assert torch.isfinite(grad).all()
    released_gradient = (grad @ e).abs().max().item()
    assert released_gradient < 1e-12, released_gradient
    assert p.grad is None
    shift = torch.randn(19, 2, dtype=x.dtype) @ e.t()
    shifted, _, _ = proto.roi_proto_loss_with_gate(
        x + shift, p, basis=e, beta=1.0,
        coupled_metric='projected_cosine')
    assert torch.allclose(loss, shifted, atol=1e-12, rtol=1e-12)
    b = bank_module.CausalDirectionBank(16, num_pcs=2, min_samples=96)
    assert b.orthonormal_basis() is None
    b._update_pcs_from_batch(e.t().float())
    b._update_pcs_from_batch(torch.linalg.qr(torch.randn(16, 2))[0].t())
    basis = b.orthonormal_basis()
    assert torch.allclose(basis.t() @ basis, torch.eye(2), atol=1e-6)
    assert torch.equal(b.trim(x.detach().float(), alpha=.1), x.detach().float())
    # APR remains operative on fewer than min_samples RoIs with an existing bank.
    short_loss, _, _ = proto.roi_proto_loss_with_gate(
        x[:3], p, basis=basis.double(), beta=1.0,
        coupled_metric='projected_cosine')
    assert short_loss is not None and torch.isfinite(short_loss)
    m = proto.RoIPrototypeMemory(min_candidates=1, k_per_domain_class=4)
    labels = torch.ones(len(x), dtype=torch.long)
    scores = torch.ones(len(x))
    m.start_stage('overcast')
    assert m.compute_loss(x, labels, scores) is None
    m.collect('overcast', x.detach(), labels, scores)
    assert m.compute_loss(x, labels, scores) is None  # history-only
    m.finalize_stage('overcast')
    m.start_stage('rainy')
    m.collect('rainy', x[:2].detach(), labels[:2], scores[:2])
    buf = io.BytesIO()
    torch.save(m.runtime_state_dict(), buf)
    buf.seek(0)
    restored = proto.RoIPrototypeMemory(min_candidates=1, k_per_domain_class=4)
    restored.load_runtime_state_dict(torch.load(buf))
    assert restored.summary()['prototypes_per_domain_class'] == {'overcast': {1: 4}}
    assert restored.summary()['current_candidates'] == {1: 2}
    assert restored.current_domain == 'rainy'
    print('PASS: beta=0 bit-exact legacy loss; shrinkage decomposition; '
          'finite gradients; complement invariance and released gradient %.3g; '
          'post-EMA QR; small-RoI APR; history-only; rank-local recovery'
          % released_gradient)


if __name__ == '__main__':
    main()
