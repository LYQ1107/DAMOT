"""Requested CPU release-space checks; torch and the standard library only."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[2]
    new = module(code / 'darth/models/utils/roi_proto_memory.py', 'release_memory_test')
    old = module(code.parent / 'code_original/darth/models/utils/roi_proto_memory.py', 'old_memory_test')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    records = []
    for dtype in [torch.float32, torch.float64]:
        raw = torch.randn(64, 256, dtype=dtype, requires_grad=True)
        prototypes = torch.randn(300, 256, dtype=dtype)
        full, _, _ = new.roi_proto_loss_with_gate(raw, prototypes)
        empty, _, _ = new.roi_proto_loss_with_gate(raw, prototypes,
                            space_mode='release', space_basis=torch.empty(256, 0, dtype=dtype))
        assert torch.equal(full, empty), 'Empty released basis must use the exact full branch'
        for k in [2, 16]:
            E = torch.linalg.qr(torch.randn(256, k, dtype=dtype), mode='reduced')[0]
            normalized = F.normalize(raw.detach(), dim=1).requires_grad_()
            proto_unit = F.normalize(prototypes, dim=1).detach()
            loss, _, _ = new.roi_proto_release_normalized(normalized, proto_unit, E)
            gradient = torch.autograd.grad(loss, normalized)[0]
            in_component = gradient @ E
            max_component = float(in_component.abs().max())
            relative_norm = float(in_component.norm() / gradient.norm())
            assert max_component <= 1e-6 and relative_norm <= 1e-6
            assert torch.isfinite(gradient).all() and torch.isfinite(loss)
            training_loss, _, _ = new.roi_proto_loss_with_gate(raw, prototypes,
                                                space_mode='release', space_basis=E)
            raw_gradient = torch.autograd.grad(training_loss, raw, retain_graph=True)[0]
            assert torch.isfinite(raw_gradient).all()
            # Default-off function equality supplements the full-model U2.
            for beta in [0., 1.]:
                for metric in ['projected_l2', 'projected_cosine']:
                    ref = old.roi_proto_loss_with_gate(raw, prototypes, basis=E,
                                                       beta=beta, coupled_metric=metric)
                    actual = new.roi_proto_loss_with_gate(raw, prototypes, basis=E,
                                                          beta=beta, coupled_metric=metric,
                                                          space_mode='full')
                    assert all(torch.equal(a, b) for a, b in zip(ref, actual))
            records.append(dict(dtype=str(dtype), k=k, empty_basis_bitwise_full=True,
                                normalized_node_gradient_in_E_max=max_component,
                                normalized_node_gradient_in_E_relative_norm=relative_norm,
                                all_gradients_finite=True, full_function_bitwise_original=True))
    result = dict(status='passed', tests=['U1c', 'U1d', 'U1e'], records=records,
                  gradient_node='after L2 normalization, before its Jacobian (as requested)')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
