"""Audit BLAS-reduction sensitivity of the unchanged historical weather probe."""
from common import *
import numpy as np
import mkl
import evidence_cpu as evidence
import ics_probe_analyze as p


def main():
    rows = []
    for model in ['F', 'HF']:
        with historical_probe_blas():
            data, bank, Z, fit = evidence.load(model)
        E = np.load(str(EVIDENCE / 'part1' / model / 'bases/INT_00.npy'))
        reference = {r['basis']: float(r['weather_bacc_pp']) for r in read_csv(EVIDENCE / 'part1' / model / 'deletion.csv')}
        for threads in [4, 2, 1]:
            mkl.set_num_threads(threads)
            for space, X in [('Full', Z), ('Complement', Z - (Z @ E) @ E.T)]:
                result = 100 * p.weather_probe(X[fit], data['weather'][fit], X[~fit], data['weather'][~fit], 0)
                expected = reference['Full' if space == 'Full' else 'INT']
                row = dict(model=model, space=space, MKL_runtime_threads=threads,
                           actual=result, reference=expected, difference_pp=result - expected)
                rows.append(row)
                print(json.dumps(row), flush=True)
        mkl.set_num_threads(4)
    write_csv(ROOT / 'logs/probe_numerical_diagnostic.csv', rows)


if __name__ == '__main__':
    main()
