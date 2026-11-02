"""Ratio-of-sums restoration summaries and grouped A/B/C figures, no inference."""
from common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = ROOT / 'drift'
BASIS = ['FAP residual PCA', 'Random', 'Identity oracle', 'Label-free nuisance stage1']
LABELS = {
    'FAP residual PCA': 'FAP intervention',
    'Random': 'Random (mean ± SD, n=10)',
    'Identity oracle': 'Identity oracle',
    'Label-free nuisance stage1': 'Label-free nuisance (stage1)',
}
COLORS = dict(zip(BASIS, ['#0072B2', '#777777', '#D55E00', '#009E73']))
EXPECTED = {
    'F': {'FAP residual PCA': [5.6, 22.8, 52.6], 'Identity oracle': [43.5, 51.1, 61.7],
          'Random': [1.5, 5.0, 12.3], 'none_pp': 1.922},
    'HF': {'FAP residual PCA': [.2, 34.0, 55.5], 'Identity oracle': [52.9, 53.7, 53.5],
           'Random': [1.2, 5.7, 12.5], 'none_pp': 1.798},
}


def aggregate(rows):
    assert len(rows) == 3
    assert sorted(int(r['gallery_stage']) for r in rows) == [1, 2, 3]
    denominator = sum(float(r['forgetting_proxy']) for r in rows)
    assert denominator > 0
    reduction = sum(float(r['forgetting_reduction']) for r in rows)
    residual = sum(float(r['forgetting_restored']) for r in rows)
    assert abs(denominator - reduction - residual) < 1e-12
    return dict(recovered_pct=100 * reduction / denominator,
                residual_pp=100 * residual / 3, none_pp=100 * denominator / 3)


def summarize():
    paths = [EXPERIMENTS / 'idapr/A/A2_restoration.csv',
             EXPERIMENTS / 'idapr/A/A2_random_replicates.csv']
    records = [source_record(path) for path in paths]
    ordinary, random = [read_csv(path) for path in paths]
    summary, reps, checks = [], [], []
    for model in ['F', 'HF']:
        for k in [2, 8, 16]:
            for basis in BASIS:
                if basis == 'Random':
                    values = []
                    for replicate in range(10):
                        rows = [r for r in random if r['model'] == model and r['query_stage'] == '4'
                                and int(r['k']) == k and int(r['replicate']) == replicate]
                        value = aggregate(rows)
                        reps.append(dict(model=model, basis=basis, k=k, replicate=replicate, **value))
                        values.append(value)
                    recovered = np.array([r['recovered_pct'] for r in values])
                    row = dict(model=model, basis=basis, k=k, recovered_pct=float(recovered.mean()),
                               random_sd=float(recovered.std(ddof=0)), random_min=float(recovered.min()),
                               random_max=float(recovered.max()),
                               residual_pp=float(np.mean([r['residual_pp'] for r in values])),
                               none_pp=values[0]['none_pp'])
                else:
                    rows = [r for r in ordinary if r['model'] == model and r['query_stage'] == '4'
                            and r['basis'] == basis and int(r['k']) == k]
                    value = aggregate(rows)
                    row = dict(model=model, basis=basis, k=k, recovered_pct=value['recovered_pct'],
                               random_sd='', random_min='', random_max='',
                               residual_pp=value['residual_pp'], none_pp=value['none_pp'])
                summary.append(row)
                if basis in EXPECTED[model]:
                    expected = EXPECTED[model][basis][[2, 8, 16].index(k)]
                    error = abs(row['recovered_pct'] - expected)
                    checks.append(dict(model=model, basis=basis, k=k, metric='recovered_pct',
                                       expected=expected, actual=row['recovered_pct'],
                                       absolute_error=error, pass_check=bool(error < .1)))
                    assert error < .1, checks[-1]
            baseline = summary[-1]['none_pp']
            expected = EXPECTED[model]['none_pp']
            checks.append(dict(model=model, basis='None', k=k, metric='none_pp', expected=expected,
                               actual=baseline, absolute_error=abs(baseline - expected),
                               pass_check=bool(abs(baseline - expected) < .001)))
            assert abs(baseline - expected) < .001, checks[-1]
            same = [r['none_pp'] for r in summary if r['model'] == model and r['k'] == k]
            assert np.ptp(same) < 1e-12
    verify_sources(records)
    write_csv(OUT / 'drift_restoration_summary.csv', summary)
    write_csv(OUT / 'drift_random_replicates.csv', reps)
    write_csv(OUT / 'selfchecks.csv', checks)
    write_json(OUT / 'protocol.json', dict(status='PASS', source_files=records,
        recovery='100 * sum(reduction over old domains) / sum(forgetting_proxy over old domains)',
        residual='100 * sum(forgetting_restored over old domains) / 3',
        random='One ratio-of-sums per replicate; ten-replicate mean, population SD (ddof=0), min/max',
        query_stage=4, gallery_stages=[1, 2, 3], dimensions=[2, 8, 16], checks=len(checks),
        all_selfchecks_pass=True, no_averaging_of_domain_ratios=True))
    print('DRIFT_SELFCHECK_PASS', len(checks), flush=True)
    return summary, reps


def draw(summary, reps, version, size):
    single = size == 'single'
    bases = BASIS if version == 'C' else BASIS[:3]
    fig, axes = plt.subplots(1, 2, sharey=True,
                             figsize=(3.3, 4.6) if single else (7., 3.45))
    width = .78 / len(bases)
    use_residual = version == 'B'
    field = 'residual_pp' if use_residual else 'recovered_pct'
    for ax, model in zip(axes, ['F', 'HF']):
        ax.axvspan(-.47, .47, color='#eeeeee', zorder=0)
        ax.axhline(0, color='black', lw=.8, zorder=1)
        for j, basis in enumerate(bases):
            rows = [next(r for r in summary if r['model'] == model and r['basis'] == basis and r['k'] == k)
                    for k in [2, 8, 16]]
            vals = [r[field] for r in rows]
            positions = np.arange(3) + (j - (len(bases) - 1) / 2) * width
            errors = None
            if basis == 'Random':
                errors = [float(np.std([r[field] for r in reps if r['model'] == model and r['k'] == k], ddof=0))
                          for k in [2, 8, 16]]
            ax.bar(positions, vals, width=width * .93, color=COLORS[basis], edgecolor='white', linewidth=.25,
                   yerr=errors, error_kw=dict(capsize=2, elinewidth=.7, capthick=.7), zorder=3)
            offset = .018 if use_residual else 1.0
            for i, (x, y) in enumerate(zip(positions, vals)):
                error = errors[i] if errors is not None else 0.
                ax.text(x, y + error + offset, ('%.2f' if use_residual else '%.1f') % y,
                        ha='center', va='bottom', rotation=90 if single or version in ['B', 'C'] else 0,
                        fontsize=9)
        none_pp = rows[0]['none_pp']
        if use_residual:
            ax.axhline(none_pp, color='black', lw=.9, ls='--', zorder=2)
        ax.set_title(model + ('; none %.3f pp' % none_pp if use_residual and not single else ''), fontsize=10)
        ax.text(.03, .98, 'ICS uses\nk = 2' if single else 'ICS uses k = 2',
                transform=ax.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['2', '8', '16'], fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.set_xlim(-.55, 2.52)
        ax.set_ylim(0, 2.53) if use_residual else ax.set_ylim(-6, 88 if single else 80)
        ax.grid(axis='y', color='.88', linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel('Residual forgetting (pp)' if use_residual else 'Recovered forgetting (%)', fontsize=9)
    handles = [Patch(facecolor=COLORS[b], label=LABELS[b]) for b in bases]
    if use_residual:
        handles.append(Line2D([0], [0], color='black', ls='--', label='Unrestored baseline'))
    ncol = 1 if single else 2
    fig.legend(handles=handles, loc='lower center', ncol=ncol, frameon=False, fontsize=9,
               bbox_to_anchor=(.5, .005), columnspacing=1., handlelength=1.3)
    fig.text(.55, .265 if single else .19, 'Subspace dimension k', ha='center', fontsize=9)
    fig.subplots_adjust(left=.17 if single else .09, right=.985, bottom=.34 if single else .30,
                        top=.92, wspace=.12 if single else .15)
    stem = OUT / ('drift_restoration_%s_%s' % (version, size))
    fig.savefig(str(stem) + '.pdf')
    fig.savefig(str(stem) + '.png', dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    summary, reps = summarize()
    for version in ['A', 'B', 'C']:
        for size in ['single', 'double']:
            draw(summary, reps, version, size)
            print('FIGURE', version, size, flush=True)


if __name__ == '__main__':
    main()
