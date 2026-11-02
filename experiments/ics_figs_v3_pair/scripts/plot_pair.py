"""CSV-only HF drift-share and restoration figures for one-column layouts."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCE = REPO / 'experiments/ics_figs_v3_energy/energy_summary.csv'
RESTORATION = REPO / 'experiments/ics_figs_v3/drift_summary.csv'
BASE_COMMIT = '5473ff81b871819e125e68d03f499c6e3b98ecef'
ORDER = ['Intervention-sensitive', 'Random', 'Identity oracle']
COLORS = ['#0072B2', '#777777', '#D55E00']
KS = [2, 8, 16]

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    assert os.environ.get(key) == '4'
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(ax, index, metric, narrow=False):
    field = 'drift_energy_pct' if metric == 'energy' else 'recovered_forgetting_pct'
    sd_field = 'random_energy_sd' if metric == 'energy' else 'random_recovery_sd'
    ax.axvspan(-.48, .48, color='#eeeeee', zorder=0)
    ax.axhline(0, color='black', lw=.7, zorder=2)
    width = .80 / 3
    artists = []
    for j, (basis, color) in enumerate(zip(ORDER, COLORS)):
        selected = [index[(basis, k)] for k in KS]
        y = [float(r[field]) for r in selected]
        sd = [float(r[sd_field]) for r in selected] if basis == 'Random' else [0.] * 3
        x = np.arange(3) + (j-1) * width
        ax.bar(x, y, width=width*.94, color=color, edgecolor='white', linewidth=.25,
               yerr=sd if basis == 'Random' else None,
               error_kw=dict(capsize=1.8 if narrow else 2, elinewidth=.65, capthick=.65), zorder=3)
        for xx, yy, ee in zip(x, y, sd):
            artists.append(ax.text(xx, yy + ee + (.5 if metric == 'energy' else 1.),
                                   '%.1f' % yy, ha='center', va='bottom', fontsize=8, rotation=90))
    ax.set_ylim((0, 43) if metric == 'energy' else (-5, 85))
    ax.set_yticks([0, 10, 20, 30, 40] if metric == 'energy' else [0, 20, 40, 60, 80])
    ax.set_xlim(-.53, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['2', '8', '16'])
    ax.tick_params(labelsize=8, length=3, pad=2)
    ax.grid(axis='y', color='.88', linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    return artists


def legend(fig, y=.008):
    handles = [Patch(facecolor=color, label=('Random (mean ± SD, n=10)' if basis == 'Random' else basis))
               for basis, color in zip(ORDER, COLORS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, y), ncol=1,
               frameon=False, fontsize=8, handlelength=1.25, labelspacing=.35)


def save(fig, name, records, size):
    fig.canvas.draw()
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_text():
            assert text.get_fontsize() >= 8
    fig.savefig(str(ROOT / (name + '.pdf')), dpi=300)
    fig.savefig(str(ROOT / (name + '.png')), dpi=300)
    with Image.open(str(ROOT / (name + '.png'))) as img:
        assert img.width == 990
        assert all(abs(dpi - 300) < .1 for dpi in img.info['dpi'])
        pixels = list(img.size)
    records.append(dict(name=name, size_inches=size, png_pixels=pixels, min_font_pt=8, dpi=300))
    plt.close(fig)
    print('FIGURE', name, flush=True)


def main():
    hashes = {str(path): sha(path) for path in [SOURCE, RESTORATION]}
    rows = read_csv(SOURCE)
    assert len(rows) == 9
    index = {(r['basis'], int(r['k'])): r for r in rows}
    old = {(r['basis'], int(r['k'])): r for r in read_csv(RESTORATION)}
    for key, r in index.items():
        assert r['model'] == 'HF' and int(r['n_rois']) == 8117
        assert r['recovered_forgetting_pct'] == old[key]['recovered_pct']
        if key[0] == 'Random':
            assert r['random_recovery_sd'] == old[key]['random_sd']
    # Preserve a compact numerical source table, without re-estimating any metric.
    fields = ['model', 'basis', 'k', 'drift_energy_pct', 'random_energy_sd',
              'recovered_forgetting_pct', 'random_recovery_sd', 'n_rois']
    with (ROOT / 'plot_data.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    figures = []
    # A standalone companion matches the previously supplied 3.3-inch plot.
    fig, ax = plt.subplots(figsize=(3.3, 3.5))
    draw(ax, index, 'energy')
    ax.set_ylabel('Drift share (%)', fontsize=8)
    ax.set_xlabel('Subspace dimension k', fontsize=8, labelpad=5)
    legend(fig)
    fig.subplots_adjust(left=.18, right=.985, top=.96, bottom=.29)
    save(fig, 'drift_share_HF_3bars', figures, [3.3, 3.5])
    # Main compact proposal: two panels side by side within one 3.3-inch column.
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.45))
    draw(axes[0], index, 'energy', narrow=True)
    draw(axes[1], index, 'restoration', narrow=True)
    axes[0].set_title('(a) Drift share (%)', fontsize=8, pad=6)
    axes[1].set_title('(b) Recovered\nforgetting (%)', fontsize=8, pad=6)
    fig.text(.53, .245, 'Subspace dimension k', ha='center', fontsize=8)
    legend(fig)
    fig.subplots_adjust(left=.12, right=.99, top=.82, bottom=.35, wspace=.32)
    save(fig, 'drift_pair_HF_horizontal', figures, [3.3, 2.45])
    # Readability-first alternative for a taller single-column allocation.
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.6), sharex=True)
    draw(axes[0], index, 'energy')
    draw(axes[1], index, 'restoration')
    axes[0].set_ylabel('Drift share (%)', fontsize=8)
    axes[1].set_ylabel('Recovered forgetting (%)', fontsize=8)
    for ax, panel in zip(axes, ['(a)', '(b)']):
        ax.text(.015, .98, panel, transform=ax.transAxes, ha='left', va='top', fontsize=8)
    axes[1].set_xlabel('Subspace dimension k', fontsize=8, labelpad=5)
    legend(fig)
    fig.subplots_adjust(left=.18, right=.985, top=.97, bottom=.20, hspace=.13)
    save(fig, 'drift_pair_HF_vertical', figures, [3.3, 4.6])
    for path, checksum in hashes.items():
        assert sha(path) == checksum
    validation = dict(status='PASS', source_sha256=hashes, data_rows=len(rows), figures=figures,
                      data_recomputed=False, GPU_used=False, training=False, selected_bases=ORDER,
                      dimensions=KS, colors=dict(zip(ORDER, COLORS)),
                      source_base_commit=BASE_COMMIT, random_error_bars='SD over ten fixed bases, ddof=0',
                      quantity='Drift share is cross-stage per-RoI feature drift energy; not weather-mean shift energy')
    (ROOT / 'validation.json').write_text(json.dumps(validation, indent=2, ensure_ascii=False) + '\n')
    print('CSV_MATCH_PASS; all sources unchanged; three PNG/PDF pairs', flush=True)


if __name__ == '__main__':
    main()
