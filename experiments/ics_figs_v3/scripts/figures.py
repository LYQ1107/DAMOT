"""HF-only restoration bars and the twenty preselected image crops."""
from common import *
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from evidence_cpu import resolve_image

plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
ORDER = ['Intervention-sensitive', 'Label-free nuisance', 'Random', 'Identity oracle']
COLORS = {'Intervention-sensitive': '#0072B2', 'Label-free nuisance': '#009E73',
          'Random': '#777777', 'Identity oracle': '#D55E00'}


def restoration_figures():
    rows = read_csv(ROOT / 'drift_summary.csv')
    assert len(rows) == 12
    for bars in [4, 3]:
        bases = ORDER if bars == 4 else [b for b in ORDER if b != 'Label-free nuisance']
        fig, ax = plt.subplots(figsize=(3.3, 3.5))
        ax.axvspan(-.48, .48, color='#eeeeee', zorder=0)
        ax.axhline(0, color='black', lw=.7, zorder=2)
        width = .80 / bars
        ymax = 0.
        ymin = 0.
        for j, basis in enumerate(bases):
            selected = [next(r for r in rows if r['basis'] == basis and int(r['k']) == k) for k in KS]
            values = [float(r['recovered_pct']) for r in selected]
            errors = [float(r['random_sd']) for r in selected] if basis == 'Random' else np.zeros(3)
            x = np.arange(3) + (j - (bars-1)/2) * width
            ax.bar(x, values, width=width * .94, color=COLORS[basis], edgecolor='white', lw=.25,
                   yerr=errors if basis == 'Random' else None,
                   error_kw=dict(capsize=2, elinewidth=.65, capthick=.65), zorder=3)
            for xx, yy, error in zip(x, values, errors):
                ymax = max(ymax, yy + error)
                ymin = min(ymin, yy - error)
                ax.text(xx, yy + error + 1, '%.1f' % yy, ha='center', va='bottom', fontsize=8, rotation=90)
        ax.set_ylim(min(-5, ymin-2), max(75, ymax+16))
        ax.set_xlim(-.53, 2.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['2', '8', '16'])
        ax.set_xlabel('Subspace dimension k', fontsize=8, labelpad=5)
        ax.set_ylabel('Recovered forgetting (%)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', color='.88', lw=.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        handles = [Patch(facecolor=COLORS[b], label=('Random (mean ± SD, n=10)' if b == 'Random' else b)) for b in bases]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .008), ncol=1,
                   frameon=False, fontsize=8, handlelength=1.25, labelspacing=.35)
        fig.subplots_adjust(left=.18, right=.985, top=.96, bottom=.34 if bars == 4 else .29)
        stem = ROOT / ('drift_restoration_HF_%dbars' % bars)
        fig.savefig(str(stem) + '.pdf')
        fig.savefig(str(stem) + '.png', dpi=300)
        plt.close(fig)
        print('DRIFT_FIGURE', stem.name, flush=True)
    write_json(ROOT / 'drift_figure_protocol.json', dict(model='HF', widths_inches=3.3,
               png_dpi=300, min_font_pt=8, no_k2_annotation=True, no_suptitle=True,
               categories=ORDER, k=KS, random_error_bars='Population SD (ddof=0), ten ratio-of-sums replicates',
               intervention_and_oracle_reference='HF_stage1_e5 fit car RoIs', colors=COLORS))


def crop_figure():
    source = EVIDENCE / 'part1/HF/plot_C_crops.csv'
    records = [fingerprint(source)]
    rows = read_csv(source)
    assert len(rows) == 20 and len(set(r['index'] for r in rows)) == 20
    assert [(r['side'], int(r['column'])) for r in rows] == [(s, c) for s in ['low', 'high'] for c in range(10)]
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.8))
    rendered = []
    for row in rows:
        path = resolve_image(row['path'])
        records.append(fingerprint(path))
        with Image.open(str(path)) as image:
            rgb = image.convert('RGB')
        bbox = tuple(map(float, json.loads(row['bbox'])))
        assert len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]
        crop = rgb.crop(bbox)
        r, c = ['low', 'high'].index(row['side']), int(row['column'])
        ax = axes[r, c]
        ax.imshow(crop)
        coordinate = ('%.3f' % float(row['e1'])).replace('-', '−')
        label = '%s, %s' % (DISPLAY[row['weather']], coordinate)
        ax.set_title(label, fontsize=8, pad=3)
        ax.axis('off')
        rendered.append(dict(row, label=label, resolved_path=str(path), crop_width=crop.width, crop_height=crop.height,
                             rendered_row=r, rendered_column=c))
    # Preserve the original 2 x 10 layout and crop aspect ratios; remove the global title.
    fig.tight_layout(pad=.6, w_pad=.35, h_pad=.5)
    stem = ROOT / 'intervention_extremes_HF'
    fig.savefig(str(stem) + '.pdf', dpi=300)
    fig.savefig(str(stem) + '.png', dpi=300)
    plt.close(fig)
    verify(records)
    write_csv(ROOT / 'crop_manifest.csv', rendered)
    write_json(ROOT / 'crop_protocol.json', dict(source_files=records, rows=2, columns=10,
               selected_rois=20, reselection=False, suptitle=None, label='Weather, signed coordinate to 3 decimals',
               same_original_crop_boxes=True, same_original_row_and_column_order=True,
               physical_size_inches=[14, 3.8], min_font_pt=8, png_dpi=300))
    print('CROP_FIGURE_COMPLETE', len(rendered), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', choices=['all', 'drift', 'crops'], default='all')
    args = parser.parse_args()
    if args.which in ['all', 'drift']:
        restoration_figures()
    if args.which in ['all', 'crops']:
        crop_figure()


if __name__ == '__main__':
    main()
