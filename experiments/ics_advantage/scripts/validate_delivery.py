"""Validate samples, paired supports, statistical exports and required artifacts."""
import argparse,json
from common import *
from feature_statistics import load,validate,VIEWS

def main(batch):
    models=MODELS if batch=='1' else [m for m in ['HF_stage4','INT_stage4','HF_stage1','INT_stage1'] if m in json.loads((ROOT/'manifests/batch2_model_provenance.json').read_text())]
    reference=load(models[0]);checks=[]
    selection=json.loads((ROOT/'manifests/tsne_selection.json').read_text())['selected'];wanted={r['annotation_id'] for r in selection}
    samples={r['domain']:set(r['dose_image_ids']) for r in json.loads((ROOT/'manifests/probe_samples.json').read_text())}
    official={r['group']:r['sha256'] for r in json.loads((ROOT/'manifests/batch1_model_provenance.json').read_text())['models']} if batch=='1' else {m:v['checkpoint_sha256'] for m,v in json.loads((ROOT/'manifests/batch2_model_provenance.json').read_text()).items()}
    for model in models:
        d=load(model);validate(d,reference,model)
        record=json.loads((ROOT/'manifests'/('infer_'+model+'.json')).read_text());assert record['status']=='complete' and record['returncode']==0
        assert record['checkpoint_sha256']==official[model]
        meta=json.loads((ROOT/'features'/model/'meta.json').read_text());assert meta['args']['seed']==0 and meta['args']['dose_frames_per_video']==5
        clean=probe.normalize(d['feat_clean'].astype(float));assert np.allclose(np.linalg.norm(clean,axis=1),1,atol=1e-7)
        for domain in DOMAINS:
            mask=d['weather']==domain
            expected=np.isin(d['image_id'][mask],list(samples[domain]))
            for view in VIEWS[2:]:assert np.array_equal(np.isfinite(d['feat_'+view][mask]).all(1),expected),(model,domain,view)
        with np.load(str(ROOT/'tsne_features'/model/'night.npz'),allow_pickle=False) as z:
            assert set(z['annotation_id'])==wanted
            mapping={int(a):i for i,a in enumerate(d['annotation_id']) if d['weather'][i]=='night'}
            difference=float(np.max(np.abs(probe.normalize(z['feat_clean'].astype(float))-clean[[mapping[int(a)] for a in z['annotation_id']]])))
            assert difference<1e-5,(model,'supplemental clean discrepancy',difference)
            for key in ['feat_clean','feat_dark_0p40','feat_fog_0p45','feat_fapk_0p50']:assert np.isfinite(z[key]).all()
        checks.append(dict(model=model,rois=len(clean),all_GT_and_metadata_exact=True,first_five_dose_support_exact=True,supplemental_clean_max_abs_difference=difference))
    root=ROOT/'analysis'/('batch'+batch)/'features'
    rows=readcsv(root/'per_view.csv');means=readcsv(root/'model_means.csv');effects=readcsv(root/'paired_effects.csv');sil=readcsv(root/'silhouette.csv')
    assert len(rows)==len(models)*3*15 and len(means)==len(models)*3 and len(sil)==len(models)*3
    for r in rows:
        for k in ['IR','IR_ci_low','IR_ci_high','cross_mAP','cross_mAP_ci_low','cross_mAP_ci_high']:assert np.isfinite(float(r[k]))
        assert float(r['IR'])>=0 and 0<=float(r['cross_mAP'])<=100 and 0<int(r['queries'])<=4000
        assert int(r['IR_bootstrap_valid'])==int(r['mAP_bootstrap_valid'])==10000
        assert float(r['IR_ci_low'])<=float(r['IR_ci_high']) and float(r['cross_mAP_ci_low'])<=float(r['cross_mAP_ci_high'])
    for pool in ['night','dawn_dusk','pooled']:
        for view in ['clean']+VIEWS:
            rr=[r for r in rows if r['pool']==pool and r['view']==view]
            assert len({r['queries'] for r in rr})==len({r['query_videos'] for r in rr})==1
    for r in means:
        group=[v for v in rows if v['model']==r['model'] and v['pool']==r['pool'] and v['view']!='clean']
        assert abs(float(r['mean_IR'])-np.mean([float(v['IR']) for v in group]))<1e-12
        assert abs(float(r['mean_cross_mAP'])-np.mean([float(v['cross_mAP']) for v in group]))<1e-12
    assert all(-1<=float(r['silhouette'])<=1 for r in sil)
    for r in effects:
        assert float(r['ci_low'])<=float(r['ci_high']) and int(r['bootstrap_valid_replicates'])==10000
    if batch=='1':
        files=[ROOT/'analysis/batch1/tracking/brightness_AssA']
        files += [ROOT/'figures/batch1'/(prefix+pool) for prefix in ['dose_response_','invariance_bars_'] for pool in ['night','dawn_dusk','pooled']]
        files += [ROOT/'figures/batch1/intervention_tsne']
    else:
        stages=[s for s in ['stage4','stage1'] if 'INT_'+s in models]
        files=[ROOT/'analysis/batch2/tracking/brightness_AssA']
        if 'stage4' in stages:files.append(ROOT/'analysis/batch2/tracking/brightness_AssA_four_domains')
        files += [ROOT/'figures/batch2'/(prefix+pool+'_'+stage) for stage in stages for prefix in ['dose_response_','invariance_bars_'] for pool in ['night','dawn_dusk','pooled']]
        files += [ROOT/'figures/batch2'/(prefix+stage) for stage in stages for prefix in ['intervention_tsne_','cross_weather_tsne_']]
    from PIL import Image
    for base in files:
        pdf=base.with_suffix('.pdf');png=base.with_suffix('.png')
        assert pdf.read_bytes().startswith(b'%PDF')
        with Image.open(str(png)) as image:image.verify()
    # Every native full-pool metric must still equal its recorded original.
    native=[]
    for model in models:
        for p in (ROOT/'tracking'/model).glob('*/aggregate_parity.csv'):native.extend(readcsv(p))
    assert native and all(r['passed']=='True' and abs(float(r['delta']))<1e-6 for r in native)
    record=dict(status='PASS',batch=batch,models=models,feature_checks=checks,view_rows=len(rows),mean_rows=len(means),
          paired_feature_rows=len(effects),native_aggregate_checks=len(native),figures_pdf_png=len(files),
          all_bootstrap_intervals_valid10000=True,source_data_mutation='none; task-local snapshots and outputs only')
    dump(ROOT/'manifests'/('batch'+batch+'_validation.json'),record)
    if batch=='1':dump(ROOT/'manifests/batch1_complete.json',dict(status='complete',models=models,single_training_seed=True))
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['1','2'],default='1');main(p.parse_args().batch)
