"""Instrument the existing sequential trainer; execute only stage4, 2 epochs."""
import copy,csv,json,os,random,sys,time
from pathlib import Path
from apr_complement_experiment import torch,np,HOOKS,Hook,get_dist_info,unwrap,instrument_fap,activity,install_checkpointing,torch_save,sha
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from protocol import write_json,write_views
import train_seq3_episode as seq

@HOOKS.register_module()
class ICSShortStageStateHook(Hook):
    def __init__(self,root):self.root=Path(root)
    def before_run(self,runner):
        rank,world=get_dist_info();assert world==4
        self.rank=rank;m=unwrap(runner.model);h=m.track_head
        common=json.loads((self.root/'manifests/common.json').read_text())
        assert sha(common['history_path'])==common['history_sha256']
        saved=torch.load(common['source_checkpoint'],map_location='cpu')['state_dict']
        actual=m.state_dict()
        # Check the entire loaded student and teacher; no reset of inherited bank.
        assert set(saved)==set(actual)
        assert all(torch.equal(v.detach().cpu(),saved[k]) for k,v in actual.items())
        self.start=int(h.causal_bank.num_updates.item());del saved,actual
        history=torch.load(common['history_path'],map_location='cpu')
        h.roi_proto_memory.load_state_dict_memory(history)
        assert set(h.roi_proto_memory._history)=={'overcast','rainy','dawn_dusk'}
        source=torch.load(common['rank_rng_sources'][str(rank)],map_location='cpu')
        for k,v in source['counters'].items():setattr(h,k,v)
        instrument_fap(m)
        self.pair_calls=0;self.insufficient=0
        self.stream=(self.root/('pair_checks_rank%d.csv'%rank)).open('w')
        self.csv=csv.DictWriter(self.stream,fieldnames=['iteration','pairs','min_samples','insufficient_pairs']);self.csv.writeheader()
        original=h._build_pairwise_residuals
        def pair_checks(*args,**kwargs):
            residual,n=original(*args,**kwargs);self.pair_calls+=1
            bad=residual is None;self.insufficient+=int(bad)
            self.csv.writerow(dict(iteration=runner.iter+1,pairs=n,min_samples=h.causal_min_samples,insufficient_pairs=int(bad)));self.stream.flush()
            return residual,n
        h._build_pairwise_residuals=pair_checks
        write_json(self.root/('start_rank%d.json'%rank),dict(num_updates=self.start,history_sha256=common['history_sha256'],checkpoint_sha256=common['checkpoint_sha256'],all_loaded_tensors_exact=True,bank_reset=False,activity=activity(m),time=time.time()))
        # Same continuation RNG for corresponding ranks in every factorial group.
        random.setstate(source['python_rng']);np.random.set_state(source['numpy_rng']);torch.set_rng_state(source['torch_rng']);torch.cuda.set_rng_state(source['cuda_rng'])
    def before_train_epoch(self,runner):
        loader=runner.data_loader
        old=json.loads(Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420/manifests/stage4_dataset.json').read_text())
        assert len(loader.dataset)==old['actual_training_samples']
        assert loader.batch_size==16 and loader.num_workers==2
        write_json(self.root/('dataset_epoch%d_rank%d.json'%(runner.epoch+1,self.rank)),dict(domain='night',training_samples=len(loader.dataset),batch_capacity_per_gpu=loader.batch_size,workers_per_gpu=loader.num_workers,sampler_type=type(loader.sampler).__name__,sampler_seed=getattr(loader.sampler,'seed',None),same_number_of_samples_as_frozen_HF=True))
    def after_train_iter(self,runner):
        if runner.iter<3 or (runner.iter+1)%20==0:
            write_json(self.root/('progress_rank%d.json'%self.rank),dict(epoch=runner.epoch+1,iteration=runner.iter+1,loss=float(runner.outputs['loss'].detach().item()),pair_checks=self.pair_calls,insufficient_pairs=self.insufficient,activity=activity(runner.model),time=time.time()))
    def after_train_epoch(self,runner):
        write_json(self.root/('epoch%d_rank%d.json'%(runner.epoch+1,self.rank)),dict(activity=activity(runner.model),insufficient_pairs=self.insufficient,pair_checks=self.pair_calls))
    def after_run(self,runner):
        self.stream.close();end=int(unwrap(runner.model).track_head.causal_bank.num_updates.item())
        write_json(self.root/('end_rank%d.json'%self.rank),dict(num_updates_start=self.start,num_updates_end=end,updates=end-self.start,pair_checks=self.pair_calls,insufficient_pairs=self.insufficient,activity=activity(runner.model),time=time.time()))

def main():
    args=seq.parse_args()
    assert args.resume_episode_from_stage==4 and args.epochs_per_stage==2
    assert args.target_order==['overcast','rainy','dawn_dusk','night']
    assert args.samples_per_gpu==16 and args.workers_per_gpu==2 and args.seed==1790355234
    install_checkpointing()
    seq.parse_args=lambda:args
    def write_plan(root,src,source_domain,targets,evals,epochs):
        if not seq.rank0():return
        common=json.loads((Path(root).parent/'manifests/common.json').read_text())
        write_json(Path(root)/'state/stage_plan.json',dict(entrypoint='train_seq3_episode.py --resume-episode-from-stage 4',target_order=targets,executed_stages=[dict(stage=4,domain='night',epochs=2)],skipped_training_stages=[1,2,3],initial_training_checkpoint=common['source_checkpoint'],source_S_initialization_overwritten_by_full_HF_stage3_before_training=True,bank_reset=False,seed=args.seed))
    seq.write_plan=write_plan
    def stage_cfg(base,target,wdir,args):
        assert target=='night';cfg=copy.deepcopy(base);cfg.work_dir=wdir;cfg.total_epochs=2
        if cfg.get('runner'):cfg.runner.max_epochs=2
        cfg.load_from=None;cfg.resume_from=None
        write_views(cfg,Path(wdir)/'resolved_views.txt',dict(entrypoint='train_seq3_episode.py',resume_episode_from_stage=4))
        return cfg
    seq.stage_cfg=stage_cfg
    def save_snap(model,path,meta):
        if not seq.rank0():return
        path=Path(path)
        if path.name=='stage00_source_only.pth':return
        assert path.name=='stage04_after_night.pth'
        endpoint=Path(args.work_dir)/'adapt/stage04_after_night/epoch_2.pth';assert endpoint.exists()
        if not path.exists():os.link(str(endpoint),str(path))
    seq.save_snap=save_snap
    seq.main()
    rank,_=get_dist_info();torch.distributed.barrier()
    if rank==0:
        out=Path(args.work_dir).parent
        assert all((out/('end_rank%d.json'%r)).exists() for r in range(4))
        ck=Path(args.work_dir)/'adapt/stage04_after_night/epoch_2.pth'
        write_json(out/'train_complete.json',dict(status='complete',stage=4,epochs=2,checkpoint=str(ck),checkpoint_sha256=sha(ck),seed=args.seed))
    torch.distributed.barrier();torch.distributed.destroy_process_group()
if __name__=='__main__':main()
