from pathlib import Path
import sys,os,json,time,threading,subprocess,traceback
os.environ['MPLBACKEND']='Agg'
os.environ['TQDM_DISABLE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hydequivnet.config import ExperimentConfig
from hydequivnet.utils import set_seed
from hydequivnet.workflows import run_single_basin,save_results
base=Path('E:/HydEquivNet_reproduction')
import argparse
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--run-name',default='single_60ep');args=p.parse_args()
out=base/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=True)
cfg=ExperimentConfig(data_dir=str(base/'data_single_daymet'),output_dir=str(out),epochs=args.epochs,batch_size=args.batch_size,skip_existing=False)
assert torch.cuda.is_available(),'CUDA required; will not silently fall back to CPU'
torch.set_num_threads(4)
set_seed(42)
stop=threading.Event()
def monitor():
    with (out/'gpu.csv').open('w') as f:
        f.write('elapsed_s,memory_used_MiB,utilization_pct,power_W\n')
        while not stop.is_set():
            row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
            f.write(f'{time.perf_counter()-start:.2f},{row}\n');f.flush();stop.wait(5)
start=time.perf_counter();thread=threading.Thread(target=monitor,daemon=True);thread.start()
meta={'command':subprocess.list2cmdline([sys.executable,*sys.argv]),'commit':subprocess.check_output(['git','-C',str(base/'repo'),'rev-parse','HEAD'],text=True).strip(),'config':vars(cfg),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'status':'running'}
(out/'run.json').write_text(json.dumps(meta,indent=2))
try:
    result=run_single_basin('01022500',cfg,torch.device('cuda'))
    torch.cuda.synchronize();save_results([result],out/'summary.csv')
    meta.update(status='complete',result=result)
except BaseException as exc:
    meta.update(status='failed',error=repr(exc));traceback.print_exc();raise
finally:
    stop.set();thread.join();meta.update(elapsed_seconds=time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
    (out/'run.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
