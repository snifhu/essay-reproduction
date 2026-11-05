from pathlib import Path
import sys,json,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from hydequivnet.config import ExperimentConfig
from hydequivnet.data import create_dataloaders
from hydequivnet.train import build_model
from hydequivnet.metrics import nse,kge,pearson_r
base=Path('E:/HydEquivNet_reproduction');out=base/'runs/single_60ep'
meta=json.loads((out/'run.json').read_text());assert meta['status']=='complete'
cfg=ExperimentConfig(**meta['config']);loaders=create_dataloaders('01022500',cfg.data_dir)
model=build_model('hen',6,5,cfg).cuda();model.load_state_dict(torch.load(next((out/'checkpoints').glob('*.pth')),weights_only=True))
torch.set_num_threads(4);model.eval();pred=[];obs=[]
with torch.no_grad():
    for x,s,y in loaders['test']:
        pred.extend(torch.exp(model(x.cuda(),s.cuda())).cpu().numpy().tolist());obs.extend((torch.exp(y)-0.01).numpy().tolist())
pred=np.asarray(pred);obs=np.asarray(obs)
raw={'nse':nse(obs,pred),'kge':kge(obs,pred),'pearson_r':pearson_r(obs,pred)}
dates=pd.date_range('2004-01-01','2008-12-31')[364:-1]
assert len(dates)==len(pred)
pd.DataFrame({'date':dates,'observed_Q_mm_day':obs,'simulated_exp_logQ_mm_day':pred}).to_csv(out/'test_predictions.csv',index=False)
(out/'raw_Q_metrics.json').write_text(json.dumps(raw,indent=2))
gpu=pd.read_csv(out/'gpu.csv');print('run',meta);print('rawQ metrics',raw);print('peak_device_memory_MiB',gpu.memory_used_MiB.max())
