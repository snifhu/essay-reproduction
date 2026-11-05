from pathlib import Path
import sys,json,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hydequivnet.data import create_dataloaders,SplitDates
from hydequivnet.config import ExperimentConfig
from hydequivnet.train import build_model,evaluate_model
base=Path('E:/HydEquivNet_reproduction'); torch.set_num_threads(4)
state=torch.load(base/'work/published.pth',map_location='cpu',weights_only=True)
model=build_model('hen',6,5,ExperimentConfig(data_dir='')).cuda()
print('checkpoint',type(state),list(state)[:5],flush=True)
model.load_state_dict(state)
results={}
for driver in ['daymet','maurer','nldas']:
    dest=base/f'work/eval_{driver}';dest.mkdir(exist_ok=True)
    for f in (base/'data_single_daymet').glob('*.txt'):
        if 'forcing' not in f.name:shutil.copy2(f,dest/f.name)
    f=next((base/f'data/basin_dataset_public_v1p2/basin_mean_forcing/{driver}/01').glob('*forcing*'))
    shutil.copy2(f,dest/f.name)
    loaders=create_dataloaders('01022500',str(dest))
    results[driver]={}
    for end in ['1999-12-31','2003-12-31']:
        loaders=create_dataloaders('01022500',str(dest),split_dates=SplitDates(train_end=end))
        for mode in ['physical','zero_legacy']:
            loaders['test'].dataset.static_data=torch.tensor([1,620.38,92.68,17.79072,0.9232]) if mode=='physical' else torch.zeros(5)
            key=end+'_'+mode
            results[driver][key]=evaluate_model(model,loaders['test'],torch.device('cuda'))
            print(driver,key,results[driver][key],flush=True)
(base/'work/published_checkpoint_eval.json').write_text(json.dumps(results,indent=2))
