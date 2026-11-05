from pathlib import Path
import sys,json,subprocess
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hydequivnet.data import load_basin_data,create_dataloaders,STATIC_INPUT_COLUMNS
root='E:/HydEquivNet_reproduction/data_single_daymet'
df,features,static,area=load_basin_data('01022500',root)
assert np.isfinite(df.values).all()
assert (static!=0).all(),static
assert len(df.index.unique())==len(df)
assert df.loc['1980':'2008'].index.equals(pd.date_range('1980-01-01','2008-12-31'))
loaders=create_dataloaders('01022500',root)
assert torch.cuda.is_available()
x=torch.randn(128,128,device='cuda'); y=x@x;torch.cuda.synchronize()
report={'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'features':features,'static':dict(zip(STATIC_INPUT_COLUMNS,map(float,static))),'forcing_area_km2':area,'date_min':str(df.index.min()),'date_max':str(df.index.max()),'samples':{k:len(v.dataset) for k,v in loaders.items()},'target_dates':{k:[str(v.dataset.x_data.shape),str(df.loc[{'train':'1980','val':'2000','test':'2004','test_scaled':'2004'}[k]:].index[364])] for k,v in loaders.items()}}
# Verify the original loader actually returns zeros and the corrected one fails loudly.
import importlib.util,tempfile
original=Path('E:/HydEquivNet_精读资料/HydEquivNet_v1.0/source_and_results/hydequivnet/data.py')
spec=importlib.util.spec_from_file_location('original_data',original)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
report['original_static_native_filename']=m._load_static_attributes('01022500',root).tolist()
with tempfile.TemporaryDirectory() as tmp:
    import shutil
    shutil.copy2(Path(root)/'basin_physical_characteristics.txt',Path(tmp)/'catchment_attributes.txt')
    report['original_static_renamed_only']=m._load_static_attributes('01022500',tmp).tolist()
    from hydequivnet.data import _load_static_attributes
    try:_load_static_attributes('99999999',tmp)
    except ValueError:report['missing_basin_raises']=True
    else:raise AssertionError('missing basin must raise')
print(json.dumps(report,indent=2))
Path('E:/HydEquivNet_reproduction/work/preflight.json').write_text(json.dumps(report,indent=2))
