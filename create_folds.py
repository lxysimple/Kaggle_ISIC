import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold 



def create(path):
    df = pd.read_csv(path) 
    sgkf = StratifiedGroupKFold(n_splits=5)
    for fold, ( _, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
        df.loc[val_ , "kfold"] = int(fold)
    
    df.to_csv(path, index=False)

create('/home/xyli/kaggle/data2018/train-metadata.csv')
create('/home/xyli/kaggle/data2019/train-metadata.csv')
create('/home/xyli/kaggle/data2020/train-metadata.csv')
create('/home/xyli/kaggle/data_others/train-metadata.csv')