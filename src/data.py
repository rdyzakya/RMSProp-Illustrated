from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris
from sklearn.preprocessing import MinMaxScaler

class DS(Dataset):
    def __init__(self, df):
        self.df = df
        self.scaler = MinMaxScaler().fit(self.df)
        self.df.iloc[:,:] = self.scaler.transform(self.df)
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, i):
        return self.df.iloc[i:i+1,0].values, self.df.iloc[i:i+1,1].values

def load_data(path):
    df = pd.read_csv(path)
    if df.shape[1] != 2:
        raise ValueError("The dataset should only consist of 1 feature and 1 output")
    dataset = DS(df)
    dataloader = DataLoader(dataset, batch_size=len(dataset)//10, shuffle=False) # batch size = n data // 10
    return dataloader # access .dataset for DS object

def load_data_reg():
    df = load_diabetes(as_frame=True)
    df = pd.concat([df["data"], df["target"]], axis=1)[["bmi","target"]]
    dataset = DS(df)
    dataloader = DataLoader(dataset, batch_size=len(dataset)//10, shuffle=False) # batch size = n data // 10
    return dataloader # access .dataset for DS object

def load_data_cls():
    df = load_iris(as_frame=True)
    df = pd.concat([df["data"], df["target"]], axis=1)[["petal length (cm)","target"]]
    df.loc[df["target"] == 2, "target"] = 1
    dataset = DS(df)
    dataloader = DataLoader(dataset, batch_size=len(dataset)//10, shuffle=False) # batch size = n data // 10
    return dataloader # access .dataset for DS object