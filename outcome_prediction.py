"""Five deterministic game-grouped folds; no outcome fields as predictors."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

BASE=["Player_Number","Wood_Production","Brick_Production","Sheep_Production","Wheat_Production","Ore_Production","Number_Diversity","Expansion_Options","Initial_Wood","Initial_Brick","Initial_Sheep","Initial_Wheat","Initial_Ore"]
FEATURE_SETS={"Placement order":["Player_Number"],"Production only":["Total_Production"],"Initial features":BASE,"Initial + readiness":BASE+["Prob_Turn_10"]}

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-40,40)))

def compare(d):
    d=d.loc[d.Completed_Game.eq(1)].copy()
    games=sorted(d.Game_ID.unique())
    folds={g:i%5 for i,g in enumerate(games)}
    d["Fold"]=d.Game_ID.map(folds)
    rows=[["Uniform baseline",.75,.5,.5,.25]]
    predictions=[]
    for name,cols in FEATURE_SETS.items():
        for fold in range(5):
            tr=d[d.Fold!=fold]
            te=d[d.Fold==fold]
            assert not set(tr.Game_ID)&set(te.Game_ID)
            x=tr[cols].to_numpy(float)
            z=te[cols].to_numpy(float)
            mu=x.mean(axis=0)
            sd=x.std(axis=0)
            sd[sd==0]=1
            x=np.column_stack([np.ones(len(x)),(x-mu)/sd])
            z=np.column_stack([np.ones(len(z)),(z-mu)/sd])
            y=tr.Won.to_numpy(float)
            w=np.zeros(x.shape[1])
            for _ in range(1000):
                penalty=.02*w.copy()
                penalty[0]=0
                w-=.15*(x.T@(sigmoid(x@w)-y)/len(y)+penalty)
            scores=sigmoid(z@w)
            predictions.extend(zip([name]*len(te),te.Game_ID,te.Player,te.Fold,te.Won,scores))
        p=pd.DataFrame([r for r in predictions if r[0]==name],columns=["Model","Game_ID","Player","Fold","Won","Win_Score"])
        top=[]
        for _,g in p.groupby("Game_ID"):
            tied=g[np.abs(g.Win_Score-g.Win_Score.max())<1e-12]
            top.append(tied.Won.mean())
        pred=p.Win_Score.ge(.5)
        rows.append([name,accuracy_score(p.Won,pred),balanced_accuracy_score(p.Won,pred),roc_auc_score(p.Won,p.Win_Score),np.mean(top)])
    return pd.DataFrame(rows,columns=["Model","Accuracy","Balanced_Accuracy","ROC_AUC","Within_Game_Top1"]),pd.DataFrame(predictions,columns=["Model","Game_ID","Player","Fold","Won","Win_Score"])
