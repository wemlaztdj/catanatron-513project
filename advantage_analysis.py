"""Grouped supervised approximation of the theoretical readiness benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.base import clone

FEATURES=['Player_Number','Wood_Production','Brick_Production','Sheep_Production','Wheat_Production','Ore_Production','Number_Diversity','Expansion_Options','Initial_Wood','Initial_Brick','Initial_Sheep','Initial_Wheat','Initial_Ore']
LABELS=['Strong','Average','Weak']

def thresholds(d):
    return tuple(d.Turns_To_50.dropna().quantile([.25,.75]))

def labels(d,q):
    t=d.Turns_To_50.fillna(np.inf)
    return np.select([t<=q[0],t<=q[1]],LABELS[:2],default='Weak')

def compare(d):
    data=d.loc[~d.Blocked].reset_index(drop=True)
    X=data[FEATURES]; groups=data.Game_ID
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(X,groups=groups))
    q=thresholds(data.iloc[tr]); ytr=labels(data.iloc[tr],q); yte=labels(data.iloc[te],q)
    models={'Majority baseline':DummyClassifier(strategy='most_frequent'),'KNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7)),'Naive Bayes':GaussianNB(),'CART':DecisionTreeClassifier(max_depth=5,min_samples_leaf=5,random_state=42),'Random Forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=1),'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))}
    metrics=lambda y,p: [accuracy_score(y,p),balanced_accuracy_score(y,p),f1_score(y,p,labels=LABELS,average='macro',zero_division=0)]
    test=[]; cv=[]; predictions=[]
    for name,m in models.items():
        m.fit(X.iloc[tr],ytr); pred=m.predict(X.iloc[te]); test.append([name,*metrics(yte,pred)])
        predictions.extend((name,g,p,a,b) for g,p,a,b in zip(data.Game_ID.iloc[te],data.Player.iloc[te],yte,pred))
        fold=[]
        for i,j in GroupKFold(n_splits=5).split(X,groups=groups):
            fq=thresholds(data.iloc[i]); yt=labels(data.iloc[i],fq); yv=labels(data.iloc[j],fq)
            fit=clone(m).fit(X.iloc[i],yt); fold.append(metrics(yv,fit.predict(X.iloc[j])))
        cv.append([name,*np.mean(fold,axis=0),*np.std(fold,axis=0)])
    results=pd.DataFrame(test,columns=['Model','Accuracy','Balanced_Accuracy','Macro_F1']); cv=pd.DataFrame(cv,columns=['Model','Accuracy','Balanced_Accuracy','Macro_F1','Accuracy_SD','Balanced_Accuracy_SD','Macro_F1_SD'])
    pred=pd.DataFrame(predictions,columns=['Model','Game_ID','Player','Actual','Predicted'])
    labeled=d.copy(); labeled['Advantage_Class']='Blocked'; labeled.loc[~labeled.Blocked,'Advantage_Class']=labels(labeled.loc[~labeled.Blocked],q)
    train_games=set(data.Game_ID.iloc[tr]); test_games=set(data.Game_ID.iloc[te]); assert not train_games & test_games
    labeled['Split']=np.where(labeled.Game_ID.isin(train_games),'Train',np.where(labeled.Game_ID.isin(test_games),'Test','Not modeled'))
    return results,cv,pred,labeled,q

if __name__=='__main__':
    import sys
    out=Path(sys.argv[1] if len(sys.argv)>1 else 'processed_data'); d=pd.read_csv(out/'starting_advantage.csv'); results,cv,pred,labeled,q=compare(d)
    results.to_csv(out/'advantage_classification_results.csv',index=False);cv.to_csv(out/'advantage_classification_cv.csv',index=False);pred.to_csv(out/'advantage_predictions.csv',index=False);labeled.to_csv(out/'advantage_labeled_positions.csv',index=False)
    print('Thresholds:',q);print(labeled.groupby(['Split','Advantage_Class']).size());print(results.to_string(index=False));print(cv.to_string(index=False))
