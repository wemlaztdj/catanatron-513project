from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from experiment_io import read_required, same_players, validate_advantage

def charts(data_dir='processed_data_5000', chart_dir='charts/5000'):
    out=Path(data_dir); dest=Path(chart_dir); dest.mkdir(parents=True,exist_ok=True)
    baseline=validate_advantage(read_required(out,"starting_advantage.csv","05_starting_advantage.ipynb"))
    c=read_required(out,"readiness_curves.csv","05_starting_advantage.ipynb",usecols=["Game_ID","Player","Personal_Turn","Player_Number","Build_Probability_Trade","Build_Probability_NoTrade"])
    if len(c)!=40*len(baseline) or not same_players(c[["Game_ID","Player"]].drop_duplicates(),baseline) or c.duplicated(["Game_ID","Player","Personal_Turn"]).any():
        raise ValueError("Readiness curves do not match starting advantage. Rerun 05a.")
    d=baseline.copy()
    d["Advantage_Class"]=np.select([d.Blocked,d.Turns_To_50.le(7),d.Turns_To_50.le(14)],["Blocked","Strong","Average"],default="Weak")
    class_title="Starting advantage classes (fixed 7/14-turn cutoffs)"
    names=["advantage_labeled_positions.csv","advantage_classification_results.csv","advantage_classification_cv.csv","advantage_predictions.csv"]
    ready=all((out/name).is_file() and (out/name).stat().st_mtime >= (out/"starting_advantage.csv").stat().st_mtime for name in names)
    if ready:
        labeled=pd.read_csv(out/names[0])
        ready=same_players(labeled,baseline)
        if ready:
            d=labeled
            r=pd.read_csv(out/names[1])
            cv=pd.read_csv(out/names[2])
            pred=pd.read_csv(out/names[3])
            ready=not pred.empty and set(pred.Game_ID)<=set(d.Game_ID) and "Logistic Regression" in set(pred.Model)
            if ready:
                class_title="Starting advantage classes (training quartiles)"
    if not ready:
        d=baseline.copy()
        d["Advantage_Class"]=np.select([d.Blocked,d.Turns_To_50.le(7),d.Turns_To_50.le(14)],["Blocked","Strong","Average"],default="Weak")
        print("05b results missing or outdated; generating descriptive charts only. Run 05_classification for classifier charts.")
    created=[]
    def save(name):
        plt.tight_layout()
        path=dest/(name+".png")
        plt.savefig(path,dpi=180,bbox_inches="tight")
        plt.savefig(dest/(name+".svg"),bbox_inches="tight")
        plt.close()
        created.append(path)
    mean=c.groupby('Personal_Turn')[['Build_Probability_Trade','Build_Probability_NoTrade']].mean()*100
    mean.columns=['4:1 trading','No trading']; mean.plot(figsize=(8,4)); plt.title('Theoretical expansion readiness: trading vs no trading'); plt.xlabel('Personal turn');plt.ylabel('Mean probability (%)');plt.ylim(0,100);save('01_trading_impact')
    curves=c.pivot_table(index='Personal_Turn',columns='Player_Number',values='Build_Probability_Trade',aggfunc='mean')*100
    curves.columns=['P'+str(x) for x in curves.columns];curves.plot(figsize=(8,4));plt.title('Expansion readiness by player order');plt.xlabel('Personal turn');plt.ylabel('Mean probability (%)');plt.ylim(0,100);save('02_player_order')
    plt.figure(figsize=(8,4));plt.hist(d.Turns_To_50.dropna(),bins=np.arange(0.5,41.5,1),edgecolor='white');plt.title(f'Turns to 50% readiness ({d.Turns_To_50.notna().sum()} threshold-reaching positions)');plt.xlabel('Personal turns');plt.ylabel('Players');save('03_turn_distribution')
    d.Advantage_Class.value_counts().reindex(['Strong','Average','Weak','Blocked'],fill_value=0).plot.bar(figsize=(7,4),rot=0);plt.title(class_title);plt.ylabel('Players');save('04_advantage_classes')
    g=d.groupby('Missing_Required_Resources').Prob_Turn_10.mean()*100;g.plot.bar(figsize=(7,4),rot=0);plt.title('Missing required resources vs turn-10 readiness');plt.xlabel('Missing production types (wood, brick, sheep, wheat)');plt.ylabel('Mean probability (%)');plt.ylim(0,100);save('05_missing_resources')
    if ready:
        comp=r.set_index('Model')[['Macro_F1']].rename(columns={'Macro_F1':'Held-out macro F1'}).join(cv.set_index('Model')[['Macro_F1']].rename(columns={'Macro_F1':'Group CV macro F1'}));comp.plot.barh(figsize=(8,4));plt.title('Classification comparison');plt.xlabel('Macro F1');plt.xlim(0,1);save('06_model_comparison')
        p=pred[pred.Model=='Logistic Regression']; ConfusionMatrixDisplay(confusion_matrix(p.Actual,p.Predicted,labels=['Strong','Average','Weak']),display_labels=['Strong','Average','Weak']).plot();plt.title('Logistic regression — held-out games');save('07_confusion_matrix')
    return created
if __name__=='__main__':
    import sys
    charts(*(sys.argv[1:3]))
