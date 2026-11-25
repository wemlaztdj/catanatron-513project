from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def charts(data_dir='processed_data', chart_dir='charts'):
    out=Path(data_dir); dest=Path(chart_dir); dest.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(out/'advantage_labeled_positions.csv'); c=pd.read_csv(out/'readiness_curves.csv'); r=pd.read_csv(out/'advantage_classification_results.csv'); cv=pd.read_csv(out/'advantage_classification_cv.csv'); pred=pd.read_csv(out/'advantage_predictions.csv')
    def save(name):
        plt.tight_layout(); plt.savefig(dest/(name+'.png'),dpi=180,bbox_inches='tight'); plt.savefig(dest/(name+'.svg'),bbox_inches='tight'); plt.close()
    mean=c.groupby('Personal_Turn')[['Build_Probability_Trade','Build_Probability_NoTrade']].mean()*100
    mean.columns=['4:1 trading','No trading']; mean.plot(figsize=(8,4)); plt.title('Theoretical expansion readiness: trading vs no trading'); plt.xlabel('Personal turn');plt.ylabel('Mean probability (%)');plt.ylim(0,100);save('01_trading_impact')
    curves=c.pivot_table(index='Personal_Turn',columns='Player_Number',values='Build_Probability_Trade',aggfunc='mean')*100
    curves.columns=['P'+str(x) for x in curves.columns];curves.plot(figsize=(8,4));plt.title('Expansion readiness by player order');plt.xlabel('Personal turn');plt.ylabel('Mean probability (%)');plt.ylim(0,100);save('02_player_order')
    plt.figure(figsize=(8,4));plt.hist(d.Turns_To_50.dropna(),bins=np.arange(0.5,41.5,1),edgecolor='white');plt.title(f'Turns to 50% readiness ({d.Turns_To_50.notna().sum()} threshold-reaching positions)');plt.xlabel('Personal turns');plt.ylabel('Players');save('03_turn_distribution')
    d.Advantage_Class.value_counts().reindex(['Strong','Average','Weak','Blocked']).plot.bar(figsize=(7,4),rot=0);plt.title('Starting advantage classes');plt.ylabel('Players');save('04_advantage_classes')
    g=d.groupby('Missing_Required_Resources').Prob_Turn_10.mean()*100;g.plot.bar(figsize=(7,4),rot=0);plt.title('Missing required resources vs turn-10 readiness');plt.xlabel('Missing production types (wood, brick, sheep, wheat)');plt.ylabel('Mean probability (%)');plt.ylim(0,100);save('05_missing_resources')
    comp=r.set_index('Model')[['Macro_F1']].rename(columns={'Macro_F1':'Held-out macro F1'}).join(cv.set_index('Model')[['Macro_F1']].rename(columns={'Macro_F1':'Group CV macro F1'}));comp.plot.barh(figsize=(8,4));plt.title('Classification comparison');plt.xlabel('Macro F1');plt.xlim(0,1);save('06_model_comparison')
    p=pred[pred.Model=='Logistic Regression']; ConfusionMatrixDisplay(confusion_matrix(p.Actual,p.Predicted,labels=['Strong','Average','Weak']),display_labels=['Strong','Average','Weak']).plot();plt.title('Logistic regression — held-out games');save('07_confusion_matrix')
if __name__=='__main__':
    import sys
    charts(*(sys.argv[1:3]))
