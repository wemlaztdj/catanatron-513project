"""Exact fixed-production readiness for a road and third settlement; no gameplay."""
from collections import Counter
from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
RESOURCES = ('WOOD','BRICK','SHEEP','WHEAT','ORE')
COST = (2,2,1,1,0)
MAX_TURNS = 40

def affordable(hand, trade=True):
    missing = sum(max(0,c-a) for a,c in zip(hand,COST))
    return missing == 0 or (trade and sum(max(0,a-c)//4 for a,c in zip(hand,COST)) >= missing)

def readiness_curve(start, gains, horizon=160, trade=True):
    """Exact probabilities via an absorbing Markov chain of capped inventories."""
    start = tuple(int(x) for x in start)
    if affordable(start,trade): return np.ones(horizon+1)
    caps = tuple(c+4*(sum(COST)-c) if trade else c for c in COST)
    start = tuple(min(a,c) for a,c in zip(start,caps))
    weights = Counter()
    for n in range(2,13):
        weights[tuple(int(x) for x in gains[n])] += (6-abs(7-n))/36
    states=[start]; index={start:0}; src=[]; dst=[]; prob=[]
    for i,s in enumerate(states):
        for gain,w in weights.items():
            v=tuple(min(c,a+b) for a,b,c in zip(s,gain,caps))
            if affordable(v,trade): continue
            if v not in index:
                index[v]=len(states); states.append(v)
            src.append(index[v]); dst.append(i); prob.append(w)
    matrix=csr_matrix((prob,(src,dst)),shape=(len(states),len(states)))
    dist=np.zeros(len(states)); dist[0]=1
    curve=np.zeros(horizon+1)
    for n in range(1,horizon+1):
        dist=matrix@dist
        curve[n]=1-dist.sum()
    assert np.all(np.diff(curve)>=-1e-9)
    return np.maximum.accumulate(np.clip(curve,0,1))

def build_advantage(p,t,f,max_turns=MAX_TURNS):
    fe=import_module('03_feature_engineering')
    nodes,_,_=fe.build_topology(t)
    maps={gid:g.set_index('Tile_ID') for gid,g in t.groupby('Game_ID')}
    rows=[]; curves=[]
    for r in p.itertuples(index=False):
        board=maps[r.Game_ID]; gains=np.zeros((13,5),dtype=int); start=np.zeros(5,dtype=int)
        for j,node in enumerate((r.Settlement_1,r.Settlement_2)):
            for tid in nodes[node]:
                tile=board.loc[tid]
                if pd.isna(tile.Resource): continue
                k=RESOURCES.index(tile.Resource); gains[int(tile.Number),k]+=1
                if j==1: start[k]+=1
        player=int(r.Placement_Order)
        a=readiness_curve(start,gains,4*max_turns,True)
        b=readiness_curve(start,gains,4*max_turns,False)
        assert np.all(a>=b-1e-9)
        own=np.arange(player,4*max_turns+1,4)
        hits=np.flatnonzero(a[own]>=.5-1e-12)
        rollhits=np.flatnonzero(a>=.5-1e-12)
        row={'Game_ID':r.Game_ID,'Player':r.Player,'Player_Number':player,'Round_1_Order':player,'Round_2_Order':9-player,'Missing_Required_Resources':int(sum(not gains[:,k].any() for k in range(4))), 'Resource_Turns_To_50':int(hits[0]+1) if len(hits) else np.nan,'Resource_Rolls_To_50':int(rollhits[0]) if len(rollhits) else np.nan}
        for turn in (1,2,3,5,10,20,40):
            if turn<=max_turns: row[f'Resource_Prob_Turn_{turn}']=a[own[turn-1]]
        for turn,n in enumerate(own,1): curves.append((r.Game_ID,r.Player,player,turn,n,a[n],b[n]))
        rows.append(row)
    d=f.merge(pd.DataFrame(rows),on=['Game_ID','Player'],validate='one_to_one')
    assert d.Player_Number.eq(d.Placement_Order).all()
    c=pd.DataFrame(curves,columns=['Game_ID','Player','Player_Number','Personal_Turn','Total_Rolls','Resource_Probability_Trade','Resource_Probability_NoTrade'])
    d['Blocked']=d.Expansion_Options.eq(0)
    d['Turns_To_50']=d.Resource_Turns_To_50.where(~d.Blocked)
    d['Rolls_To_50']=d.Resource_Rolls_To_50.where(~d.Blocked)
    d['Build_Roll_To_50']=(4*(d.Turns_To_50-1)+d.Player_Number)
    d['Status']=np.select([d.Blocked,d.Turns_To_50.isna()],['Blocked','Beyond horizon'],default='Reached')
    for turn in (1,2,3,5,10,20,40):
        if turn<=max_turns: d[f'Prob_Turn_{turn}']=d[f'Resource_Prob_Turn_{turn}'].where(~d.Blocked,0)
    c=c.merge(d[['Game_ID','Player','Blocked']],on=['Game_ID','Player'],validate='many_to_one')
    for name in ['Trade','NoTrade']: c[f'Build_Probability_{name}']=c[f'Resource_Probability_{name}'].where(~c.Blocked,0)
    return d,c

if __name__=='__main__':
    import sys
    root=Path(sys.argv[1] if len(sys.argv)>1 else 'processed_data')
    d,c=build_advantage(*(pd.read_csv(root/n) for n in ('initial_placements.csv','map_tiles.csv','initial_features.csv')))
    d.to_csv(root/'starting_advantage.csv',index=False); c.to_csv(root/'readiness_curves.csv',index=False)
    print(d.Status.value_counts()); print(d.Turns_To_50.describe()); print(d.groupby('Player_Number')[['Turns_To_50','Prob_Turn_5','Prob_Turn_10','Blocked']].mean()); print(c.groupby('Personal_Turn')[['Build_Probability_Trade','Build_Probability_NoTrade']].mean().loc[[1,2,3,5,10,20,40]])
