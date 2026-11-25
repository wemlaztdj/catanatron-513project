"""Variable-road extension; reuse the one-road benchmark for unchanged positions."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

RESOURCES=('WOOD','BRICK','SHEEP','WHEAT','ORE')
TURNS=(1,2,3,5,10,20,40)
MAX_TURNS=40
MAX_NEW_ROADS=13  # 15 pieces minus two placed in setup

def minimum_roads(own_settlements, own_roads, occupied, all_roads, neighbors):
    """Multi-source BFS. Opponent roads and nodes cannot be traversed."""
    own=set(own_settlements)
    occupied=set(occupied)
    opponents=occupied-own
    own_roads={tuple(sorted(r)) for r in own_roads}
    others=set(all_roads)-own_roads
    starts=(own | {n for r in own_roads for n in r})-opponents
    queue=deque((n,0) for n in starts)
    seen=set(starts)
    while queue:
        n,depth=queue.popleft()
        if n not in occupied and not (neighbors[n]&occupied):
            return depth
        if depth>=MAX_NEW_ROADS:
            continue
        for v in neighbors[n]:
            if v not in seen and v not in opponents and tuple(sorted((n,v))) not in others:
                seen.add(v)
                queue.append((v,depth+1))
    return None

def path_table(p,t):
    fe=import_module("03_feature_engineering")
    nodes,edges,neighbors=fe.build_topology(t)
    rows=[]
    for gid,g in p.groupby("Game_ID",sort=False):
        assert len(g)==4
        occupied=set(g.Settlement_1)|set(g.Settlement_2)
        roads=[(tuple(sorted(ast.literal_eval(r.Road_1))),tuple(sorted(ast.literal_eval(r.Road_2)))) for r in g.itertuples(index=False)]
        all_roads={e for pair in roads for e in pair}
        assert len(occupied)==8 and len(all_roads)==8 and all_roads<=edges
        for r,pair in zip(g.itertuples(index=False),roads):
            k=minimum_roads((r.Settlement_1,r.Settlement_2),pair,occupied,all_roads,neighbors)
            assert k is None or k>=1
            rows.append((gid,r.Player,k))
    return pd.DataFrame(rows,columns=["Game_ID","Player","Minimum_Roads_Required"]),nodes

def affordable(hand, trade=True, cost=(2,2,1,1,0)):
    missing = sum(max(0,c-a) for a,c in zip(hand,cost))
    return missing == 0 or (trade and sum(max(0,a-c)//4 for a,c in zip(hand,cost)) >= missing)

def readiness_curve(start, gains, horizon=160, trade=True, cost=(2,2,1,1,0)):
    """Exact probabilities via an absorbing Markov chain of capped inventories."""
    start = tuple(int(x) for x in start)
    if affordable(start,trade,cost): return np.ones(horizon+1)
    caps = tuple(c+4*(sum(cost)-c) if trade else c for c in cost)
    start = tuple(min(a,c) for a,c in zip(start,caps))
    weights = Counter()
    for n in range(2,13):
        weights[tuple(int(x) for x in gains[n])] += (6-abs(7-n))/36
    states=[start]; index={start:0}; src=[]; dst=[]; prob=[]
    for i,s in enumerate(states):
        for gain,w in weights.items():
            v=tuple(min(c,a+b) for a,b,c in zip(s,gain,caps))
            if affordable(v,trade,cost): continue
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


def _calculate(task):
    key,start,gains,k,player=task
    cost=(k+1,k+1,1,1,0)
    a=readiness_curve(start,gains,4*MAX_TURNS,True,cost)
    own=np.arange(player,4*MAX_TURNS+1,4)
    hits=np.flatnonzero(a[own]>=.5-1e-12)
    rollhits=np.flatnonzero(a>=.5-1e-12)
    row={"Game_ID":key[0],"Player":key[1],
         "Resource_Turns_To_50":float(hits[0]+1) if len(hits) else np.nan,
         "Resource_Rolls_To_50":float(rollhits[0]) if len(rollhits) else np.nan}
    for turn in TURNS:
        row[f"Resource_Prob_Turn_{turn}"]=a[own[turn-1]]
    return row

def build_variable(p,t,baseline,workers=1):
    """Only recalculate positions needing >1 road; retain all old scores."""
    keys=["Game_ID","Player"]
    assert not baseline.duplicated(keys).any()
    assert not p.duplicated(keys).any()
    assert set(p.Game_ID)==set(t.Game_ID)
    paths,nodes=path_table(p,t)
    d=baseline.merge(paths,on=keys,validate="one_to_one")
    assert len(d)==len(p)==len(baseline)
    assert d.Minimum_Roads_Required.eq(1).equals(d.Expansion_Options.gt(0)), "One-road geometry disagrees with baseline"
    for col in ["Blocked","Turns_To_50","Rolls_To_50","Build_Roll_To_50","Status"]+[f"Prob_Turn_{i}" for i in TURNS]:
        d["One_Road_"+col]=d[col]
    k=d.Minimum_Roads_Required
    d["Required_Wood"]=k+1
    d["Required_Brick"]=k+1
    d["Required_Sheep"]=np.where(k.notna(),1,np.nan)
    d["Required_Wheat"]=np.where(k.notna(),1,np.nan)
    d["Required_Ore"]=np.where(k.notna(),0,np.nan)
    d["Total_Construction_Cost"]=2*k+4
    changed=d.loc[k.gt(1),keys+["Minimum_Roads_Required"]]
    print(f"Reusing {int(k.eq(1).sum()):,} one-road scores; recalculating {len(changed):,} positions; no path {int(k.isna().sum()):,}",flush=True)
    targets=p.merge(changed,on=keys,validate="one_to_one")
    maps={gid:g.set_index("Tile_ID") for gid,g in t[t.Game_ID.isin(targets.Game_ID)].groupby("Game_ID")}
    def tasks():
        for r in targets.itertuples(index=False):
            board=maps[r.Game_ID]
            gains=np.zeros((13,5),dtype=int)
            start=np.zeros(5,dtype=int)
            for j,node in enumerate((r.Settlement_1,r.Settlement_2)):
                for tid in nodes[node]:
                    tile=board.loc[tid]
                    if pd.isna(tile.Resource):
                        continue
                    i=RESOURCES.index(tile.Resource)
                    gains[int(tile.Number),i]+=1
                    if j==1:
                        start[i]+=1
            yield ((r.Game_ID,r.Player),start,gains,int(r.Minimum_Roads_Required),int(r.Placement_Order))
    if len(targets):
        if workers>1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                calculated=list(pool.map(_calculate,tasks(),chunksize=8))
        else:
            calculated=[]
            for i,task in enumerate(tasks(),1):
                calculated.append(_calculate(task))
                if i%500==0:
                    print(f"Recalculated {i:,}/{len(targets):,}",flush=True)
        updates=pd.DataFrame(calculated).set_index(keys)
        d=d.set_index(keys)
        d.loc[updates.index,updates.columns]=updates
        d=d.reset_index()
    d["Blocked"]=d.Minimum_Roads_Required.isna()
    resource_cols=["Resource_Turns_To_50","Resource_Rolls_To_50"]+[f"Resource_Prob_Turn_{i}" for i in TURNS]
    d.loc[d.Blocked,resource_cols]=np.nan
    d["Turns_To_50"]=d.Resource_Turns_To_50.where(~d.Blocked)
    d["Rolls_To_50"]=d.Resource_Rolls_To_50.where(~d.Blocked)
    d["Build_Roll_To_50"]=4*(d.Turns_To_50-1)+d.Player_Number
    d["Status"]=np.select([d.Blocked,d.Turns_To_50.isna()],["No legal path","Beyond horizon"],default="Reached")
    for turn in TURNS:
        d[f"Prob_Turn_{turn}"]=d[f"Resource_Prob_Turn_{turn}"].where(~d.Blocked,0)
        assert (d[f"Prob_Turn_{turn}"]>=d[f"One_Road_Prob_Turn_{turn}"]-1e-10).all()
    d["Advantage_Class"]=np.select([d.Blocked,d.Turns_To_50.le(7),d.Turns_To_50.le(14)],["No legal path","Strong","Average"],default="Weak")
    return d

def run(data_dir="processed_data",output_dir="processed_data_variable",workers=1):
    root=Path(data_dir)
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    p,t,b=[pd.read_csv(root/n) for n in ("initial_placements.csv","map_tiles.csv","starting_advantage.csv")]
    d=build_variable(p,t,b,workers)
    d.to_csv(out/"starting_advantage.csv",index=False)
    print(d.Minimum_Roads_Required.value_counts(dropna=False).sort_index())
    print(d.Status.value_counts())
    print(d.Advantage_Class.value_counts())
    return d

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="processed_data")
    parser.add_argument("--output-dir",default="processed_data_variable")
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    run(args.data_dir,args.output_dir,args.workers)
