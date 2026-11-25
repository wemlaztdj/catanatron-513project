"""Extract outcome labels separately from setup predictors."""
from pathlib import Path
import json
import pandas as pd

def extract(folder=Path("data")):
    rows=[]
    for path in sorted(folder.glob("*.json")):
        g=json.loads(path.read_text(encoding="utf-8"))
        colors=g["colors"]
        assert len(set(colors))==4
        setup=[r[0] for r in g["action_records"][:16]]
        for j,c in enumerate(colors+colors[::-1]):
            assert setup[2*j][:2]==[c,"BUILD_SETTLEMENT"]
            assert setup[2*j+1][:2]==[c,"BUILD_ROAD"]
        rolls=0
        personal={c:0 for c in colors}
        third={}
        for action,_ in g["action_records"][16:]:
            c,kind,value=action
            if kind=="ROLL":
                rolls+=1
                personal[c]+=1
            if kind=="BUILD_SETTLEMENT" and c not in third:
                third[c]=(personal[c],rolls)
        vp=[g["player_state"][f"P{i}_ACTUAL_VICTORY_POINTS"] for i in range(4)]
        winners=[i for i,v in enumerate(vp) if v>=g["game"]["vps_to_win"]]
        complete=len(winners)==1
        for i,c in enumerate(colors):
            t,n=third.get(c,(None,None))
            rows.append([g["game"]["id"],c,i+1,vp[i],int(complete),
                         int(i==winners[0]) if complete else None,t,n,personal[c],rolls])
    df=pd.DataFrame(rows,columns=["Game_ID","Player","Player_Number","Final_VP","Completed_Game","Won","Actual_Turns_To_Third","Actual_Rolls_To_Third","Observed_Personal_Rolls","Game_Total_Rolls"])
    assert not df.duplicated(["Game_ID","Player"]).any()
    return df

if __name__=="__main__":
    extract().to_csv("processed_data/actual_outcomes.csv",index=False)
