from pathlib import Path
from collections import Counter, defaultdict
import ast
import pandas as pd

# Change this folder to the location of your two CSV files.
DATA = Path("processed_data")
RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

def build_topology(tiles):
    # Base-map node numbering follows Catanatron's tile order and
    # corner order: N, NE, SE, S, SW, NW.
    coords = tiles[["Tile_ID", "Coordinate_X", "Coordinate_Y",
                    "Coordinate_Z"]].drop_duplicates().sort_values("Tile_ID")
    assert len(coords) == 19
    assert coords["Tile_ID"].tolist() == list(range(19))
    vertices = {}
    node_tiles = defaultdict(list)
    edges = set()
    corners = [(0,-2), (1,-1), (1,1), (0,2), (-1,1), (-1,-1)]
    for tile in coords.itertuples(index=False):
        x = 2 * tile.Coordinate_X + tile.Coordinate_Z
        y = 3 * tile.Coordinate_Z
        nodes = [vertices.setdefault((x+dx, y+dy), len(vertices))
                 for dx, dy in corners]
        for node in nodes:
            node_tiles[node].append(tile.Tile_ID)
        for i in range(6):
            edges.add(tuple(sorted((nodes[i], nodes[(i+1) % 6]))))
    assert len(node_tiles) == 54 and len(edges) == 72
    neighbors = defaultdict(set)
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    return node_tiles, edges, neighbors

def make_features(placements, tiles):
    node_tiles, edges, neighbors = build_topology(tiles)
    assert set(placements.Game_ID) == set(tiles.Game_ID)
    assert not placements.duplicated(["Game_ID", "Player"]).any()
    assert not tiles.duplicated(["Game_ID", "Tile_ID"]).any()
    maps = {gid: group.set_index("Tile_ID")
            for gid, group in tiles.groupby("Game_ID")}
    output = []
    for gid, group in placements.groupby("Game_ID", sort=False):
        board = maps[gid]
        assert len(group) == 4 and len(board) == 19
        occupied = set(group.Settlement_1) | set(group.Settlement_2)
        assert len(occupied) == 8
        assert not any(a in occupied and b in occupied for a,b in edges)
        all_roads = {tuple(sorted(ast.literal_eval(road)))
                     for col in ["Road_1", "Road_2"] for road in group[col]}
        assert len(all_roads) == 8 and all_roads <= edges
        for player in group.itertuples(index=False):
            production = Counter()
            starting = Counter()
            numbers = set()
            scores = []
            expansion = set()
            for round_no in [1, 2]:
                node = getattr(player, f"Settlement_{round_no}")
                road = ast.literal_eval(getattr(player, f"Road_{round_no}"))
                assert node in road
                score = 0
                for tile_id in node_tiles[node]:
                    tile = board.loc[tile_id]
                    if pd.isna(tile.Resource):
                        continue
                    number = int(tile.Number)
                    pips = 6 - abs(7 - number)
                    score += pips
                    production[tile.Resource] += pips
                    numbers.add(number)
                    if round_no == 2:
                        starting[tile.Resource] += 1
                scores.append(score)
                # Sites reachable by adding exactly one road to this
                # initial road's outer endpoint, with the setup frozen.
                tip = next(n for n in road if n != node)
                if tip not in occupied:
                    for target in neighbors[tip]:
                        edge = tuple(sorted((tip, target)))
                        if (edge not in all_roads and target not in occupied
                            and not (neighbors[target] & occupied)):
                            expansion.add(target)
            total = sum(scores)
            row = {"Game_ID": gid, "Player": player.Player,
                   "Placement_Order": player.Placement_Order,
                   "Settlement_1_Score": scores[0],
                   "Settlement_2_Score": scores[1]}
            row.update({f"{r.title()}_Production": production[r]
                        for r in RESOURCES})
            row.update({"Total_Production": total,
                        "Expected_Cards_Per_Roll": total / 36,
                        "Resource_Diversity": sum(production[r] > 0 for r in RESOURCES),
                        "Number_Diversity": len(numbers),
                        "Any_Production_Probability": sum(6-abs(7-n) for n in numbers)/36})
            row.update({f"Initial_{r.title()}": starting[r] for r in RESOURCES})
            row["Initial_Total"] = sum(starting.values())
            row["Expansion_Options"] = len(expansion)
            output.append(row)
    return pd.DataFrame(output)

if __name__ == "__main__":
    placements = pd.read_csv(DATA / "initial_placements.csv")
    tiles = pd.read_csv(DATA / "map_tiles.csv")
    features = make_features(placements, tiles)
    features.to_csv(DATA / "initial_features.csv", index=False)
    print(features.shape)
    print(features.head().to_string(index=False))
