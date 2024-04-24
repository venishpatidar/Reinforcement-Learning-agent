from scipy import stats
import json
import os

directory = "results"
layouts = sorted(os.listdir(directory + '/ActorCriticAgent'))
pairs = [
    ("ActorCriticAgent", "ApproximateQAgent"),
    ("ActorCriticAgent", "ReinforceAgent"),
    ("ApproximateQAgent", "ReinforceAgent"),
]
t_test_results = {}

for l in layouts:
    data = {"ActorCriticAgent": [], "ApproximateQAgent": [], "ReinforceAgent": []}
    for c in data:
        with open("/".join([directory, c, l])) as f:
            data[c] = [float(n) for n in f.read().split("\n") if n]
    
    t_test_results[l] = {}

    for c1, c2 in pairs:
        d = {}
        t_test_results[l][c1 + "-" + c2] = d
        d["t_stat"], d["p_value"] = stats.ttest_ind(data[c1], data[c2])
        d["null_hypothesis"] = "rejected" if d["p_value"] < 0.05 else "accepted"

with open("t_test_results.json", "w") as f:
    json.dump(t_test_results, f, indent=4)
