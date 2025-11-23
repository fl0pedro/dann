from main import main
import json

res = {}
for i in range(4, 10):
    for j in range(3, i):
        for k in (1, 2, 3, 12, 14, 16, 18):
            for t in range(5):
                print(2**i, 2**j, k, t)
                name, r = main(["-d", f"{2**j}", "-s", f"{2**i}", "--id", f"{k}", "--batch-size", "100", "--trial", f"{t}", "--data-dir", "/tmp/cifar10_data", "--dataset", "cifar10"])
                res[name] = r

with open("res.json", "r") as f:
    json.dump(res, f)
