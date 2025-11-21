from main import main

res = {}
for i in range(2, 10):
    for j in range(i):
        for k in (1, 2, 3, 12, 13, 14, 15):
            for t in range(5):
                name, r = main(["-d", f"{2**j}", "-s", f"{2**i}", "--id", f"{k}", "--batch-size", "100", "--trial", f"{t}", "--data-dir", "/tmp/fmnist_data", "--dataset", "fmnist"])
                res[name] = r

