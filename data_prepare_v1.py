import pandas as pd
import glob
import os
import gc


train_dir = "./data/train_data/"
test_dir = "./data/test_data/"
output_dir = "./data/data_v1/"

train_files = glob.glob(os.path.join(train_dir, "*.txt"))
train_files.sort()
test_files = glob.glob(os.path.join(test_dir, "*.txt"))
os.makedirs(output_dir, exist_ok=True)

def convert_data(data_files, out_file):
    num_samples = 0
    for txt_file in data_files:
        with open(txt_file, "r", encoding="utf-8") as fd:
            for line in fd:
                rs = line.strip().split("\t")
                log_id = rs[0]
                label_t1, label_t2, label_t3 = rs[1].replace("-", "0"),\
                    rs[2].replace("-", "0"), rs[3].replace("-", "0")
                if "1" in set([label_t1, label_t2, label_t3]):
                    label = "1"
                else:
                    label = "0"
                feat_list = [[] for _ in range(26)]
                for fs in rs[4].split(" "):
                    feat_id, field_id = fs.split(":")
                    feat_list[int(field_id) - 1].append(feat_id)
                feat_list = ["^".join(feat) for feat in feat_list]
                row = [log_id, label, label_t1, label_t2, label_t3] + feat_list
                out_file.write(",".join(row) + "\n")
                num_samples += 1
    return num_samples

fout = open(os.path.join(output_dir, "data.csv"), "w", encoding="utf-8")
headers = ["log_id", "label", "label_t1", "label_t2", "label_t3",
            "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11",
            "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21",
            "f22", "f23", "f24", "f25", "f26"]
fout.write(",".join(headers) + "\n")
num_samples = convert_data(train_files, fout)
fout.close()
print(f"data.csv saved: {num_samples} samples.")

data = pd.read_csv(os.path.join(output_dir, "data.csv"))
valid_samples = 234912 # equal to testing samples
train_samples = len(data) - valid_samples
train_data = data.iloc[:train_samples, :]
valid_data = data.iloc[train_samples:, :]
del data
gc.collect()
train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
valid_data.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
print(f"train.csv saved: {train_samples} samples.")
print(f"valid.csv saved: {valid_samples} samples.")

fout = open(os.path.join(output_dir, "test.csv"), "w", encoding="utf-8")
fout.write(",".join(headers) + "\n")
test_samples = convert_data(test_files, fout)
print(f"test.csv saved: {test_samples} samples.")
