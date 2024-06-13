import jsonlines
import pandas as pd

input_file = './data/medical-cancer-doc/alldata_1_for_kaggle.csv'
output_file = './data/medical-cancer-doc/std/test.jsonl'

df = pd.read_csv(input_file, encoding='latin1')
# {"idx": 14, "text": "lend some dignity to a dumb story ", "label": 1}
print(df[:5])

label_mapping = {"Thyroid_Cancer": 0, "Colon_Cancer": 1, "Lung_Cancer": 2}

with jsonlines.open(output_file, 'w') as writer:
    for index, row in df.iterrows():
        obj = {
            "idx": index,
            "text": row['a'],
            "label": label_mapping[row['0']],
        }
        writer.write(obj)

# ########### test part ###########
# with jsonlines.open(output_file, 'r') as reader:
#     for json_obj in reader:
#         print(json_obj['text'][:40])
#         if json_obj['idx'] > 5:
#             break
# ########### test part ###########
