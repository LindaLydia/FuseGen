import jsonlines
import pandas as pd
import pyarrow.parquet as pq


# input_file = './data/agnews/test-00000-of-00001.parquet'
# output_file = './data/agnews/std/test.jsonl'
# input_file = './data/agnews/train-00000-of-00001.parquet'
# output_file = './data/agnews/std/train.jsonl'
# # input_file = './data/agnews/validation-00000-of-00001.parquet'
# # output_file = './data/agnews/std/test.jsonl'

# # Replace 'your_file.parquet' with the path to your Arrow file
# table = pq.read_table(input_file)


# # Convert to pandas DataFrame if needed
# df = table.to_pandas()
# print(df[:5])

# # label_map = {"World": 0, "Sports": 1, "Business": 2, "Science/Technology": 3}

# with jsonlines.open(output_file, 'w') as writer:
#     for index, row in df.iterrows():
#         # print(f"{index=}, {row=}")
#         obj = {
#             "idx": index,
#             "text": row['text'],
#             "label": row['label'],
#         }
#         writer.write(obj)


# ########### test part ###########
# with jsonlines.open(output_file, 'r') as reader:
#     for json_obj in reader:
#         print(json_obj['text'][:40])
#         if json_obj['idx'] > 5:
#             break
# ########### test part ###########



# ########### get sports news out for further decomposition ###########
input_file = './data/agnews/test-00000-of-00001.parquet'
output_file = './data/markednews/test.csv'
input_file = './data/agnews/train-00000-of-00001.parquet'
output_file = './data/markednews/train.csv'

table = pq.read_table(input_file)
df = table.to_pandas()
print(df[-10:])

for index, row in df.iterrows():
    if '$' in row['text']:
        print(f"{index=}")
        df.loc[index, 'label'] = 4
print("after")
print(df[-10:])
print(f"{len(df)=}")

df.to_csv(output_file)


# ########### get sports news out for further decomposition ###########
