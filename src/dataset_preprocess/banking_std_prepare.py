import jsonlines
import pandas as pd
import pyarrow.parquet as pq


input_file = './data/banking77/test-00000-of-00001.parquet'
output_file = './data/banking77/std/test.jsonl'

# input_file = './data/banking77/train-00000-of-00001.parquet'
# output_file = './data/banking77/std/train.jsonl'

# Replace 'your_file.parquet' with the path to your Arrow file
table = pq.read_table(input_file)

# Convert to pandas DataFrame if needed
df = table.to_pandas()
print(df[:5])

with jsonlines.open(output_file, 'w') as writer:
    for index, row in df.iterrows():
        obj = {
            "idx": index,
            "text": row['text'],
            "label": row['label'],
        }
        writer.write(obj)


########### test part ###########
with jsonlines.open(output_file, 'r') as reader:
    for json_obj in reader:
        print(json_obj['text'][:40])
        if json_obj['idx'] > 5:
            break
########### test part ###########