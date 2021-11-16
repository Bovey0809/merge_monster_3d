#%%
import tqdm
import random
import pandas as pd
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def extract_data(idx, json_results):
    res = []
    for data in json_results:
        if data['image_id'] == idx:
            res.append(data)
    dataframe = pd.DataFrame(res)
    dataframe['x1'] = dataframe.bbox.apply(lambda x: x[0])
    dataframe['y1'] = dataframe.bbox.apply(lambda x: x[1])
    dataframe['x2'] = dataframe.bbox.apply(lambda x: x[0] - x[2])
    dataframe['y2'] = dataframe.bbox.apply(lambda x: x[1] - x[3])
    dataframe.drop(columns=['bbox'], inplace=True)
    return dataframe


def calculate(mine):
    cocoapi = COCO(
        annotation_file='./data/coco/annotations/instances_val2017.json')
    cocodets = cocoapi.loadRes(mine)
    cocoeval = COCOeval(cocoapi, cocodets, 'bbox')
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()


lianfeng = './results-1.json'
mine = './results.bbox.json'

with open(lianfeng, 'r') as f:
    lian = json.load(f)

with open(mine, 'r') as f:
    mine = json.load(f)

# %%

idxs = set([i['image_id'] for i in lian])
idx = random.choice(list(idxs))
idx = 163057
lianres = extract_data(idx, lian)
mineres = extract_data(idx, mine)
merge_data = pd.concat([lianres, mineres]).sort_values(by='score')
# %%
lianres.describe()
# %%
mineres.describe()
# %%

for idx in tqdm.tqdm(idxs):
    lianres = extract_data(idx, lian)
    mineres = extract_data(idx, mine)
    if sum(lianres.score - mineres.score) > 1e-3:
        print(idx)

# %%
idx = 212573
lianres = extract_data(idx, lian)
mineres = extract_data(idx, mine)

# %%
