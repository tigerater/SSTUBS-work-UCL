import json
from jsondiff import diff

with open('sstubsLarge-0104.json') as json_file:
    data1 = json.load(json_file)

with open('miner/enrichedSStuBsLarge-0104.json') as json_file2:
    data2 = json.load(json_file2)

diff(data1,data2)