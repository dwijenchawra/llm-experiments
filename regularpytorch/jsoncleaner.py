import json
import time
from pprint import pprint

with open('/scratch/gilbreth/dchawra/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json', 'r') as file:
    data = json.load(file)

new_data = []
for entity in data:
    if len(entity['conversations']) > 0:
        new_data.append(entity)
    # elif len(entity['conversations']) == 1:
    #     print(entity['conversations'])
    else:
        print('No conversations')

with open('/scratch/gilbreth/dchawra/sgpt_nonempty.json', 'w') as file:
    json.dump(new_data, file, indent=4)


