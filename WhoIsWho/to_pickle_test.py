import json
import os
import numpy as np
import pandas as pd
from collections import OrderedDict

with open('data/cna_data/cna_test_pub.json') as file:
    cna_valid_pub = json.load(file, object_pairs_hook=OrderedDict)
with open('data/cna_data/cna_test_unass_competition.json') as file:
    cna_valid_unass = json.load(file, object_pairs_hook=OrderedDict)

### cna_valid_unass

cna_valid_unass = pd.DataFrame(cna_valid_unass, columns=['cna_valid_unass'])

cna_valid_unass['cna_valid_unass'] = cna_valid_unass['cna_valid_unass'].apply(lambda x: x.split('-'))

cna_valid_unass['paper_id'] = cna_valid_unass['cna_valid_unass'].apply(lambda x: x[0])
cna_valid_unass['author_idx'] = cna_valid_unass['cna_valid_unass'].apply(lambda x: x[1])

del cna_valid_unass['cna_valid_unass']

cna_valid_unass.to_pickle('./pkl/cna_valid_unass.pkl')

### cna_valid_pub

# paper_id, author_names&orgs, title, venue, year, keywords, abstract
valid_pub_info = pd.DataFrame.from_dict(cna_valid_pub, orient='index').reset_index(drop=True).rename({'id':'paper_id'}, axis=1)

valid_pub_info.head()

valid_pub_info.to_pickle('./pkl/valid_pub_info.pkl')