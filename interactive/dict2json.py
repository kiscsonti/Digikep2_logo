import json

import json

configDict = {
  'w': 512,
  'h': 512,
  'objects': 1000,
  'noise': False,
  'images': 64
}
app_json = json.dumps(configDict, sort_keys=True)

with open('config.json', 'w') as json_file:
  json.dump(configDict, json_file)