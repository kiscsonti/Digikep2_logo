
import os
from distutils.dir_util import copy_tree, remove_tree
import json

# generated_data_path = "/media/kiscsonti/521493CD1493B289/egyetem/mester/1.felev/digikep/logo_class/Digikep2_logo/Generator/Linux/output/"
# train_data_path = "/media/kiscsonti/521493CD1493B289/egyetem/mester/1.felev/digikep/logo_class/Digikep2_logo/Generator/Linux/train/"
# test_data_path = "/media/kiscsonti/521493CD1493B289/egyetem/mester/1.felev/digikep/logo_class/Digikep2_logo/Generator/Linux/test/"
#
# if not os.path.exists(train_data_path):
#     os.mkdir(train_data_path)
#
# copy_tree(generated_data_path, train_data_path)
# remove_tree(generated_data_path)


generator_location = "/home/petigep/college/orak/digikep2/Digikep2_logo/Generator/Linux/"
config_location = "/home/petigep/college/orak/digikep2/Digikep2_logo/Generator/Linux/config.json"
# config = generator_location + "config.json"


with open(config_location, "r") as in_f:
    x = json.load(in_f)

print(x)

print(x["image"])
x["image"]["constantSetupRate"] = 0.1
print(x)


with open(config_location, "w") as out_f:
    json.dump(x, out_f, indent=4)
