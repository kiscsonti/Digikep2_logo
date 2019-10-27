
import os
from distutils.dir_util import copy_tree, remove_tree


generated_data_path = "/media/kiscsonti/521493CD1493B289/egyetem/mester/1.felev/digikep/logo_class/Digikep2_logo/Generator/Linux/output/"
train_data_path = "/media/kiscsonti/521493CD1493B289/egyetem/mester/1.felev/digikep/logo_class/Digikep2_logo/Generator/Linux/train/"
test_data_path = "/media/kiscsonti/521493CD1493B289/egyetem/mester/1.felev/digikep/logo_class/Digikep2_logo/Generator/Linux/test/"

if not os.path.exists(train_data_path):
    os.mkdir(train_data_path)

copy_tree(generated_data_path, train_data_path)
remove_tree(generated_data_path)