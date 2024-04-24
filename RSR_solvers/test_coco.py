import time
import pickle
import torch.linalg as LA
from sklearn.utils.extmath import randomized_svd

from FMS import *
from PRSGM import *


def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)

    return data


mat_path = '/home/tianyu/Research/Eigencontours/Preprocessing/output_coco_val_v1_node_360_iouth_0.0_processMode_centroid/matrix'
mat = load_pickle(mat_path)
print(mat.shape)

# U_svd = load_pickle('/home/tianyu/Research/Eigencontours/Preprocessing/output_coco_train_v1_node_360/U')

dd = [8, 48, 96]
for d in dd:
    fms = FMS(verbose=True)
    U_fms = fms.run(mat.t()/294.15, d)

    with open('/home/tianyu/Research/Eigencontours/Preprocessing/output_coco_val_v1_node_360_iouth_0.0_processMode_centroid/U_fms_'+str(d)+'.pickle', 'wb') as f:
        pickle.dump(U_fms, f, protocol=pickle.HIGHEST_PROTOCOL)

    # prsgm = PRSGM(verbose=True)
    # U_prsgm = prsgm.run(mat/294.15, d)

    # with open('/home/tianyu/Research/Eigencontours/Preprocessing/output_coco_train_v1_node_360/U_dpcp_'+str(d)+'.pickle', 'wb') as f:
    #     pickle.dump(U_prsgm, f, protocol=pickle.HIGHEST_PROTOCOL)
