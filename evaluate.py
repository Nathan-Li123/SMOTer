import sys

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from gtr.evaluation.cvid_evaluation import eval_track


if __name__ == '__main__':
    eval_track('output/GTR_MOT/BYTE_CVID_FPN/inference_cvid_val/cvideval')