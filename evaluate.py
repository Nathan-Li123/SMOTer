import sys

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from smoter.evaluation.bensmot_evaluation import eval_track


if __name__ == '__main__':
    eval_track('output/SMOTer/BYTE_BENSMOT_FPN/inference_bensmot_val/bensmoteval')