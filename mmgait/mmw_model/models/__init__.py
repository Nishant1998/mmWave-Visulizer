from .dgcnn import DGCNN
from .mlp import MLP
from .pointnet import PointNet
from .point_transformer import PointTransformer
from .pointmlp import PointMLP

model_map = {
    'dgcnn': DGCNN,
    'mlp': MLP,
    'pointnet': PointNet,
    'pointtransformer': PointTransformer,
    'pointmlp': PointMLP,
}

def make_model(cfg, info):
    num_classes = info['num_classes']

    if cfg.MODEL.MODEL_NAME in ['dgcnn', 'mlp', 'pointnet', 'pointtransformer', 'pointmlp']:
        return model_map[cfg.MODEL.MODEL_NAME](info=info)