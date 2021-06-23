from nets.textrcnn_attn import TextRCNNAttnConfig, TextRCNNAttn
from nets.textrcnn import TextRCNNConfig, TextRCNN
from nets.dpcnn import DPCNNConfig, DPCNN
from run import RunConfig


def gen_net(net_name):
    run_config = RunConfig()

    if net_name == 'rcnn':
        model_config = TextRCNNConfig()
        return TextRCNN(run_config, model_config), run_config, model_config
    elif net_name == 'dpcnn':
        model_config = DPCNNConfig()
        return DPCNN(run_config, model_config), run_config, model_config
    elif net_name == 'rcnnattn':
        model_config = TextRCNNAttnConfig()
        return TextRCNNAttn(run_config, model_config), run_config, model_config
