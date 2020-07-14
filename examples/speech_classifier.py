import numpy as np
import torch

from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor

slu_model_path = 'espnet/egs/e2e_slu/slu1/exp/sbert__tune_2_0_warmup_700000_epoch_100_lr_30_loss_l1_loss/model.loss.best'

classifier_model_path = {
    'fluentai': 'fluentai_sbert/pytorch_model.bin',
    'swbd': 'swbd_sbert/pytorch_model.bin',
    'mrda': 'mrda_sbert/pytorch_model.bin',
}

class SpeechClassifier(torch.nn.Module):
        def __init__(self, task):
            super(SpeechClassifier, self).__init__()
            torch.manual_seed(1)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False  # https://github.com/pytorch/pytorch/issues/6351

            self.slu = load_trained_model(slu_model_path)[0]
            self.classifier = torch.load(classifier_model_path[task], map_location='cpu')

        def forward(self, xs, labels=None):
            ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
            xs = [to_device(self.slu, to_torch_tensor(xx).float()) for xx in xs]
            xs_pad = pad_list(xs, 0.0)
            embeddings = self.slu(xs_pad, ilens, None)
            outputs = self.classifier(embeddings, labels)
            return outputs
