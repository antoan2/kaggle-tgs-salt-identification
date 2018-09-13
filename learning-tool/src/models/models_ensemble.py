import os
import glob

import torch
import torch.nn as nn

from .null_mask_classifier import NullMaskClassifier

class ModelsEnsemble(nn.Module):
    def __init__(self, model_type, models=[]):
        super(ModelsEnsemble, self).__init__()
        self.model_type = model_type
        self.models = models

    def forward(self, x):
        outputs_to_stack = []
        for model in self.models:
            model_output = model(x)
            outputs_to_stack.append(model_output)
        outputs = torch.stack(outputs_to_stack)
        return torch.mean(outputs, dim=0)

    def get_predictions(self, outputs):
        return self.model_type().get_predictions(outputs)


    def add_model(self, model):
        self.models.append(model)

    def save(self, timestamp):
        p_models = '/models/experiment-{timestamp}'.format(timestamp=timestamp)
        os.mkdir(p_models)
        for count, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(p_models, str(count) + '.pkl'))


    def load(self, timestamp):
        p_models = '/models/experiment-{timestamp}'.format(timestamp=timestamp)
        filenames = sorted(glob.glob(os.path.join(p_models, '*.pkl')))
        print(filenames)
        self.models = []
        for filename in filenames:
            model = self.model_type()
            model.load_state_dict(torch.load(filename))
            self.models.append(model)

    def cuda(self):
        for model in self.models:
            model.cuda()
