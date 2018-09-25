import os
import glob

import torch
import torch.nn as nn

class ModelsEnsemble(nn.Module):
    def __init__(self, model_type, **kwargs):
        super(ModelsEnsemble, self).__init__()
        self.model_type = model_type
        self.models = []
        self.kwargs = kwargs

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

    def save(self, experiment_id):
        p_models = '/models/{experiment_id}'.format(experiment_id=experiment_id)
        os.mkdir(p_models)
        for count, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(p_models, str(count) + '.pkl'))


    def load(self, experiment_id):
        p_models = '/models/{experiment_id}'.format(experiment_id=experiment_id)
        filenames = sorted(glob.glob(os.path.join(p_models, '*.pkl')))
        print(filenames)
        self.models = []
        for filename in filenames:
            model = self.model_type(**self.kwargs)
            model.load_state_dict(torch.load(filename))
            self.models.append(model)

    def cuda(self):
        for model in self.models:
            model.cuda()

    def no_grad(self):
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
