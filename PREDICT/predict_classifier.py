import runpy
import torch
import copy
from torch.nn.parallel import DataParallel
from openchem.models.openchem_model import predict
from openchem.data.utils import create_loader




config_file = r'example_configs/tox21_rnn_config.py'
config_module = runpy.run_path(config_file)

model_config = config_module.get('model_params', None)
model_config['use_cuda'] = torch.cuda.is_available()
model_object = config_module.get('model', None)

model = model_object(params=model_config)
model = DataParallel(model)
model.load_state_dict(torch.load(r'logs/tox21_rnn_log/checkpoint/epoch_10'))

predict_dataset = copy.deepcopy(model_config['predict_data_layer'])
predict_loader = create_loader(predict_dataset,
                                batch_size=model_config['batch_size'],
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True)



predict(model, predict_loader)

torch.save(model.module.state_dict(), 'model_LOL.pth')