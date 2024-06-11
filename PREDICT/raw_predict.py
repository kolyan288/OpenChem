import runpy
import torch
import copy
from torch.nn.parallel import DataParallel
from openchem.models.openchem_model import predict
from openchem.data.utils import create_loader


config_file = r'example_configs/logp_gcnn_config.py'
config_module = runpy.run_path(config_file)

model_config = config_module.get('model_params', None)
model_config['use_cuda'] = torch.cuda.is_available()
model_object = config_module.get('model', None)

model = model_object(params=model_config)
model = DataParallel(model)
model.module.load_state_dict(torch.load(r'model_OpenChem.pth'))

predict_dataset = copy.deepcopy(model_config['predict_data_layer'])
predict_loader = create_loader(predict_dataset,
                                batch_size=model_config['batch_size'],
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True)


predict(model, predict_loader)

