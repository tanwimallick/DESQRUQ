import yaml
import pandas as pd

synthetic_data = pd.read_csv('synthetic_hyperparameters.csv')
path = 'data100/data0' #change the path
file_yaml = path + '0/model/dcrnn_la.yaml'
my_dict = yaml.load(open(file_yaml))
for i in range(0,100): # change the range
    print(i)
    fpath = path + str(i) + '/'
    my_dict['base_dir'] = fpath + 'model'
    my_dict['data']['dataset_dir'] = fpath + 'METR-LA'
    my_dict['data']['graph_pkl_filename'] = fpath + 'sensor_graph/adj_mx.pkl'
    
    my_dict['data']['batch_size'] = int(synthetic_data.iloc[i,0])
    #print('batch_size', int(synthetic_data.iloc[i,0]))
    my_dict['data']['val_batch_size'] = int(synthetic_data.iloc[i,0])
    my_dict['data']['test_batch_size'] = int(synthetic_data.iloc[i,0])
    my_dict['model']['cl_decay_steps'] = int(synthetic_data.iloc[i,1])
    my_dict['train']['epochs'] = int(synthetic_data.iloc[i,2])
    my_dict['model']['filter_type'] = str(synthetic_data.iloc[i,3])
    my_dict['train']['base_lr'] = float(synthetic_data.iloc[i,4])
    my_dict['train']['lr_decay_ratio'] = float(synthetic_data.iloc[i,5])
    my_dict['model']['max_diffusion_step'] = int(synthetic_data.iloc[i,6])
    my_dict['train']['max_grad_norm'] = int(synthetic_data.iloc[i,7])
    my_dict['model']['num_encoder_layers'] = int(synthetic_data.iloc[i,8])
    my_dict['model']['num_decoder_layers'] = int(synthetic_data.iloc[i,9])
    my_dict['model']['rnn_units'] = int(synthetic_data.iloc[i,10])    
    
    with open(fpath + 'model/dcrnn_la.yaml', 'w') as yaml_file:
        yaml.dump(my_dict, yaml_file, default_flow_style=False)


