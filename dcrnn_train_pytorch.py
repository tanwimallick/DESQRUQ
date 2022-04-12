from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import os
import pandas as pd

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
from lib.metrics import calculate_metrics

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train()

        data_tag = supervisor_config.get('data').get('dataset_dir')

        folder = data_tag + '/results/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        mean_score, y_preds, y_truths = supervisor.evaluate('test')
        for horizon_i in range(len(y_preds)):
            y_pred = y_preds[horizon_i]
            eval_dfs = y_truths[horizon_i]
            eval_dfs = pd.DataFrame(eval_dfs)
            df_pred0 = pd.DataFrame(y_pred[:,:,0])
            df_pred1 = pd.DataFrame(y_pred[:,:,1])
            df_pred2 = pd.DataFrame(y_pred[:,:,2])
            filename = os.path.join('%s/results/'%data_tag, 'dcrnn_speed_mu_prediction_%s.h5' %str(horizon_i+1))
            df_pred0.to_hdf(filename, 'results')
            filename = os.path.join('%s/results/'%data_tag, 'dcrnn_speed_sigma_prediction_%s.h5' %str(horizon_i+1))
            df_pred1.to_hdf(filename, 'results')
            filename = os.path.join('%s/results/'%data_tag, 'dcrnn_speed_high_prediction_%s.h5' %str(horizon_i+1))
            df_pred2.to_hdf(filename, 'results')
            filename = os.path.join('%s/results/'%data_tag, 'dcrnn_speed_org_%s.h5' %str(horizon_i+1))
            eval_dfs.to_hdf(filename, 'results')

            mae, mape, rmse = calculate_metrics(df_pred1, eval_dfs, null_val=0)
            print(
                "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                    horizon_i + 1, mae, mape, rmse
                ))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
