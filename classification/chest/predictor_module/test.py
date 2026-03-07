# -*- coding: utf-8 -*-

import torch
import util
import numpy as np
import pandas as pd
import os
import pickle

if os.path.exists("/home/gpuuser5/Akshay/MRG/classification/IU-exp/Predictor/file/thresholds.pkl"):
    with open("/home/gpuuser5/Akshay/MRG/classification/IU-exp/Predictor/file/thresholds.pkl", "rb") as f:
        thresholds = pickle.load(f) 
        
if __name__ == '__main__':    
        
    opt, fn = util.parser_model('test')

    avg = opt.avg
    weight = opt.weight
    auc_loss = opt.auc_loss
    init_type = opt.init_type
    image_size = opt.image_size
    batch_size = 1 #opt.batch_size
    class_name = opt.class_name
    gpu_ids = [g for g in range(torch.cuda.device_count())]
    test_loader, test_dataset_size = util.load_data('file/test.csv', 'test', batch_size, image_size, class_name)
    
    if auc_loss == 'auc':
        model_path = '/home/gpuuser5/Akshay/MRG/classification/IU-exp/Predictor/runs/1_0.01_16_-1_-1_0_1.0_337843/model_auc.pkl'
    else:
        model_path = '/home/gpuuser5/Akshay/MRG/classification/IU-exp/Predictor/runs/1_0.01_16_-1_-1_0_1.0_337843/model_final.pkl'
        
    model = torch.load(model_path, weights_only=False)
    
    print('Test')

    if avg == 0:
        with torch.no_grad():
            test_loss = 0.0
            model.eval()
            all_preds, all_labels = [], []
            
            for k, test_data in enumerate(test_loader):
                model.set_input(test_data)
                model('test')
                test_loss += float(model.loss)
                predicted, labels = model.predicted_val()
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())
                if k == 0:
                    p_all = predicted
                    l_all = labels
                    
                else:
                    p_all = np.concatenate((p_all, predicted), axis=0)
                    l_all = np.concatenate((l_all, labels), axis=0)

            _ = util.print_metrics1(l_all, p_all, mode='test', name=fn, class_name=class_name, auc_loss=auc_loss)
            
            df = pd.DataFrame({
            "predicted": all_preds,
            "actual": all_labels
            })
            df.to_csv("/workspace/udit/Akshay/Keshav/class_model_chest/runs/-1_0.01_16_-1_-1_0_1.0_337843/test_predictions.csv", index=False)
            
    else:
        with torch.no_grad():
            test_loss = 0.0
            model.eval()
            all_preds, all_labels = [], []
            for k, test_data in enumerate(test_loader):
                model.set_input(test_data)
                model('test')
                
                test_loss += float(model.loss)
                predicted, labels  = model.predicted_val()
                #print(predicted)
                #pred_classes = np.argmax(predicted, axis=1)
                pred_classes = (np.array(predicted) > 0.5).astype(int)
                #print(pred_classes)
                #predicted, labels = model.predicted_val()
                all_preds.extend(pred_classes.tolist())
                all_labels.extend(labels.tolist())
                if k == 0:
                #if k == 0:
                    p_all = predicted
                    l_all = labels
                else:
                    p_all = np.concatenate((p_all, predicted), axis=0)
                    l_all = np.concatenate((l_all, labels), axis=0)  
            _,_,_,_,_ = util.print_metrics2(l_all, p_all, mode='test', name=fn, class_name=class_name, auc_loss=auc_loss, thresholds = thresholds)
            
            df = pd.DataFrame({
             "predicted": all_preds,
             "actual": all_labels
             })
            df.to_csv("/home/gpuuser5/Akshay/MRG/classification/IU-exp/Predictor/runs/1_0.01_16_-1_-1_0_1.0_337843/tes_preds.csv", index=False)
