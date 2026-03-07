# -*- coding: utf-8 -*-

import torch
import util
import numpy as np
#from gradcam import run
import pickle
import pandas as pd
import os
import pickle

if os.path.exists("file/thresholds.pkl"):
    with open("file/thresholds.pkl", "rb") as f: 
        #print("yes")
        thresholds = pickle.load(f)
             
if __name__ == '__main__':   
        
    opt, fn = util.parser_model('test')

    avg = opt.avg
    weight = opt.weight
    auc_loss = opt.auc_loss
    init_type = opt.init_type
    image_size = opt.image_size
    batch_size = opt.batch_size
    class_name = opt.class_name
    gpu_ids = [g for g in range(torch.cuda.device_count())]
    test_loader, test_dataset_size = util.load_data('file/test.csv', 'test', batch_size, image_size, class_name)
    
    if auc_loss == 'auc':
        model_path = fn+'/model_auc.pkl'
    else:
        model_path = fn+'/model_final.pkl'
        
    model = torch.load(model_path, weights_only=False)
    #print(type(model))
    
    print('Test')

    if avg == 0:
        with torch.no_grad():
            test_loss = 0.0
            model.eval()
            all_preds, all_labels = [], []
            p_all = None
            l_all = None
            
            for k, test_data in enumerate(test_loader):
                model.set_input(test_data)
                model('test')
                test_loss += float(model.loss)
                predicted, labels = model.predicted_val()
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())
                if p_all is None:
                  p_all = predicted
                  l_all = labels
                else:
                  # Concatenate with previous predictions
                  p_all = np.concatenate((p_all, predicted), axis=0)
                  l_all = np.concatenate((l_all, labels_np), axis=0)

            _ = util.print_metrics1(l_all, p_all, mode='test', name=fn, class_name=class_name, auc_loss=auc_loss)
            
            df = pd.DataFrame({
            "predicted": all_preds,
            "actual": all_labels
            })
            df.to_csv("runs/last/val_preds.csv", index=False)
            
    else:
        with torch.no_grad():
            test_loss = 0.0
            model.eval()
            all_preds, all_labels = [], []
            predictions_dict = {}
            for k, test_data in enumerate(test_loader):
                keys_list = list(test_data.keys())
                seventh_value = test_data[keys_list[7]]
                #print(seventh_value)
                model.set_input(test_data)
                model('test')
                
                test_loss += float(model.loss)
                predicted, labels  = model.predicted_val()
                
                result_list = {"Image_ID": seventh_value[0], "predicted": predicted.tolist()[0]}
                
                with open(f"image_predictions/{seventh_value[0]}.pkl", "wb") as f:
                   pickle.dump(result_list, f)

                pred_classes = (np.array(predicted) > thresholds).astype(int)
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
            _, _, _, _ = util.print_metrics2(l_all, p_all, mode='test', name=fn, class_name=class_name, auc_loss=auc_loss, thresholds = thresholds)
            
            df = pd.DataFrame({
             "predicted": all_preds,
             "actual": all_labels
             })
            df.to_csv("runs/last/val_preds.csv", index=False)
