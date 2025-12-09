import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class XY_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        print("LOSS KWARGS", kwargs)
        self.max_force= kwargs['loss_kwargs'].get('max_force')
        super().__init__()
        
    def forward(self, prediction, target, mask,weight,original,  expweight=0.):
        mag = torch.linalg.norm(target, dim=1, keepdim=True) 
        
        MSE =F.mse_loss(prediction, target, reduction='none')

        loss_weight = torch.exp(torch.minimum(torch.abs(mag),self.max_force*torch.ones_like(mag))*expweight)
    
        return {'mse_loss': MSE.mean(), 'base_loss': torch.mean(MSE*loss_weight) }


class r_MSE_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, prediction, target, mask,weight,original, expweight=0.):
        MSE =F.mse_loss(prediction, target)
        return {'mse_loss': MSE, 'base_loss': MSE }

class weighted_MSE_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, prediction, target, mask,weight,original,  expweight=0.):
        MSE =F.mse_loss(prediction, target, reduction='none')
        w_MSE_t = torch.mul(MSE, weight)
        w_MSE = torch.mean(w_MSE_t)        
        return {'w_mse_loss': w_MSE, 'mse_loss': MSE.mean(), 'base_loss': w_MSE }
    

   
class avg_weighted_MSE_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, prediction, target, mask,weight, original, expweight=0.):
        MSE =F.mse_loss(prediction, target, reduction='none')
        w_MSE_t = torch.mul(MSE, weight)
        w_MSE = torch.mean(w_MSE_t)   
        
    #     def normcorr(input1,input2):
    # input1_0_mean = input1 - np.mean(input1)
    # input2_0_mean = input2 - np.mean(input2)
    # corr = np.mean(input1_0_mean * input2_0_mean)
    # corr_norm = corr/np.std(input1_0_mean)/np.std(input2_0_mean)
    # return corr_norm

        return {'aw_mse_loss': w_MSE + MSE.mean(),'w_mse_loss': w_MSE, 'mse_loss': MSE.mean(), 'base_loss': w_MSE + MSE.mean() }
    

class weighted_MSE_corr_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, prediction, target, mask,weight, original, expweight=0.):
        MSE =F.mse_loss(prediction, target, reduction='none')
        w_MSE_t = torch.mul(MSE, weight)
        w_MSE = torch.mean(w_MSE_t)   
        MSE_mean = MSE.mean()
                
        mask_mean_target = (target.sum())/(mask.sum())
        mask_mean_prediction = (prediction.sum())/(mask.sum())
        mask_mean_original = (original.sum())/(mask.sum())

        # zero_mean_target = target - mask_mean_target
        # std_target = np.sqrt((zero_mean_target * zero_mean_target * mask).sum()/mask.sum())

        # zero_mean_prediction = prediction - mask_mean_prediction
        # std_prediction = np.sqrt((zero_mean_prediction * zero_mean_prediction * mask).sum()/mask.sum())

        # zero_mean_original = original - mask_mean_original
        # std_original = np.sqrt((zero_mean_original * zero_mean_original * mask).sum()/mask.sum())

        # corr_zyxin_nn = (zero_mean_original * zero_mean_prediction * mask).sum()/mask.sum() / std_prediction / std_original
        # corr_pfak_nn = (zero_mean_target * zero_mean_prediction * mask).sum()/mask.sum() / std_prediction / std_original


        ########### since there is mask.sum() was cancelled out in the formula, adjust below
        zero_mean_target = target - mask_mean_target
        std_target = ((zero_mean_target * zero_mean_target * mask).sum()).sqrt()

        zero_mean_prediction = prediction - mask_mean_prediction
        std_prediction = ((zero_mean_prediction * zero_mean_prediction * mask).sum()).sqrt()

        zero_mean_original = original - mask_mean_original
        std_original = ((zero_mean_original * zero_mean_original * mask).sum()).sqrt()

        corr_zyxin_nn = (zero_mean_original * zero_mean_prediction * mask).sum() / std_original / std_prediction
        corr_pfak_nn = (zero_mean_target * zero_mean_prediction * mask).sum() / std_target / std_prediction
        corr_zyxin_pfak = (zero_mean_original * zero_mean_target * mask).sum() / std_original / std_target 


        # corr_zyxin_pfak and corr_zyxin_nn should be the same, corr_pfak_nn should be 1, in ideal cases
        aw_mse_corr_loss_value = w_MSE + MSE_mean/2 + (corr_zyxin_nn - corr_zyxin_pfak)*(corr_zyxin_nn - corr_zyxin_pfak)/1 + (1 - corr_pfak_nn)*(1 - corr_pfak_nn)

    # input1_0_mean = input1 - np.mean(input1)
    # input2_0_mean = input2 - np.mean(input2)
    # corr = np.mean(input1_0_mean * input2_0_mean)
    # corr_norm = corr/np.std(input1_0_mean)/np.std(input2_0_mean)
    # return corr_norm

        return {'aw_mse_corr_loss': aw_mse_corr_loss_value,'w_mse_loss': w_MSE, 'mse_loss': MSE_mean, 'aw_mse_loss': w_MSE + MSE_mean, 'base_loss': aw_mse_corr_loss_value }

loss_function_dict = {                        
                        'xy': XY_loss_dict,
                        'r_mse': r_MSE_loss_dict,
                        'w_mse': weighted_MSE_loss_dict,
                        'aw_mse': avg_weighted_MSE_loss_dict,
                        'aw_mse_corr': weighted_MSE_corr_loss_dict, 
                        }
