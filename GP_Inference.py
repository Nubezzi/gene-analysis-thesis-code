from core import *
import itertools
import random
import pandas as pd
import matplotlib.patches as mpatches

def CR_inference_mod(x = None, t = None, trials = 20, epochs = 100, debug=False, learning_rate=1e-2, period_start= 20, period_end = 24, disp = False):
    
    period     = np.arange(period_start, period_end + 1, 2) # Add a step for example 20 -> 22 -> 24
    res_per    = list()
    x_torch    = Array(x).to_tensor() 
    x_scale    = Array(np.arange(1,92,1)).to_tensor() 
    t_torch    = Array(t.reshape(-1,1)).to_tensor()
    t_torch.norm()
    
    for per in period:
        # Init Torch related parameters:
        torch_dtype = torch.float32
        pos_infinite = float("inf")
        res_list   = list()
        # Run the inference:
        loglik_arr = torch.zeros(trials, dtype=torch_dtype) + pos_infinite
        
        for i in range(trials):
            if debug:
                print('-' * 15, 'i_trial=',i + 1, '|' ,'n_trial=',trials, '-' * 15)
            
            # Initialize the starting position:
            p        = torch.tensor(per,dtype=torch.float32)
            l        = torch.tensor((per/6 - per/8)*float(torch.rand(1)) + per/8, dtype=torch.float32, requires_grad=True)
            sigma_f2 = torch.tensor(float(torch.randn(1) * 0.5 + 0.5)**2,dtype=torch.float32, requires_grad=True)        
            sigma_n2 = torch.tensor(float(torch.rand(1) * 0.09 + 0.01)**2,dtype=torch.float32, requires_grad=True)  
           
            para = Parameters(dtype='torch')
            para.p = p
            para.l = l
            para.sigma_f2 = sigma_f2
            para.sigma_n2 = sigma_n2
            
                       
            # Initiate the coressponding GP (Done)
            pse = GaussianProcess(x_arr = x_torch, t_arr = t_torch, kernel_para = para, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat) 
            torch_para_list = []
            for p in [para.p] + [para.l] + [para.sigma_f2] + [para.sigma_n2]:
                if hasattr(p, 'requires_grad') and p.requires_grad:
                    torch_para_list.append(p)
                
            para.torch_para_list = torch_para_list
            #print(torch_para_list)
        
        
            # Configure optimizer(Done)
            if torch_para_list:
                optimizer = torch.optim.Adam(torch_para_list, lr=learning_rate)
        
            # Optimization:
            curr_loglik = pos_infinite
            for j in range(epochs):
                minus_loglik = -pse.cal_marginal_loglik()
                #if debug:
                #    print(f'i={j} -loglik={curr_loglik}')
                optimizer.zero_grad()
                minus_loglik.backward(retain_graph=True) 
                optimizer.step()
                # Clamp
                with torch.no_grad():
                    para.l.clamp_(min=2)
                    para.sigma_f2.clamp_(min=0.05,max=5)
                    para.sigma_n2.clamp_(min=0.001,max=4)
               
                pse.update_kernel_para(para)
                if abs((minus_loglik.item() - curr_loglik) / curr_loglik) < 1e-6:
                    break
                curr_loglik = minus_loglik.item()

            loglik = -minus_loglik.item()
            para.loglik = loglik
            res_list.append(para)
            loglik_arr[i] = loglik
            """
            if debug:
                para.disp(title='estimated parameters')
                print(loglik)
            """
        
        max_ind = np.nanargmax(loglik_arr)
        res = res_list[max_ind]
        res_per.append(res)
        if disp and debug:
            res.disp(title=f'Estimated Parameters')
            test_pse = GaussianProcess(x_arr = x_torch, t_arr = t_torch, kernel_para = res, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat)
            new_pse   = GaussianProcess(x_arr = x_scale, kernel_para = res, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat)
            test_pse.plot_f_gp(new_pse)
            test_pse.plot_data()
            plt.show()
        
            
    
    final_loglik = [param.loglik for param in res_per]
    idx = np.nanargmax(final_loglik)
    final_res = res_per[idx]
    
    if disp:
        final_res.disp(title=f'Final Estimated Parameters')
        final_pse = GaussianProcess(x_arr = x_torch, t_arr = t_torch, kernel_para = final_res, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat)
        new_pse   = GaussianProcess(x_arr = x_scale, kernel_para = final_res, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat)
        final_pse.plot_f_gp(new_pse)
        final_pse.plot_data()
    
    return final_res


def CR_inference_mod_2(x = None, t = None, trials = 20, epochs = 100, debug=True, learning_rate=1e-2, period_start= 20, period_end = 24, disp = False):
    
    period     = np.arange(period_start, period_end + 1, 2) # Add a step for example 20 -> 22 -> 24
    res_per    = list()
    x_torch    = Array(x).to_tensor() 
    x_scale    = Array(np.arange(1,92,1)).to_tensor() 
    t_torch    = Array(t.reshape(-1,1)).to_tensor()
    t_torch.norm()
    
    for per in period:
        # Init Torch related parameters:
        torch_dtype = torch.float32
        pos_infinite = float("inf")
        res_list   = list()
        # Run the inference:
        loglik_arr = torch.zeros(trials, dtype=torch_dtype) + pos_infinite
        
        for i in range(trials):
            if debug:
                print('-' * 15, 'i_trial=',i + 1, '|' ,'n_trial=',trials, '-' * 15)
            
            # Initialize the starting position:
            p        = torch.tensor(300,dtype=torch.float32)
            l        = torch.tensor((per/4 - per/6)*float(torch.rand(1)) + per/6, dtype=torch.float32, requires_grad=True)
            sigma_f2 = torch.tensor(float(torch.randn(1) * 0.5 + 0.5)**2,dtype=torch.float32, requires_grad=True)        
            sigma_n2 = torch.tensor(float(torch.rand(1) * 0.9 + 0.1)**2,dtype=torch.float32, requires_grad=True)  
           
            para = Parameters(dtype='torch')
            para.p = p
            para.l = l
            para.sigma_f2 = sigma_f2
            para.sigma_n2 = sigma_n2
            
                       
            # Initiate the coressponding GP (Done)
            pse = GaussianProcess(x_arr = x_torch, t_arr = t_torch, kernel_para = para, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat) 
            torch_para_list = []
            for p in [para.p] + [para.l] + [para.sigma_f2] + [para.sigma_n2]:
                if hasattr(p, 'requires_grad') and p.requires_grad:
                    torch_para_list.append(p)
                
            para.torch_para_list = torch_para_list
            #print(torch_para_list)
        
        
            # Configure optimizer(Done)
            if torch_para_list:
                optimizer = torch.optim.Adam(torch_para_list, lr=learning_rate)
        
            # Optimization:
            curr_loglik = pos_infinite
            for j in range(epochs):
                minus_loglik = -pse.cal_marginal_loglik()
                """
                if debug:
                    print(f'i={j} -loglik={curr_loglik}')
                    
                """
                optimizer.zero_grad()
                minus_loglik.backward(retain_graph=True) 
                optimizer.step()
                # Clamp
                with torch.no_grad():
                    para.l.clamp_(min=8)
                    para.sigma_f2.clamp_(min=0.05,max=1)
                    para.sigma_n2.clamp_(min=0.25,max=5)
               
                pse.update_kernel_para(para)
                if abs((minus_loglik.item() - curr_loglik) / curr_loglik) < 1e-6:
                    break
                curr_loglik = minus_loglik.item()

            loglik = -minus_loglik.item()
            para.loglik = loglik
            res_list.append(para)
            loglik_arr[i] = loglik
            """
            if debug:
                para.disp(title='estimated parameters')
                print(loglik)
            """
        
        max_ind = np.nanargmax(loglik_arr)
        res = res_list[max_ind]
        res_per.append(res)
        """
        if disp and debug:
            res.disp(title=f'Estimated Parameters')
            test_pse = GaussianProcess(x_arr = x_torch, t_arr = t_torch, kernel_para = res, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat)
            new_pse   = GaussianProcess(x_arr = x_scale, kernel_para = res, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat)
            test_pse.plot_f_gp(new_pse)
            test_pse.plot_data()
            plt.show()
        """
        
            
    
    final_loglik = [param.loglik for param in res_per]
    idx = np.nanargmax(final_loglik)
    final_res = res_per[idx]
    
    if disp:
        final_res.disp(title=f'Final Estimated Parameters')
        final_pse = GaussianProcess(x_arr = x_torch, t_arr = t_torch, kernel_para = final_res, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat)
        new_pse   = GaussianProcess(x_arr = x_scale, kernel_para = final_res, ker_func = pse_kernel, covmat_func = cal_pse_cov_mat)
        final_pse.plot_f_gp(new_pse)
        final_pse.plot_data()
    
    return final_res

if __name__ == '__main__': 
    import sys
    print(f"This is the name of the script: {sys.argv[0]}") 
    print(f"Number of arguments: {len(sys.argv)}")
    print(f"The arguments are: {str(sys.argv)}") 
    
    #raise Exception()
    
    filename = sys.argv[1]
    out_dir = sys.argv[2]
    num = filename[-6:-4]
    
    y   = pd.read_csv(filename)
    y = y.iloc[: , 1:]
    data_2 = pd.read_csv('HKO_partial.csv')
    print(y)
    annot  = y.iloc[:,0:2]
    gene  = np.array(y.iloc[0,2:], dtype = np.float32).reshape([-1,2])
        
    p = []
    cri_1 = []
    cri_2 = []
    cri_3 = []
    count = 0
        
    for x in range(len(y)):  
        gene  = np.array(y.iloc[x,2:], dtype = np.float32).reshape([-1,2])
        x_arr = np.arange(0,92+1,4)
        plt.figure(figsize=(20,8))
            
        plt.subplot(1,2,1)
        param_1 = CR_inference_mod_2(x = x_arr, disp= True, t = gene.T.flatten(), trials= 10, period_start=24) 
        plt.xlabel("ZT (hours)")
        plt.ylabel("Normalized gene expression")
        plt.title("Null model of GP")
            
        plt.subplot(1,2,2)
        param_2 = CR_inference_mod(x = x_arr, disp = True, t = gene.T.flatten(), trials= 10, period_start=24, debug = False)
        plt.xlabel("ZT (hours)")
        plt.ylabel("Normalized gene expression")
        plt.title("Circadian detection model with GP")
        plt.suptitle(f"Expression of gene {y.iloc[x,0]}",fontsize=20)
            
        p.append(param_2.p.item())
        cri_1.append(np.sqrt(param_2.sigma_f2.item()/param_2.sigma_n2.item()))
        cri_2.append(np.sqrt(param_2.sigma_f2.item()/param_2.sigma_n2.item())/param_2.l.item())
        cri_3.append((param_2.loglik-param_1.loglik)/len(gene.flatten()))
        
        red_patch = mpatches.Patch(color='red',label= f'SNR = {round(cri_1[x],2)}; nSNR = {round(cri_2[x],2)}; LLR = {round(cri_3[x],2)}')
        blue_patch = mpatches.Patch(color='blue', label= f'sigma_f2 = {round(param_2.sigma_f2.item(),2)}; sigma_n2 = {round(param_2.sigma_n2.item(),2)}')
        yellow_patch = mpatches.Patch(color='yellow', label = f'period = {param_2.p.item()}')
        plt.legend(handles=[red_patch, blue_patch, yellow_patch])
        # plt.show()
        plt.savefig(f"{out_dir}/gene{x}_{num}.png")
        count += 1

    cols = {'Period':p,'SNR':cri_1,'nSNR':cri_2,'LLR':cri_3}
    df   = pd.DataFrame(cols)
    print(df)
    print(annot)
    df_final = pd.concat([annot, df], axis=1, join='inner')
    print(df_final)
    df_final.to_csv(f"{out_dir}/GP_{out_dir}_{num}.csv")
    
    

    
    
    
    
    
    
    
