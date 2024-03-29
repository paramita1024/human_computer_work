import time
import os
import sys 
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data 
from triage_human_machine import triage_human_machine


class eval_triage:
	def __init__(self,data_file,real_flag=None, real_wt_std=None):
		self.data=load_data(data_file)
		self.real=real_flag		
		self.real_wt_std=real_wt_std


	def eval_loop(self,param,res_file,option):	
		res=load_data(res_file,'ifexists')		
		for std in param['std']:
			if self.real:
				data_dict=self.data
				triage_obj=triage_human_machine(data_dict,self.real)
			else:
				if self.real_wt_std:
					data_dict = {'X':self.data['X'],'Y':self.data['Y'],'c': self.data['c'][str(std)]}
					triage_obj=triage_human_machine(data_dict,self.real_wt_std)
				else:
					test={'X':self.data.Xtest,'Y':self.data.Ytest,'human_pred':self.data.human_pred_test[str(std)]}
					data_dict = {'test':test,'dist_mat':self.data.dist_mat,  'X':self.data.Xtrain,'Y':self.data.Ytrain,'human_pred':self.data.human_pred_train[str(std)]}
					triage_obj=triage_human_machine(data_dict,False)
			if str(std) not in res:
				res[str(std)]={}
			for K in param['K']:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in param['lamb']:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					# res[str(std)][str(K)][str(lamb)]['greedy'] = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim='greedy')
					print 'std-->', std, 'K--> ',K,' Lamb--> ',lamb
					res_dict = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim=option)
					res[str(std)][str(K)][str(lamb)][option] = res_dict
		save(res,res_file)


def main():
	
	#---------------Synthetic Data------------------------
	s = ['sigmoid_n500d10','Gauss_n500d10'][int(sys.argv[1])]
	if 'sigmoid' in s : # Sigmoid 
		list_of_std = [0.001]#,.005,0.01,.05]
		list_of_lamb= [0.001]#,0.005,0.01,0.05] #[0.01,0.05] [.1,.5] # [0.0001,0.001,0.01,0.1,.4,.6,.8]#[[int(sys.argv[2])]] # [0.0001,0.001,0.01,0.1] #
	else:
		list_of_std =[0.001]#,0.005,0.01,.05] # [.01,.05,.1,.5, 1] # [[0.001,0.01,0.1][int(sys.argv[1])]]
		list_of_lamb= [0.005]#,.0005] #0.0001, [.1,.5] # [0.0001,0.001,0.01,0.1,.4,.6,.8]#[[int(sys.argv[2])]] # [0.0001,0.001,0.01,0.1] #
	option =['greedy','diff_submod','distort_greedy','stochastic_distort_greedy','kl_triage' ][int(sys.argv[2])]
	if option in ['diff_submod', 'stochastic_distort_greedy' ]:
		list_of_K =[0.1,0.2,0.3,.4,0.5,.6,.7,.8,.9] 
	else:
		list_of_K=[0.99]
	data_file='../Synthetic_data/data_dict_' +s
	res_file='../Synthetic_data/res_' +s
	# ---------------------------------------------------------
	obj=eval_triage(data_file,real_wt_std=True)
	param={'std':list_of_std,'K':list_of_K,'lamb':list_of_lamb}
	obj.eval_loop(param,res_file,option) 
	#---------Real Data-------------------------------------------
	# list_of_std=[0.0] # [[0.0][int(sys.argv[1])]] # np.array([0.0]int(sys.argv[1])]])
	# list_of_std = [0.001,0.01,0.1] # [[0.001,0.01,0.1][int(sys.argv[1])]]
	# list_of_K=[0.4]#[0.2,0.6,0.8]#[0.01,0.1,0.2,0.3,.4,0.5,.6,.7,.8,.9,.99]
	# list_of_lamb= [.5] # [0.0001,0.001,0.01,0.1,.4,.6,.8]#[[int(sys.argv[2])]] # [0.0001,0.001,0.01,0.1] #
	# param={'std':list_of_std,'K':list_of_K,'lamb':list_of_lamb}
	# option='distort_greedy'
	# path = '../Real_Data/STARE/5/'
	# path = '../Real_Data/STARE/11/'
	# path = '../Real_Data/Messidor/MESSIDOR/Risk_edema/' 
	# path = '../Real_Data/Messidor/MESSIDOR/Retino_grade/'
	# path = '../Real_Data/EyePAC/'
	# path = '../Real_Data/Messidor/Messidor_txt/'
	# selected_cols =[ str(i) for i in [7,8,13,14,15,16] ] 
	# path = '../Real_Data/CheXpert/data/'+selected_cols[int(sys.argv[1])]+'/'
	#----------------------------------
	# data_file = path + 'data_split_pca'
	# res_file= path + 'res_pca'
	# obj=eval_triage(data_file,real_wt_std=True)
	# obj.eval_loop(param,res_file,option)
	#---------------Real Data-----------------------------
	# path='../Real_Data/Hatespeech/Davidson/'
	# data_file= path + 'input_tr'
	# res_file= path + 'res'+'_lamb_'+str(list_of_lamb[0])
	# obj=eval_triage(data_file,real_flag=True)
	# obj.eval_loop(param,res_file,option)
	
	#----------------------------------------------------
	#----------------------------------------------------
	# path='../Real_Data/Movielens/ml-20m/'
	# data_file= path + 'data_tr_splitted'
	# res_file= path + 'res'+'_lamb_'+str(list_of_lamb[0])
	# obj=eval_triage(data_file,real_flag=True)
	# obj.eval_loop(param,res_file,option)
	#---------------------------------------------------
	# path='../Real_Data/BRAND_DATA/'
	# data_file= path + 'data_ht4_vec_split_1'	
	# res_file= path + 'res_ht4_1'# +'_lamb_'+str(list_of_lamb[0])
	# obj=eval_triage(data_file,real_flag=True)
	# obj.eval_loop(param,res_file,option)

if __name__=="__main__":
	main()

	# def eval_kl(self,param,res_file,option):
	# 	res=load_data(res_file,'ifexists')		
	# 	for std in param['std']:
	# 		if self.real:
	# 			data_dict=self.data
	# 		if self.real_wt_std:
	# 			test ={'X':self.data['test']['X'],'Y':self.data['test']['Y'],'c':self.data['test']['c'][str(std)]}
	# 			data_dict = {'X':self.data['X'],'Y':self.data['Y'],'c': self.data['c'][str(std)], 'test':test}

	# 		if str(std) not in res:
	# 			res[str(std)]={}
	# 		for K in param['K']:
	# 			if str(K) not in res[str(std)]:
	# 				res[str(std)][str(K)]={}
	# 			for lamb in param['lamb']:
	# 				if str(lamb) not in res[str(std)][str(K)]:
	# 					res[str(std)][str(K)][str(lamb)]={}

	# 				print 'std-->', std, 'K--> ',K,' Lamb--> ',lamb
	# 				data_dict['lamb'] = lamb

	# 				kl_obj=kl_triage(data_dict)
	# 				res[str(std)][str(K)][str(lamb)][option] = {'train_res':0,'test_res':{'nearest':{'error':kl_obj.testing(K)} } }
	# 	save(res,res_file)