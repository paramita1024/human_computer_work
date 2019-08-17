from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
import random

class generate_data:
	def __init__(self,n,dim,list_of_std,std_y=None):
		self.n=n
		self.dim=dim
		self.list_of_std=list_of_std
		self.std_y=std_y
	
	def generate_X(self):
		self.X=rand.uniform(-7,7,(self.n,self.dim))


	def white_Gauss(self,std=1):
		return rand.normal(0,std,self.n)

		
	def generate_Y_sigmoid(self):
		def sigmoid(x):
			return 1/float(1+np.exp(-x))
	
		# self.w=rand.uniform(-1,1,self.dim)
		# self.Y=np.array(map(self.sigmoid,self.X.dot(self.w)))
		self.Y=np.array(map(sigmoid,self.X.flatten()))
		x=self.X
		y=self.Y
		plt.scatter(x,y,c='red')
		plt.show()

	def generate_Y_Gauss(self):
		def gauss(x):
			divide_wt = np.sqrt(2*np.pi)*std
			return np.exp(-(x*x)/float(2*std*std))/divide_wt
		# self.w=rand.uniform(0,1,self.dim)
		std=self.std_y
		self.Y=np.array(map(gauss,self.X.flatten()))
		# self.Y=self.X.dot(self.w)+self.white_Gauss(std=0.01)
		x=self.X
		y=self.Y
		plt.scatter(x,y,c='red')
		plt.show()

	def generate_Y_Mix_of_Gauss(self,no_Gauss,prob_Gauss):
		self.Y=np.zeros(self.n)
		for itr,p in zip(range(no_Gauss),prob_Gauss):
			w=rand.uniform(0,1,self.dim)
			self.Y += p*self.X.dot(w)

	def generate_human_prediction(self):
		
		self.human_pred={}
		for std in self.list_of_std:
			self.human_pred[str(std)]=self.Y+self.white_Gauss(std=std)
		# self.human_pred=self.Y+white_Gauss(std=0.01)

	def append_X(self):
		# print self.X.shape
		self.X=np.concatenate((self.X, np.ones((self.n,1))) ,axis=1)

	def split_data(self,frac):

		indices=np.arange(self.n)
		random.shuffle(indices)
		num_train=int(frac*self.n)
		indices_train=indices[:num_train]
		indices_test=indices[num_train:]
		self.Xtest=self.X[indices_test]
		self.Xtrain=self.X[indices_train]
		self.Ytrain=self.Y[indices_train]
		self.Ytest=self.Y[indices_test]
		self.human_pred_train={}
		self.human_pred_test={}
		for std in self.list_of_std:
			self.human_pred_train[str(std)]=self.human_pred[str(std)][indices_train]
			self.human_pred_test[str(std)]=self.human_pred[str(std)][indices_test]


		n_test=self.Xtest.shape[0]
		n_train=self.Xtrain.shape[0]
		self.dist_mat=np.zeros((n_test,n_train))
		for te in range(n_test):
			for tr in range(n_train):
				self.dist_mat[te,tr]=LA.norm(self.Xtest[te]-self.Xtrain[tr] )

	def visualize_data(self):
		x=self.X[:,0].flatten()
		y=self.Y
		plt.scatter(x,y)
		plt.show()
		
def main():
	n=20
	dim=1
	frac=0.8
	option='sigmoid'
	if option=='sigmoid':
		list_of_std=np.array([.01,.05,.1,.5])#([0.001,0.01,0.1,0.5,1]) 
		obj=generate_data(n,dim,list_of_std)
		obj.generate_X()
		obj.generate_Y_sigmoid()
		obj.generate_human_prediction()
		obj.append_X()
		obj.split_data(frac)
		# obj.visualize_data()
		save(obj,'../Synthetic_data/data_sigmoid_n_20')
		del obj
	
	
	# generate sigmoid
	if option=='Gauss':
		std_y=2
		list_of_std=np.array([0.001,.005,0.01,.05,.1])#([0.001,0.01,0.1,0.5,1]) 
		obj=generate_data(n,dim,list_of_std,std_y)
		obj.generate_X()
		obj.generate_Y_Gauss()
		obj.generate_human_prediction()
		obj.append_X()
		obj.split_data(frac)
		save(obj,'../Synthetic_data/data_Gauss_'+str(std_y))
		del obj

	# gene<!-- <!-- ######rate gaussian
#######	# obj=generate_data(n,dim)
#######	# obj.generate_X()
#######	# obj.generate_Y_Mix_of_Gauss(2,[.5,.5])
#######	# obj.generate_human_prediction(list_of_std)
#######	# save(obj,'../Synthetic_data/data_Mix_Gaussian')
#######	# del obj
######	 --> -->
if __name__=="__main__":
	main()
