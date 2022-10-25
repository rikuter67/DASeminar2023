# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
# �N���X�̒�`�n�܂�
class linearRegression():
	#------------------------------------
	# 1) �w�K�f�[�^����у��f���p�����[�^�̏�����
	# x: �w�K���̓f�[�^�i���̓x�N�g���̎������~�f�[�^����numpy.array�j
	# y: �w�K�o�̓f�[�^�i�f�[�^����numpy.array�j
	# kernelType: �J�[�l���̎�ށi������Fgaussian�j
	# kernelParam: �J�[�l���̃n�C�p�[�p�����[�^�i�X�J���[�j
	def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
		# �w�K�f�[�^�̐ݒ�
		self.x = x
		self.y = y
		self.xDim = x.shape[0]
		self.dNum = x.shape[1]
		
		# �J�[�l���̐ݒ�
		self.kernelType = kernelType
		self.kernelParam = kernelParam
	#------------------------------------

	#------------------------------------
	# 2) �ŏ����@��p���ă��f���p�����[�^���œK��
	# �i����̌v�Z��For����p�����ꍇ�j
	def train(self):
		self.w = np.zeros([self.xDim,1])
		


	#------------------------------------

	#------------------------------------
	# 2) �ŏ����@��p���ă��f���p�����[�^���œK���i�s�񉉎Z�ɂ�荂�����j
	def trainMat(self):
		self.w = np.zeros([self.xDim,1])
		pdb.set_trace()
		ones = np.ones(200)
		ones = np.expand_dims(ones,axis = 0)
		pdb.set_trace()
		x_dash = np.concatenate((self.x,ones),axis = 0)

		ones = np.ones(200)
		ones = np.expand_dims(ones,axis = 0)
		y_ep = np.expand_dims(self.y,axis = 0)
	
		x_dash = np.concatenate((self.x,ones),axis = 0)
	
		x_T = x_dash.T
		x_ex = np.matmul(x_dash,x_T)
		v_ex = np.matmul(y_ep,x_dash.T)
		x_f = np.linalg.inv(x_ex)
	
		self.w = np.matmul(x_f,v_ex.T)
		pdb.set_trace()

		
	#------------------------------------
	
	#------------------------------------
	# 3) �\��
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	def predict(self,x):

		ones = np.ones((1,x.shape[1]))
		test_x_dash = np.concatenate((x,ones),axis = 0)
		y =  np.matmul(test_x_dash.T,self.w)
		pdb.set_trace()
		return y
	#------------------------------------

	#------------------------------------
	# 4) ��摹���̌v�Z
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	# y: �o�̓f�[�^�i�f�[�^���j
	def loss(self,x,y):
		loss = np.sum((y - self.predict(x).T) ** 2) / x.shape[1]
		pdb.set_trace()
		return loss
	#------------------------------------
# �N���X�̒�`�I���
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":
	
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	#myData = rg.artificial(200,100, dataType="1D")
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=False)
	
	# 2) ���`��A���f��
	#regression = linearRegression(myData.xTrain,myData.yTrain)
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
	
	# 3) �w�K�iFor���Łj
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
	
	# 4) �w�K�i�s��Łj
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix: time={0:.4} sec".format(eTime-sTime))

	# 5) �w�K�������f����p���ė\��
	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	# 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
	predict = regression.predict(myData.xTest)
	myData.plot(predict,isTrainPlot=False)
	
#���C���̏I���
#-------------------
	