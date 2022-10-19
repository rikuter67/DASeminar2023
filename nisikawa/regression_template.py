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
		self.o = np.ones([self.dNum,1]).T				#1が200個ある行列
		np.expand_dims(self.x,0)						#self.xの次元を0の方向に増やす
		self.x_add = np.concatenate((self.x, self.o),0)	#self.x, self.oを合体する

		self.x_add_t = self.x_add.T
		f_x = np.matmul(self.x_add, self.x_add_t)		#sigma(x' * x'.T)の部分
		xsum_inv = np.linalg.inv(f_x)					#f_xの逆行列を取る
		f_y = np.matmul(self.x_add,self.y)				#sigma(x * y)

		self.w = np.matmul(xsum_inv,f_y)
		
	#------------------------------------
	
	#------------------------------------
	# 3) �\��
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	def predict(self,x):
		
		y = self.w[0] * x + self.w[1]
		return y
	#------------------------------------

	#------------------------------------
	# 4) 二乗損失の計算
	# x: 入力データ（入力次元　×　データ数）
	# y: 出力データ（データ数）
	def loss(self,x,y):
		#pdb.set_trace()
		loss = np.sum((y - self.predict(x)) ** 2) / x.shape[1]
		return loss
	#------------------------------------
# �N���X�̒�`�I���
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":
	
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	#myData = rg.artificial(200,100, dataType="1D")
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	
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
	predict = np.squeeze(predict)
	myData.plot(predict,isTrainPlot=False)
	
#���C���̏I���
#-------------------
	