import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.signal import argrelmin, argrelmax
import scipy.stats.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#import bayesian as offcd
from functools import partial
pd.options.mode.chained_assignment = None


def ReadCSV(filename):
	return pd.read_csv(filename,header=None,names=['sample','x','y','z','label'],index_col=False,dtype={'sample':np.int64})

def ShowData(data):
	data.info()
	#print data.head()
	print data.drop_duplicates(['label'])['label']
	# check for missing data
	nan_flag = False
	for c in data.columns:
		if any(data[c] == np.nan):
			print c, 'contains NaNs'
			nan_flag = True
	if not nan_flag:
		print 'No missing values.'
	PlotData_before(data)

def PlotData_before(data):
	_fig,_axes = plt.subplots(nrows=7, ncols=3, figsize=(20, 9))
	name = ['x','y','z']
	for n in xrange(0,3):
		for c in xrange(1, 8):
			_axes[c-1][n].plot(data[data['label']==c][name[n]], linewidth=.5)
			_axes[c-1][n].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	pass

def PlotData(data,axes,index):
	x = (index-1)/4
	y = index - x*4 - 1
	for c in xrange(1, 8):
		axes[x][y].plot(data[data['label'] == c]['m'], linewidth=.5)
	axes[x][y].set_xlim(0, len(data))
	axes[x][y].set_ylim(3400, 4100)
	axes[x][y].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	pass

def PlotData_after(data):
	_fig,_axes = plt.subplots(nrows=7, ncols=4, figsize=(20, 9))
	name = ['x','y','z','m']
	for n in xrange(0,4):
		for c in xrange(1,8):
			filter_test_data = data[data['label']==c][name[n]]
			filter_test_y = data[data['label']==c][name[n]+'l'] 
			filter_test_x = xrange(len(filter_test_y))
			_axes[c-1][n].plot(filter_test_x, filter_test_data, linewidth=1, label='data')
			_axes[c-1][n].plot(filter_test_x, filter_test_y, linewidth=2, label='filtered data')
			_axes[c-1][n].set_xlim(0, len(filter_test_data))
			#_axes[c-1][n].set_ylim(1900, 2300)

def butter_lowpass(cutoff, fs, order=5): 
	nyq = 0.5 * fs 
	normal_cutoff = cutoff / nyq 
	b, a = butter(order, normal_cutoff, btype='low', analog=False) 
	return b, a 

def butter_lowpass_filter(data, cutoff, fs, order=5): 
	b, a = butter_lowpass(cutoff, fs, order=order) 
	y = lfilter(b, a, data) 
	return y 

def butter_highpass(cutoff, fs, order=5): 
	nyq = 0.5 * fs 
	normal_cutoff = cutoff / nyq 
	b, a = butter(order, normal_cutoff, btype='high', analog=False) 
	return b, a 

def butter_highpass_filter(data, cutoff, fs, order=5): 
	b, a = butter_highpass(cutoff, fs, order=order) 
	y = lfilter(b, a, data) 
	return y

def ProcessData(data):
	data = data[data.label != 0]
	data.loc[:,'m'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)

	order = 7
	fs = 52.0
	cutoff = 1.

	for c in ['x', 'y', 'z', 'm']:
		data.loc[:,c+'l'] = butter_lowpass_filter(data[c], cutoff, fs, order)
		data.loc[:,c+'h'] = butter_highpass_filter(data[c], cutoff, fs, order)
	return data

def Preprocessing(filename,axes=False,index=1):
	print '#############################'
	print filename
	data = ReadCSV(filename)
	if axes is not False and index == 1:
		plt.style.use('fivethirtyeight')
		ShowData(data)
		pass
	data = ProcessData(data)
	
	if axes is not False:
		PlotData(data,axes,index)
		pass
	if axes is not False  and index == 1:
		PlotData_after(data)
	return data

def rms(series):
	return np.sqrt((series**2).mean())

def min_max_mean(series):
	mins = argrelmin(series)[0]
	maxs = argrelmax(series)[0]
	min_max_sum = 0
	if len(mins) <= len(maxs):
		for j, arg in enumerate(mins):
			min_max_sum += series[maxs[j]] - series[arg]
	else:
		for j, arg in enumerate(maxs):
			min_max_sum += series[arg] - series[mins[j]]
	if min(len(mins), len(maxs)) == 0:
		return (0,0,0)
	return (min_max_sum/float(min(len(mins), len(maxs))),max(maxs)-min(mins),len(mins)+len(maxs))

def extract_features(data, y, window_len,task2=False):#num_windows):
	i = 0
	#window_len = len(data)/(num_windows/2)
	if task2:
		num_windows = len(data)-window_len+1
	else:
		num_windows = len(data)/(window_len/2)
	#print 'num_windows = 208, window_len = ' , str(len(data)/(208/2))
	#print 'now num_windows = '+ str(num_windows)+', window_len = '+str(window_len)
	features = []
	targets = []
	for n in range(num_windows):
		win = data[i:i+window_len]
		if task2:
			target = y.iloc[i]
		else:
			try:
				target = int(y[i:i+window_len].mode())
			except:
				target = int(y[i:i+window_len])
		targets.append(target)
		for c in data.columns:
			s = np.array(win[c])
			rms_val = rms(s)
			(min_max,peak,peaknum) = min_max_mean(s)
			mean = s.mean()
			std = s.std()
			skew = st.skew(s)
			kurtosis = st.kurtosis(s)
			coefficients = std/mean
			logpower = np.log10((s**2)).sum()
			new_features = [rms_val, min_max, mean, std, skew, kurtosis,peak,peaknum,coefficients,logpower]
			#new_features = [rms_val, min_max, mean, std]
			features.append(new_features)
		if (task2):
			i += 1
		else:
			i += window_len/2
	features = np.array(features)
	features.shape = num_windows, 120#48#72
	targets = np.array(targets)
	return features, targets

def Train(X_train,y_train):
	#classifiers = [KNeighborsClassifier()]
	classifiers = [LogisticRegression(C=1, penalty='l2'),KNeighborsClassifier(),RandomForestClassifier(n_estimators=20, class_weight='balanced'),AdaBoostClassifier(),GaussianNB(),DecisionTreeClassifier(max_depth=5)]
	for clf in classifiers:
		print 'fit##################'
		clf.fit(X_train, y_train)
	return classifiers

def Test1(classifiers,X_test,y_test):
	for clf in classifiers:
		print 'predict#############################'
		print clf
		y_pred = clf.predict(X_test)
		print pd.Series(y_test).mode()
		print pd.Series(y_pred).mode()
		print y_pred
		print accuracy_score(y_test,y_pred)
		print classification_report(y_test, y_pred)
		try:
			if int(pd.Series(y_pred).mode()) == int(pd.Series(y_test).mode()):
				return 1
		except:
			pass
	return 0

def Test2(classifiers,X_test,y_test,index):
	for clf in classifiers:
		print 'predict#############################'
		print clf
		y_pred = clf.predict(X_test)
		print index
		print y_pred
		print y_test
		print accuracy_score(y_test,y_pred)
		print classification_report(y_test, y_pred)
		point = np.where(y_test == index)[0]
		if len(point)!=0:
			point = point[len(point)-1]
		else:
			point = -1
		print point
		if len(np.where(y_pred == index)[0])!=0:
			ppred = np.where(y_pred == index)[0]
		else:
			ppred = [0]
		if len(ppred)!=0:
			ppred = ppred[len(ppred)-1]
		else:
			ppred = -1
		print ppred
		print np.where(y_test == index)
		print np.where(y_pred == index)
		if point == ppred:
			print 'point '+str(point)+' is the turn point!'
			return 1
	return 0

def Task1():
	features = []
	targets = []
	for i in xrange(1,13):
		trains = Preprocessing('data/'+str(i)+'.csv',False,i)
		for c in xrange(1,8):
			train = trains[trains['label']==c]
			feature = train.drop(['sample', 'label'], axis=1)
			target = train['label']
			feature, target = extract_features(feature, target, 520)
			if len(features) == 0:
				features = feature
				targets = target
			else :
				features = np.append(features,feature,axis=0)
				targets = np.append(targets,target)
	classifiers = Train(features,targets)
	
	for i in xrange(13,16):
		tests = Preprocessing('data/'+str(i)+'.csv',False,i)
		trues = 0
		for c in xrange(1,8):
			test = tests[tests['label']==c]
			feature = test.drop(['sample', 'label'], axis=1)
			target = test['label']
			feature, target = extract_features(feature, target, 600)
			trues += Test1(classifiers,feature,target)
		print 'result:' +str(trues/8.)
def Task2():
	trains = []
	tests = []
	for i in xrange(1,13):
		trains.append(Preprocessing('data/'+str(i)+'.csv',False,i))
	for i in xrange(13,16):
		tests.append(Preprocessing('data/'+str(i)+'.csv',False,i))
	for c in xrange(2,8):
		features = []
		targets = []
		trues = 0
		for i in xrange(0,12):
			point = trains[i][trains[i]['label']==c].index[0]
			train = trains[i].iloc[point-1040:point+2080,:]
			#train = trains[i].iloc[point-52:point+104,:]
			feature = train.drop(['sample', 'label'], axis=1)
			target = train['label']
			feature, target = extract_features(feature, target, 520,True)
			#feature, target = extract_features(feature, target, 52,True)
			print target
                	if len(features)==0:
                	       features = feature
                	       targets = target
                	else:
                	       features=np.append(features,feature,axis=0)
                	       targets =np.append(targets,target,axis=0)
		classifiers = Train(features,targets)
		for i in xrange(0,3):
			point = tests[i][tests[i]['label']==c].index[0]
			test = tests[i].iloc[point-1040:point+2080,:]
			#test = tests[i].iloc[point-52:point+104,:]
			feature = test.drop(['sample', 'label'], axis=1)
			target = test['label']
			feature, target = extract_features(feature, target, 520,True)
			trues +=Test2(classifiers,feature,target,targets[0])
		print 'result:' +str(trues/3.)
	
	'''
	for i in xrange(13,16):
		tests = Preprocessing('data/'+str(i)+'.csv',False,i)
		for c in xrange(2,7):
			_fig,_axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 9), squeeze=False)
			
                        point = tests[tests['label']==c].index[0]
			test = tests.iloc[point-1000:point+1000,:]
			features = test.drop(['sample', 'label'], axis=1)
			#features = test[['m']]
			#_axes[0,0].plot(features[:])
			targets = test['label']
			features, targets = extract_features(features, targets, 3)
			#_axes[1,0].plot(features[:])
			print '#####'
			Q_ifm, P_ifm, Pcp_ifm = offcd.offline_changepoint_detection(features,partial(offcd.const_prior, l=(len(features)+1)),offcd.ifm_obs_log_likelihood,truncate=-20)
			Q_full, P_full, Pcp_full = offcd.offline_changepoint_detection(features,partial(offcd.const_prior, l=(len(features)+1)),offcd.fullcov_obs_log_likelihood, truncate=-20)

			Q, P, Pcp = offcd.offline_changepoint_detection(features, partial(offcd.const_prior, l=(len(features)+1)), offcd.gaussian_obs_log_likelihood, truncate=-40)
			_axes[0,0].plot(np.exp(Pcp_ifm).sum(0))
			_axes[1,0].plot(np.exp(Pcp_full).sum(0))
			_axes[2,0].plot(np.exp(Pcp).sum(0))
			plt.show()
	'''
def Task3():
	features = []
	targets = []
        for i in xrange(1,13):
                trains = Preprocessing('data/'+str(i)+'.csv',False,i)
		feature = trains.drop(['sample', 'label'], axis=1)
		target = trains['label']
		feature, target = extract_features(feature, target, 3)
		if len(features) == 0:
			features = feature
			targets = target
		else :
			features = np.append(features,feature,axis=0)
			targets = np.append(targets,target)
	classifiers = Train(features,targets)
	
	for i in xrange(13,16):
		tests = Preprocessing('data/'+str(i)+'.csv',False,i)
		trues = 0
		feature = tests.drop(['sample', 'label'], axis=1)
		target = tests['label']
		feature, target = extract_features(feature, target, 3)
		trues += Test1(classifiers,feature,target,c)
		print 'result:' +str(trues/8.)
	

if __name__ == '__main__':
	paint = False
	if paint is not False:
		fig, paint = plt.subplots(nrows=3, ncols=4, figsize=(20, 9))
		for i in xrange(1,13):
			Preprocessing('data/'+str(i)+'.csv',paint,i)
		print 'painting ... This may take a while, please wait'
		plt.show()

	#Task1()
	Task2()	
	#Task3()	
