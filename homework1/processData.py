import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def ReadCSV(filename):
	return pd.read_csv(filename, sep=', ', engine='python', header=None, names=['age','workclass','fnlwgt','education','edu_num','marital_statu','occupation','relationship','race','sex','cap_gain','cap_loss','hours','country','income'], index_col=False)

def showData(data):
	print data.info()
	#print data.head()
	print '#############################'
	print data.drop_duplicates(['workclass'])['workclass']
	print '#############################'
	print data.drop_duplicates(['education'])['education']
	print '#############################'
	print data.drop_duplicates(['marital_statu'])['marital_statu']
	print '#############################'
	print data.drop_duplicates(['occupation'])['occupation']
	print '#############################'
	print data.drop_duplicates(['relationship'])['relationship']
	print '#############################'
	print data.drop_duplicates(['race'])['race']
	print '#############################'
	print data.drop_duplicates(['sex'])['sex']
	print '#############################'
	print data.drop_duplicates(['country'])['country']
	print '#############################'
	print data.drop_duplicates(['income'])['income']
	print '#############################'
	pass

def cleanData(data):
	data.loc[ (data.workclass == '?'), 'workclass' ] = "other"
	data.loc[ (data.occupation == '?'), 'occupation' ] = "other"
	data.loc[ (data.country == '?'), 'country' ] = "other"
	return data

def processData(data):
	#i = 0
	#for _education in data.drop_duplicates(['education'])['education']:
	#	data.loc[(data.education == _education).'education']=i
	#	i += 1

	print '#############################'
	le = preprocessing.LabelEncoder() 
	le.fit(data.drop_duplicates(['education'])['education'])
	_education = pd.DataFrame(le.transform(data['education']),columns=['_education'])
	le.fit(data.drop_duplicates(['income'])['income'])#<=50k 0 ; >50k 1
	_income = pd.DataFrame(le.transform(data['income']),columns=['_income'])
	
	print '#############################'
	dummies_workclass = pd.get_dummies(data['workclass'], prefix = 'workclass')
	dummies_marital_statu = pd.get_dummies(data['marital_statu'], prefix = 'marital')
	dummies_occupation = pd.get_dummies(data['occupation'], prefix = 'occupation')
	dummies_relationship = pd.get_dummies(data['relationship'], prefix = 'relationship')
	dummies_race = pd.get_dummies(data['race'], prefix = 'race')
	dummies_sex = pd.get_dummies(data['sex'], prefix = 'sex')
	dummies_country = pd.get_dummies(data['country'], prefix = 'country')
	
	data = pd.concat([data,_education,_income,dummies_workclass,dummies_marital_statu,dummies_occupation,dummies_relationship,dummies_country,dummies_sex,dummies_race], axis=1)
	
	print '#############################'
	data.loc[ (data.age <= 20), 'age' ] = 1#"youth"
	data.loc[ (data.age > 60), 'age' ] = 4#"old"
	data.loc[ (data.age > 40) , 'age' ] = 3#"adultToOld"
	data.loc[ (data.age > 20), 'age' ] = 2#"adult"
	data['age'] = data['age'].astype('uint8')
	
	return data

def plotData(data):
	print '#############################'
	label_old = data._income[data.age==4].value_counts()
	label_adultToOld = data._income[data.age==3].value_counts()
	label_adult = data._income[data.age==2].value_counts()
	label_youth = data._income[data.age==1].value_counts()
	label_old = label_old.rename({0:'<=50k',1:'>50k'})
	label_adultToOld = label_adultToOld.rename({0:'<=50k',1:'>50k'})
	label_adult = label_adult.rename({0:'<=50k',1:'>50k'})
	label_youth = label_youth.rename({0:'<=50k',1:'>50k'})
	df=pd.DataFrame({'old':label_old,'adultToOld':label_adultToOld,'adult':label_adult, 'youth':label_youth}).transpose()
	df.plot(kind='bar', stacked=True)
	plt.title("age")
	plt.xlabel("age group")
	plt.ylabel("people numbers")

	print '#############################'
	fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 14))
	axes[0][0].violinplot([data[data._income==0]['edu_num'],data[data._income==1]['edu_num']],showmeans=False,showmedians=True)
	axes[0][0].set_title('edu_num')
	axes[0][1].violinplot([data[data._income==0]['fnlwgt'],data[data._income==1]['fnlwgt']],showmeans=False,showmedians=True)
	axes[0][1].set_title('fnlwgt')
	axes[1][0].violinplot([data[data._income==0]['cap_loss'],data[data._income==1]['cap_loss']],showmeans=False,showmedians=True)
	axes[1][0].set_title('cap_loss')
	axes[1][1].violinplot([data[data._income==0]['cap_gain'],data[data._income==1]['cap_gain']],showmeans=False,showmedians=True)
	axes[1][1].set_title('cap_gain')
	axes[2][0].violinplot([data[data._income==0]['hours'],data[data._income==1]['hours']],showmeans=False,showmedians=True)
	axes[2][0].set_title('hours')
	axes[2][1].violinplot([data[data._income==0]['age'],data[data._income==1]['age']],showmeans=False,showmedians=True)
	axes[2][1].set_title('age')

	print '#############################'
	label_nm = data._income[data.marital_statu=='Never-married'].value_counts()
	label_mcs = data._income[data.marital_statu=='Married-civ-spouse'].value_counts()
	label_d = data._income[data.marital_statu=='Divorced'].value_counts()
	label_msa = data._income[data.marital_statu=='Married-spouse-absent'].value_counts()
	label_s = data._income[data.marital_statu=='Separated'].value_counts()
	label_mas = data._income[data.marital_statu=='Married-AF-spouse'].value_counts()
	label_w = data._income[data.marital_statu=='Widowed'].value_counts()
	df=pd.DataFrame({'Never-married':label_nm,'Married-civ-spouse':label_mcs,'Divorced':label_d,'Married-spouse-absent':label_msa,'Separated':label_s,'Married-AF-spouse':label_mas,'Widowed':label_w}).transpose()
	df.plot(kind='bar', stacked=True)
	plt.title("marital_statu")
	plt.xlabel("marital_statu")
	plt.ylabel("people numbers")
	plt.show()
	pass

def Preprocessing(filename):
	data = ReadCSV(filename)
	#showData(data)
	data = cleanData(data)
	data = processData(data)
	#showData(data)
        #plotData(data)
	data.drop(['education','income','marital_statu','workclass','occupation','relationship','race','sex','country'], axis=1, inplace=True)
	return data

def Train(X_train,y_train):
	X_train = StandardScaler().fit_transform(X_train)
	classifiers = [KNeighborsClassifier(3),SVC(kernel="linear", C=0.025),SVC(gamma=2, C=1),GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),DecisionTreeClassifier(max_depth=5),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),MLPClassifier(alpha=1),AdaBoostClassifier(),GaussianNB(),QuadraticDiscriminantAnalysis()]
	for clf in classifiers:
		clf.fit(X_train, y_train)
		print clf
	return classifiers

def Test(X_train,y_train,X_test,y_test,classifiers):
	X_train = StandardScaler().fit_transform(X_train)
	X_test = StandardScaler().fit_transform(X_test)
	
	###################
	cm = plt.cm.RdBu
	cm_bright = ListedColormap(['#FF0000', '#0000FF'])
	X = np.concatenate((X_train,X_test))
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),np.arange(y_min, y_max, .02))
	
	fig,axes = plt.subplots(1, len(classifiers) + 1, figsize=(12, 14))
	axes[0].set_title("Input data")
	axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
	axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
	axes[0].set_xlim(xx.min(), xx.max())
	axes[0].set_ylim(yy.min(), yy.max())
	axes[0].set_xticks(())
	axes[0].set_yticks(())
	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process","Decision Tree", "Random Forest", "Neural Net", "AdaBoost","Naive Bayes", "QDA"]
	i = 1
	####################

	for name,clf in zip(names,classifiers):
		print 'predict#############################'
		y_pred = clf.predict(X_test)
		print clf
		print(classification_report(y_test, y_pred))

        	####################
        	score = clf.score(X_test, y_test)
        	if hasattr(clf, "decision_function"):
        		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        	else:
        		Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        	Z = Z.reshape(xx.shape)
    		axes[i].set_title(name)
		axes[i].contourf(xx, yy, Z, cmap=cm, alpha=.8)
        	axes[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        	axes[i].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        	axes[i].set_xlim(xx.min(), xx.max())
        	axes[i].set_ylim(yy.min(), yy.max())
        	axes[i].set_xticks(())
        	axes[i].set_yticks(())
        	axes[i].text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')
        	i += 1
        	####################
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	train = Preprocessing('data/adult.data.txt')
	features = 'age|cap_gain|cap_loss|country_*|edu_num|fnlwgt|hours|marital_*|occupation_*|race_*|relationship_*|sex_*|workclass_*'
	print 'train##################'
	classifiers = Train(train.filter(regex=features),train['_income'])
	test = Preprocessing('data/adult.test.txt')
	print 'test##################'
	test['country_Holand-Netherlands'] = 0
	Test(train.filter(regex=features),train['_income'],test.filter(regex=features),test['_income'],classifiers)
