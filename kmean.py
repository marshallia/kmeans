import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.arff as sc
import sklearn.cluster as sk
dt,meta=sc.loadarff('diabetes.arff')
data=pd.DataFrame(dt)
data1=data.drop('class',axis=1)
feature=len(data1.columns)
centroid=pd.DataFrame()
def kmeans():
	X = np.array(data1.iloc[0:768])
	kmeans =sk.KMeans(n_clusters=2,random_state=0).fit(X,y=None)
	data1['label']=pd.Series(kmeans.labels_)
	centroid=pd.DataFrame(kmeans.cluster_centers_,columns=['preg','plas','pres','skin','insu','mass','pedi','age'])
	print(centroid)
	print(data1)
	print(kmeans.inertia_)
	return	centroid
centroid=kmeans()
data1.set_index('label',inplace=True)

plt.figure(1)
plt.plot(data['insu'],data['age'],'bo',)
plt.xlabel('age')
plt.ylabel('insu')
plt.figure(2)
plt.plot(data1.loc[[0],['insu']],data1.loc[[0],['age']],'bo',data1.loc[[1],['insu']],data1.loc[[1],['age']],'ro',centroid.loc[[0],['insu']],centroid.loc[[0],['age']],'x',centroid.loc[[1],['insu']],centroid.loc[[1],['age']],'x')
plt.xlabel('age')
plt.ylabel('insu')

plt.show()
