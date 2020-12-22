from sklearn.decomposition import PCA           #加载PCA算法包


y=data.target
x=data.data
pca=PCA(n_components=2)     #加载PCA算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(x)#对样本进行降维