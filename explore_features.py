



# type(prediction)


fig = px.line(data_frame=pd.DataFrame(X.T))
fig.show()






n_neighbors = 10
n_dims_red = 2

# umap 
reducer = umap.UMAP(
    n_neighbors = n_neighbors, 
    n_components = n_dims_red, 
    metric = 'euclidean',
    n_jobs = -1
    )

# reducer.fit(X[0:25000])
# X_trans = reducer.transform(X)
X_trans = reducer.fit_transform(X)

X_trans.shape

# standardize
scaler = StandardScaler()
scaler.fit(X_trans)
X_scaled = scaler.transform(X_trans)


X_scaled.shape



fig = px.scatter(
    x =X_scaled[:,0],
    y =X_scaled[:,1]
    )


fig.show()





