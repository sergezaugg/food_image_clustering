



import numpy as np
import umap.umap_ as umap






X = np.random.uniform(size = [500,20])
X.shape

reducer = umap.UMAP(
    n_neighbors = 5, 
    n_components = 5, 
    metric = 'euclidean',
    n_jobs = -1
    )

# X2D_trans_1 = reducer.fit_transform(X)
# X2D_trans_1.shape





reducer.fit(X, ensure_all_finite=True)
X2D_trans_2 = reducer.transform(X)
X2D_trans_2.shape

