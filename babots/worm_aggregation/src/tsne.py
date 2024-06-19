import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne(archive):
    vectors = [individual.vector for individual in archive]
    data = np.array([v.to_array() for v in vectors])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(data)

    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.title('t-SNE of Vector Data')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()
