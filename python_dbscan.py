import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def region_query(data, point_id, eps, distance_func):
    """
    Find all points within a distance of eps from a point.
    :param data: The data points to be clustered.
    :param point_id: The ID of the point to find neighbors for.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :return: A list of point IDs.
    """
    neighbors = []
    for i in range(len(data)):
        if distance_func(data[point_id], data[i]) <= eps:
            neighbors.append(i)
    return neighbors


def dbscan(eps, min_samples, data, distance_func):
    """
    DBSCAN algorithm for clustering data points.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :param data: The data points to be clustered.
    :param distance_func: The distance function to use.
    :return: A list of cluster labels for each data point.
    """

    label = [-2] * len(data)  # -2: unvisited, -1: noise, 0, 1, 2, ...: cluster labels
    cluster_id = 0
    # For each point in the dataset ...
    for point_id in range(len(data)):
        # If the point was already visited, move on to the next point.
        if label[point_id] > -2:
            continue
        # Find all points in the neighborhood of the current point.
        neighbors = region_query(data, point_id, eps, distance_func)
        # If there are not enough neighbors to form a cluster ...
        if len(neighbors) < min_samples:
            # Label the point as noise.
            label[point_id] = -1
        # Otherwise, a new cluster is found!
        else:
            queue = [point_id]
            label[point_id] = cluster_id

            while queue:
                current_point = queue.pop(0)
                neighbors = region_query(data, current_point, eps, distance_func)

                # If there are enough neighbors, add them to the queue.
                if len(neighbors) >= min_samples:
                    unvisited_neighbors = [n for n in neighbors if label[n] == -2]
                    for neighbor in unvisited_neighbors:
                        label[neighbor] = cluster_id
                    queue += unvisited_neighbors

            cluster_id += 1
    return label


if __name__ == '__main__':
    df = pd.read_csv("https://reneshbedre.github.io/assets/posts/tsne/tsne_scores.csv")


    def distance_func(x, y):
        ndim = len(x)
        return sum((x[i] - y[i]) ** 2 for i in range(ndim)) ** 0.5


    clusters_labels = dbscan(eps=2.5, min_samples=4, data=df.values, distance_func=distance_func)

    p = sns.scatterplot(data=df, x="t-SNE-1", y="t-SNE-2", hue=clusters_labels, legend="full", palette="deep")
    sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')
    plt.savefig("python_dbscan.png")
    plt.show()
