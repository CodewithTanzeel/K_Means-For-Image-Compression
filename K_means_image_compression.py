import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

def read_image():
    img = cv2.imread('flopy.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def initialize_means(img, clusters):
    points = img.reshape((-1, img.shape[2]))
    m, n = points.shape
    means = np.zeros((clusters, n))
    for i in range(clusters):
        rand_indices = np.random.choice(m, size=10, replace=False)
        means[i] = np.mean(points[rand_indices], axis=0)
    return points, means

def k_means(points, means, clusters):
    iterations = 10
    m, n = points.shape
    index = np.zeros(m, dtype=int)

    for _ in range(iterations):
        diff = points[:, np.newaxis, :] - means[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        index = np.argmin(dists, axis=1)

        for k in range(clusters):
            cluster_points = points[index == k]
            if len(cluster_points) > 0:
                means[k] = np.mean(cluster_points, axis=0)

    return means, index

def compress_image(means, index, img, clusters):
    centroid = np.array(means)
    recovered = centroid[index.astype(int), :]
    recovered = recovered.reshape(img.shape)
    recovered_uint8 = (recovered * 255).astype(np.uint8)
    recovered_bgr = cv2.cvtColor(recovered_uint8, cv2.COLOR_RGB2BGR)

    plt.imshow(recovered)
    plt.show()

    cv2.imwrite('compressed_' + str(clusters) + '_colors.png', recovered_bgr)

if __name__ == '__main__':
    img = read_image()
    clusters = 16
    print(f"Running K-means with {clusters} colors...")
    points, means = initialize_means(img, clusters)
    means, index = k_means(points, means, clusters)
    compress_image(means, index, img, clusters)
    print("Done.")