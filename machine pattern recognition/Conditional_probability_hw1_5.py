import numpy as np
def main():
    w1 = [[0.42, -0.2, 1.3, 0.39, -1.6, -0.029, -0.23, 0.27, -1.9, 0.87],
    [-0.087, -3.3, -0.32, 0.71, -5.3, 0.89, 1.9, -0.3, 0.76, -1.0],
          [0.58, -3.4, 1.7, 0.23, -0.15, -4.7, 2.2, -0.87, -2.1, -2.6]]
    w2 = [[-0.4, -0.31, 0.38, -0.15, -0.35, 0.17, -0.011, -0.27, -0.065, -0.12],
          [0.58, 0.27, 0.055, 0.53, 0.47, 0.69, 0.55, 0.61, 0.49, 0.054],
          [0.089, -0.04, -0.035, 0.011, 0.034, 0.1, -0.18, 0.12, 0.0012, -0.063]]
    w3 = [[0.83, 1.1, -0.44, 0.047, 0.28, -0.39, 0.34, -0.3, 1.1, 0.18],
          [1.6, 1.6, -0.41, -0.45, 0.35, -0.48, -0.079, -0.22, 1.2, -0.11],
          [-0.014, 0.48, 0.32, 1.4, 3.1, 0.11, 0.14, 2.2, -0.46, -0.49]]

    # 1. mean, variance of w1, x in w
    for col in range(3):
        meanw1x = sum(w1[col])/len(w1[col])
        varw1x = sum([(i-meanw1x)**2 for i in w1[col]])/len(w1[col])
        print('Part a, x{} mean is {}, variance is {}'.format(col+1, meanw1x, varw1x))
    print('\n')

    # 2. multivariate Gaussian
    for m in [[w1[0],w1[1]], [w1[0], w1[2]], [w1[1], w1[2]]]:
        print('Part b, μ is {}, var is {}'.format([np.mean(m, axis=1)], np.multiply(np.cov(m), 0.9) ))
    print('\n')
    # 3. multivariate Gaussian
    print('Part c, μ is {}, var is {}'.format([np.mean(w1, axis=1)], np.multiply(np.cov(w1),0.9)))
    print('\n')
    # 4. covariance = 0
    var4 = []
    for col in range(3):
        meanw2x = sum(w2[col])/len(w2[col])
        varw2x = sum([(i-meanw2x)**2 for i in w2[col]])/len(w2[col])
        var4.append(varw2x)
    print('Part d, μ is {}, diag(var) is {}'.format([np.mean(w2, axis=1)], var4))
    print('\n')


if __name__=="__main__":
    main()