import gaussian_random_fields as gr
import random
import numpy as np

def main(N, n_train, n_test):
    '''N is the grid number along edges'''
    dataset = {}
    train_data = np.zeros((n_train, N, N))
    test_data = np.zeros((n_test, N, N))
    ntr, nte = n_train/6, n_test/6
    # random data in whole field
    
    # random selected points for whole field
    # Gaussian random field
    # triangle function w randomness
    x = np.linspace(-1,1,N, dtype=np.float32) 
    y = np.linspace(-1,1,N, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    # polynomial function w randomness
    # jump function w/wo randomness
    ntr, nte = n_train-5*ntr, n_test-5*nte

    dataset['train'] = train_data
    dataset['test'] = test_data

def random_data(N):
    coef = np.random.randint(-10, 10, size=(2,))
    return coef[0]*np.random((N,N)) + coef[1]

def random_selected_points(N):
    num = N/2
    data = np.zeros((N,N))
    for _ in range(num):
        i = np.random.randint(N)
        j = np.random.randint(N)
        data[i][j] = np.random.randint(-10, 10)*random.random()
    return data

def Gaussian_random_field(N):
    alpha = random.uniform(2.0,5.0)
    return gr.gaussian_random_field(alpha=alpha, size=N)

def triangle_function(xx, yy):
    '''xx, yy are grid point coordinates'''
    coef = np.random.randint(-10, 10, size=(3,))
    return coef[0]*np.sin(coef[1]*np.pi*xx)*np.sin(coef[2]*np.pi*yy)

def polynomial_function(xx, yy):
    '''xx, yy are grid point coordinates'''
    coef = np.random.randint(-10, 10, size=(4,))
    return coef[0]*xx**2+coef[1]*yy**2+coef[2]*xx*yy+coef[3]

if __name__ == "__main__":
    N = 2**4+1 # number of elements along edges
    n_train = 1000 # number of training data
    n_test = 200 # number of testing data
    main(N, n_train, n_test)