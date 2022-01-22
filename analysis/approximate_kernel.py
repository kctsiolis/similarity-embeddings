#Given a matrix W, turn it into a kernel via W^T * W
#Learn a new matrix V such that ||V^T * V - W^T * W||^2 is small

import argparse
import numpy as np
from analysis.compare_kernels import load_matrix, compute_kernel, compute_distance, l2_normalization

def get_args(parser):
    parser.add_argument('-w', dest='w', help='.npy file containing matrix.')
    parser.add_argument('--normalize', dest='normalize', choices=['true', 'false'], default='false',
        help='Whether or not to L2 normalize the rows of the matrices.')
    parser.add_argument('--steps', dest='steps', type=int, default=1000,
        help='Maximum number of gradient descent steps')
    parser.add_argument('--tol', dest='tol', type=float, default=1e-6,
        help='Gradient norm tolerance')
    parser.add_argument('--lr', dest='lr', type=float, default=0.1,
        help='Learning rate')

    return parser

def initialize_v(size):
    return np.random.normal(size=size)

def compute_gradient(v, v_ker, w_ker):
    return 4 * np.matmul(v_ker-w_ker,v)

def approximate_kernel(args):
    w_file = args.w
    steps = args.steps
    tol = args.tol
    lr = args.lr
    normalize = args.normalize

    w = load_matrix(w_file)

    if normalize == 'true':
        w = l2_normalization(w)

    v = initialize_v(w.shape)
    w_ker = compute_kernel(w)

    for i in range(steps):
        v_ker = compute_kernel(v)

        #Loss computation
        loss = (compute_distance(v_ker,w_ker,'frobenius'))**2
        mean_dist = compute_distance(v_ker,w_ker,'mean')
        print("Epoch {} Loss: {:.4f}".format(i, loss))
        print("Epoch {} Mean Distance: {:.4f}".format(i, mean_dist))
        
        #Gradient computation
        grad = compute_gradient(v,v_ker,w_ker)
        grad_norm = np.linalg.norm(grad)
        print("Epoch {} Gradient Norm: {:.4f}".format(i, grad_norm))
        v = v - lr*grad

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()
    approximate_kernel(args)