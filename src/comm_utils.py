import numpy as numpy

def generate_wGaussian(K, num_H, var_noise=1, Pmin=0, seed=2017):
    # H[:,j,k] channel from k tx to j rx
    print('Generate Data ... (seed = %d)' % seed)
    numpy.random.seed(seed)
    Pmax = 1
    Pini = Pmax*numpy.ones((num_H,K,1) )
    alpha = numpy.random.rand(num_H,K)
    # alpha = numpy.ones((num_H,K))
    #alpha = numpy.ones((num_H,K))
    fake_a = numpy.ones((num_H,K))
    #var_noise = 1
    X=numpy.zeros((K**2,num_H))
    Y=numpy.zeros((K,num_H))
    total_time = 0.0
    CH = 1/numpy.sqrt(2)*(numpy.random.randn(num_H,K,K)+1j*numpy.random.randn(num_H,K,K))
    H=abs(CH)
    Y = batch_WMMSE2(Pini,alpha,H,Pmax,var_noise)
    Y2 = batch_WMMSE2(Pini,fake_a,H,Pmax,var_noise)
    return H, Y, alpha, Y2

def batch_WMMSE2(p_int, alpha, H, Pmax, var_noise):
    N = p_int.shape[0]
    K = p_int.shape[1]
    vnew = 0
    b = numpy.sqrt(p_int)
    f = numpy.zeros((N,K,1) )
    w = numpy.zeros( (N,K,1) )


    mask = numpy.eye(K)
    rx_power = numpy.multiply(H, b)
    rx_power_s = numpy.square(rx_power)
    valid_rx_power = numpy.sum(numpy.multiply(rx_power, mask), 1)

    interference = numpy.sum(rx_power_s, 2) + var_noise
    f = numpy.divide(valid_rx_power,interference)
    w = 1/(1-numpy.multiply(f,valid_rx_power))
    #vnew = numpy.sum(numpy.log2(w),1)


    for ii in range(100):
        fp = numpy.expand_dims(f,1)
        rx_power = numpy.multiply(H.transpose(0,2,1), fp)
        valid_rx_power = numpy.sum(numpy.multiply(rx_power, mask), 1)
        bup = numpy.multiply(alpha,numpy.multiply(w,valid_rx_power))
        rx_power_s = numpy.square(rx_power)
        wp = numpy.expand_dims(w,1)
        alphap = numpy.expand_dims(alpha,1)
        bdown = numpy.sum(numpy.multiply(alphap,numpy.multiply(rx_power_s,wp)),2)
        btmp = bup/bdown
        b = numpy.minimum(btmp, numpy.ones((N,K) )*numpy.sqrt(Pmax)) + numpy.maximum(btmp, numpy.zeros((N,K) )) - btmp

        bp = numpy.expand_dims(b,1)
        rx_power = numpy.multiply(H, bp)
        rx_power_s = numpy.square(rx_power)
        valid_rx_power = numpy.sum(numpy.multiply(rx_power, mask), 1)
        interference = numpy.sum(rx_power_s, 2) + var_noise
        f = numpy.divide(valid_rx_power,interference)
        w = 1/(1-numpy.multiply(f,valid_rx_power))
    p_opt = numpy.square(b)
    return p_opt


def np_sum_rate(H,p,alpha,var_noise):
    H = numpy.expand_dims(H,axis=-1)
    K = H.shape[1]
    N = H.shape[-1]
    p = p.reshape((-1,K,1,N))
    rx_power = numpy.multiply(H, p)
    rx_power = numpy.sum(rx_power,axis=-1)
    rx_power = numpy.square(abs(rx_power))
    mask = numpy.eye(K)
    valid_rx_power = numpy.sum(numpy.multiply(rx_power, mask), axis=1)
    interference = numpy.sum(numpy.multiply(rx_power, 1 - mask), axis=1) + var_noise
    rate = numpy.log(1 + numpy.divide(valid_rx_power, interference))
    w_rate = numpy.multiply(alpha,rate)
    sum_rate = numpy.mean(numpy.sum(w_rate, axis=1))
    return sum_rate


def simple_greedy(X,AAA,label):

    n = X.shape[0]
    thd = int(numpy.sum(label)/n)
    Y = numpy.zeros((n,K))
    for ii in range(n):
        alpha = AAA[ii,:]
        H_diag = alpha * numpy.square(numpy.diag(X[ii,:,:]))
        xx = numpy.argsort(H_diag)[::-1]
        for jj in range(thd):
            Y[ii,xx[jj]] = 1
    return Y

def np_sum_rate_all(H,p,alpha,var_noise):
    H = numpy.expand_dims(H,axis=-1)
    K = H.shape[1]
    N = H.shape[-1]
    p = p.reshape((-1,K,1,N))
    rx_power = numpy.multiply(H, p)
    rx_power = numpy.sum(rx_power,axis=-1)
    rx_power = numpy.square(abs(rx_power))
    mask = numpy.eye(K)
    valid_rx_power = numpy.sum(numpy.multiply(rx_power, mask), axis=1)
    interference = numpy.sum(numpy.multiply(rx_power, 1 - mask), axis=1) + var_noise
    rate = numpy.log(1 + numpy.divide(valid_rx_power, interference))
    w_rate = numpy.multiply(alpha,rate)
    sum_rate = numpy.sum(w_rate, axis=1)
    return sum_rate