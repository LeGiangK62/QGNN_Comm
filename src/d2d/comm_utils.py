import numpy as np

def generate_wGaussian(K, num_H, var_noise=1, Pmax=1, seed=2017):
    # H[:,j,k] channel from k tx to j rx
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pini = Pmax*np.ones((num_H,K,1) )
    alpha = np.random.rand(num_H,K)
    # alpha = np.ones((num_H,K))
    #alpha = np.ones((num_H,K))
    fake_a = np.ones((num_H,K))
    #var_noise = 1
    X=np.zeros((K**2,num_H))
    Y=np.zeros((K,num_H))
    total_time = 0.0
    CH = 1/np.sqrt(2)*(np.random.randn(num_H,K,K)+1j*np.random.randn(num_H,K,K))
    H=abs(CH)
    Y = batch_WMMSE2(Pini,alpha,H,Pmax,var_noise)
    Y2 = batch_WMMSE2(Pini,fake_a,H,Pmax,var_noise)
    return H, Y, alpha, Y2

def batch_WMMSE2(p_int, alpha, H, Pmax, var_noise):
    N = p_int.shape[0]
    K = p_int.shape[1]
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros((N,K,1) )
    w = np.zeros( (N,K,1) )


    mask = np.eye(K)
    rx_power = np.multiply(H, b)
    rx_power_s = np.square(rx_power)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)

    interference = np.sum(rx_power_s, 2) + var_noise
    f = np.divide(valid_rx_power,interference)
    w = 1/(1-np.multiply(f,valid_rx_power))
    #vnew = np.sum(np.log2(w),1)


    for ii in range(100):
        fp = np.expand_dims(f,1)
        rx_power = np.multiply(H.transpose(0,2,1), fp)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        bup = np.multiply(alpha,np.multiply(w,valid_rx_power))
        rx_power_s = np.square(rx_power)
        wp = np.expand_dims(w,1)
        alphap = np.expand_dims(alpha,1)
        bdown = np.sum(np.multiply(alphap,np.multiply(rx_power_s,wp)),2)
        btmp = bup/bdown
        b = np.minimum(btmp, np.ones((N,K) )*np.sqrt(Pmax)) + np.maximum(btmp, np.zeros((N,K) )) - btmp

        bp = np.expand_dims(b,1)
        rx_power = np.multiply(H, bp)
        rx_power_s = np.square(rx_power)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        interference = np.sum(rx_power_s, 2) + var_noise
        f = np.divide(valid_rx_power,interference)
        w = 1/(1-np.multiply(f,valid_rx_power))
    p_opt = np.square(b)
    return p_opt


def np_sum_rate(H,p,alpha,var_noise):
    H = np.expand_dims(H,axis=-1)
    K = H.shape[1]
    N = H.shape[-1]
    p = p.reshape((-1,K,1,N))
    rx_power = np.multiply(H, p)
    rx_power = np.sum(rx_power,axis=-1)
    rx_power = np.square(abs(rx_power))
    mask = np.eye(K)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
    interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + var_noise
    rate = np.log(1 + np.divide(valid_rx_power, interference))
    w_rate = np.multiply(alpha,rate)
    sum_rate = np.mean(np.sum(w_rate, axis=1))
    return sum_rate


def simple_greedy(X,AAA,label):

    n = X.shape[0]
    thd = int(np.sum(label)/n)
    Y = np.zeros((n,K))
    for ii in range(n):
        alpha = AAA[ii,:]
        H_diag = alpha * np.square(np.diag(X[ii,:,:]))
        xx = np.argsort(H_diag)[::-1]
        for jj in range(thd):
            Y[ii,xx[jj]] = 1
    return Y

def np_sum_rate_all(H,p,alpha,var_noise):
    H = np.expand_dims(H,axis=-1)
    K = H.shape[1]
    N = H.shape[-1]
    p = p.reshape((-1,K,1,N))
    rx_power = np.multiply(H, p)
    rx_power = np.sum(rx_power,axis=-1)
    rx_power = np.square(abs(rx_power))
    mask = np.eye(K)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
    interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + var_noise
    rate = np.log(1 + np.divide(valid_rx_power, interference))
    w_rate = np.multiply(alpha,rate)
    sum_rate = np.sum(w_rate, axis=1)
    return sum_rate

def get_directLink_channel_losses(channel_losses):
    return np.diagonal(channel_losses, axis1=1, axis2=2)  # layouts X N

def get_crossLink_channel_losses(channel_losses):
    N = np.shape(channel_losses)[-1]
    return channel_losses * ((np.identity(N) < 1).astype(float))

def batch_fp(weights, g, var_noise, input_x):
    number_of_samples, N, _ = np.shape(g)
    assert np.shape(g)==(number_of_samples, N, N)
    assert np.shape(weights)==(number_of_samples, N)
    g_diag = get_directLink_channel_losses(g)
    g_nondiag = get_crossLink_channel_losses(g)
    # For matrix multiplication and dimension matching requirement, reshape into column vectors
    weights = np.expand_dims(weights, axis=-1)
    g_diag = np.expand_dims(g_diag, axis=-1)
    x = input_x#np.ones([number_of_samples, N, 1])
    #tx_power = general_para.tx_power
    #output_noise_power = general_para.output_noise_power
    # tx_powers = np.ones([number_of_samples, N, 1]) * tx_power  # assume same power for each transmitter
    # Run 100 iterations of FP computations
    # In the computation below, every step's output is with shape: number of samples X N X 1
    for i in range(100):
        # Compute z
        p_x_prod = x 
        z_denominator = np.matmul(g_nondiag, p_x_prod) + var_noise
        z_numerator = g_diag * p_x_prod
        z = z_numerator / z_denominator
        # compute y
        y_denominator = np.matmul(g, p_x_prod) + var_noise
        y_numerator = np.sqrt(z_numerator * weights * (z + 1))
        y = y_numerator / y_denominator
        # compute x
        x_denominator = np.matmul(np.transpose(g, (0,2,1)), np.power(y, 2))
        x_numerator = y * np.sqrt(weights * (z + 1) * g_diag)
        x_new = np.power(x_numerator / x_denominator, 2)
        x_new[x_new > 1] = 1  # thresholding at upperbound 1
        x = x_new
    assert np.shape(x)==(number_of_samples, N, 1)
    x_final = np.squeeze(x, axis=-1)
    return x_final

