from cvae_sampler import *
from util import *
import numpy as np
import signal
import sys

if __name__ == '__main__' :
    dataset = Dataset()
    cvae = CVAESampler(epoch=2000000, mb_size=128)
    dataset.load('solutions.txt')
    signal.signal(signal.SIGINT, cvae.signal_handler) 
    if dataset.mode == 'load_all' :
        data = dataset.get_data(0.8)
        train_data = data['train']
        cvae.train(train_data['samples'], train_data['conditions'])
    else :
        cvae.train_from_str(dataset.str, dataset.n_data)
    cvae.save()
    # xs, xg, obs = np.random.rand(4), np.random.rand(4), np.random.randn(9, 2vx_scale, vx_shift = 2*1.5, 1.5/2)
    x_scale, x_shift = WIDTH/100.0, WIDTH/200.0
    y_scale, y_shift = HEIGHT/100.0, HEIGHT/200.0
    v_scale, v_shift = 2*1.5, 1.5/2
    rg = np.random.random
    xs = np.reshape([rg(1)*x_scale-x_shift, rg(1)*y_scale-y_shift, rg(1)*v_scale-v_shift, rg(1)*v_scale-v_shift],(4))
    xg = np.reshape([rg(1)*x_scale-x_shift, rg(1)*y_scale-y_shift, rg(1)*v_scale-v_shift, rg(1)*v_scale-v_shift],(4))
    obs = np.reshape([[rg(1)*x_scale-x_shift,rg(1)*y_scale-y_shift] for _ in range(9)],(9,2))
    y, z = cvae.sample(30, xs, xg, obs)
    print 'conditional variable :'
    print 'xs :', xs, 'xg :', xg, 'obs :', obs
    print 'samples :', y, '\nlatent var :', z
    print 'done'
