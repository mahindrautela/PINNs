import lib.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from lib.pinn_string import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
import math
import time

def u0(tx):
    """
    Initial wave form.
 
    Args:
        tx: variables (t, x) as tf.Tensor.
        c: wave velocity.
        k: wave number.
        sd: standard deviation.

    Returns:
        u(t, x) as tf.Tensor.
    """

    #t = tx[..., 0, None]
    #x = tx[..., 1, None]
    z = -np.sin(1*math.pi*tx)
    return z

def du0_dt(tx):
    """
    First derivative of t for the initial wave form.

    Args:
        tx: variables (t, x) as tf.Tensor.

    Returns:
        du(t, x)/dt as tf.Tensor.
    """

    with tf.GradientTape() as g:
        g.watch(tx)
        u = u0(tx)
    du_dt = g.batch_jacobian(u, tx)[..., 0]
    return du_dt

# number of training samples
num_train_samples = 10000
    
# number of test samples
num_test_samples = 1000
    
# Analytical solution of wave equation
c = 1
L = 1
n = L
T = 1
xx = np.linspace(0,L,num_test_samples)
tt = np.linspace(0,T,num_test_samples)
usol = np.zeros((num_test_samples,num_test_samples))
for i,xi in enumerate(xx):
    for j,tj in enumerate(tt):
        usol[i,j] = -np.sin(math.pi*xi)*np.cos(n*math.pi*c*tj/L)

plt.plot(xx,usol[:,251])

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for the wave equation.
    """

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network,c).build() #c = 1 m/s

    # create 2 linspace columns
    vec1 = np.linspace(0,1,num_train_samples)
    vec2 = np.linspace(0,1,500)
    vec3 = np.linspace(0,1,500)
    mat1 = np.vstack([vec1,vec1]).T
    mat2 = np.vstack([vec2,vec2]).T
    mat3 = np.vstack([vec3,vec3]).T
    print('\nShape of linspace matrix ==>',mat2.shape)
    
    # create training input
    #tx_eqn = mat1
    tx_eqn = np.random.rand(num_train_samples, 2)
    tx_eqn[..., 0] = T*tx_eqn[..., 0]                      # t =  0 ~ +1
    tx_eqn[..., 1] = L*tx_eqn[..., 1]                      # x = 0 ~ +10
    print('\nShape of t_eqn ==>',tx_eqn.shape)
    
    #tx_ini = mat
    tx_ini = np.random.rand(num_train_samples, 2)
    tx_ini[..., 0] = 0                                     # t = 0
    tx_ini[..., 1] = L*tx_ini[..., 1]                      # x = 0 ~ +10
    print('\nShape of tx_ini ==>',tx_ini.shape)
    
    #tx_bnd = mat
    tx_bnd = np.random.rand(num_train_samples, 2)
    tx_bnd[..., 0] = T*tx_bnd[..., 0]                      # t =  0 ~ +1
    tx_bnd[..., 1] = L*np.round(tx_bnd[..., 1])            # x =  0 or +10
    print('\nShape of tx_bnd ==>',tx_bnd.shape)
    
    # create training output
    u_zero = np.zeros((num_train_samples, 1))
    #u_ini = u0(tf.constant(tx_ini)).numpy()
    #du_dt_ini = du0_dt(tf.constant(tx_ini)).numpy()
    u_ini = u0(tx_ini[:,1,None])
    du_dt_ini = np.zeros((num_train_samples, 1))
    
    # plot the initial conditions
    x_ini_sort = np.sort(tx_ini[:,1])
    u_ini0 = u0(x_ini_sort)
    fig = plt.figure(figsize=(7,4))
    plt.plot(x_ini_sort,u_ini0)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Initial condition on u')

    # train the model using L-BFGS-B algorithm
    begin = time.time()
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [u_zero, u_ini, du_dt_ini, u_zero]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()
    end = time.time()
    totaltime = end-begin
    print("Total runtime of the program is",totaltime)

    # predict u(t,x) distribution
    t_flat = np.linspace(0, T, num_test_samples)
    #print('\nShape of t_flat',t_flat.shape)
    x_flat = np.linspace(0, L, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 3) # A grid layout to place subplots within a figure.
    plt.subplot(gs[0, :])
    vmin, vmax = -1.0, +1.0
    plt.pcolormesh(t, x, u, cmap='rainbow', shading = 'auto', norm=Normalize(vmin=vmin, vmax=vmax))
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(vmin, vmax)
    
    # plot u(t=const, x) cross-sections
    tfrac = np.array([0.25,0.5,0.75])
    t_cross_sections = (T*tfrac).tolist()
    idx = [int(x) for x in (num_test_samples*tfrac)]
    
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        #print(t_cs)
        full = np.full(t_flat.shape, t_cs)
        #print(full.shape)
        #print(full)
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        #print(tx.shape)
        #print(tx)
        u = network.predict(tx, batch_size=num_test_samples)
        #print(u.shape)
        plt.plot(x_flat, u, 'b-', linewidth = 2)
        plt.plot(x_flat, usol[:,idx[i]], 'r--', linewidth = 2)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
        plt.ylim(-1,1)
        plt.legend(['Exact','Prediction'], loc = 'upper right',fontsize=4)
    plt.tight_layout()
    plt.savefig('result_img_dirichlet.png', transparent=True)
    plt.savefig('result_img_dirichlet.eps', transparent=True, format='eps',
                dpi = 1200)
    plt.show()
    
    # comparison plots
    nn = [251,501,751]
    mm = 2
    tx = np.stack([np.full(t_flat.shape, t_cross_sections[mm]), x_flat], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    
    fig2 = plt.figure(figsize=(7,4))
    plt.plot(x_flat,usol[:,nn[mm]], 'b-', linewidth = 2)       
    plt.plot(x_flat,u, 'r--', linewidth = 2)
    plt.xlabel('$x$')
    plt.ylabel('$u(x,t)$')    
    plt.title('$t = 0.25s$', fontsize = 10)
    #plt.set_xlim([-1.1,1.1])
    plt.legend(['Exact','Prediction'],loc = 'best')
    plt.ylim(-1.1,1.1)