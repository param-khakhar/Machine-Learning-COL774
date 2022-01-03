from q2b import SGD
from q2a import sample

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


def cost(X,Y,theta):
    htheta = np.matmul(X,theta)
    diff = Y-htheta
    diff = diff.reshape(diff.shape[0])
    res = np.sum(diff * diff)/2*diff.shape[0]
    return res

if __name__ == '__main__':

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Gettting the X's and Y's from the function implemented in the file q2a.py.
    X,Y = sample()

    # Concatenating the two arrays in a left/right fashion.
    combined = np.hstack((X,Y))

    # Shuffling the rows of the combined array, and separating the X's and Y's.
    np.random.shuffle(combined)

    X = combined[:,:-1]
    Y = combined[:,-1:]

    batches = [1,100,10000,1000000]
    # batches = [100]
    fig = plt.figure(figsize = (10,10))
    ax = Axes3D(fig)
    for batch in batches:
        model = SGD(X.shape[1],batch)

        params,Jthetalist,Thetalist = model.fit(X,Y)

        #print(params)
        #print(len(Thetalist[0]),len(Thetalist[1]),len(Thetalist[2]))

        # #Plot the contour

        #Surface plot. Selecting some points for plotting the graph using stride for efficiency, due to the uniform graph.
        #ax.plot_surface(T1, T2, Z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)

        ax.set_xlabel('theta 0')
        ax.set_xlim((-1,4))
        ax.set_ylabel('theta 1')
        ax.set_ylim((-1,4))
        ax.set_zlabel('theta 2')
        ax.set_zlim((-1,4))

        ax.view_init(20, -60)

        # Create animation, using matplotlib.animate.
        # line, = ax.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
        # point, = ax.plot([], [], [], '*', color = 'red')
        # display_value = ax.text(2., 2., 27.5, '')

        # # def init():
        # #     line.set_data([], [])
        # #     line.set_3d_properties([])
        # #     point.set_data([], [])
        # #     point.set_3d_properties([])
        # #     display_value.set_text('')

        # #     return line, point, display_value

        # # def animate(i):
        # #     # Animate line
        # #     line.set_data(Thetalist[0][:i*10:10], Thetalist[1][:i*10:10])
        # #     line.set_3d_properties(Thetalist[2][:i*10:10])
            
        # #     # Animate points
        # #     point.set_data(Thetalist[0][i*10], Thetalist[1][i*10])
        # #     point.set_3d_properties(Thetalist[2][i*10])

        #     # Animate value display
        #     #display_value.set_text('Theta = ' + str(Thetalist[0][i])+" "+str(Thetalist[1][i])+" "+str(Thetalist[2][i]))

        #     return line, point, display_value

        ax.plot(Thetalist[0],Thetalist[1],Thetalist[2],label = "Batch Size: "+str(batch))
    ax.set_title("Path to Convergence")
    ax.legend(loc = 1)
    #print(Thetalist[0][-1],Thetalist[1][-1],Thetalist[2][-1])
    # anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Thetalist[0])//10, interval=120, repeat_delay=60, blit=True)

    # mywriter = animation.FFMpegWriter(fps=60)
    # anim.save(output_dir+"q2d_"+str(batch)+".mp4",writer = mywriter)
    # #plt.show()
    fig.savefig(output_dir+"/q2d.png",dpi = 200)