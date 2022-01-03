from q1a import LinearRegression

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
    res = diff.dot(diff)/(2*diff.shape[0])
    
    return res

if __name__ == '__main__':
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    X = pd.read_csv(data_dir+"/linearX.csv")
    Y = pd.read_csv(data_dir+"/linearY.csv")

    X = (X - X.mean())/X.std()

    newX = np.ones([X.shape[0],X.shape[1]+1])
    newX[:,1:] = newX[:,1:]*X   
    model = LinearRegression(newX.shape[1],0.025)

    params,Jthetalist,Thetalist = model.fit(newX,Y.values)

    #print(params,len(Jthetalist),len(Thetalist))

    #Setup of meshgrid of theta values. A mesh grid is a 2D grid of values
    Xs, Ys = np.meshgrid(np.linspace(-1,2,150),np.linspace(-1,1,100))

    #Computing the cost function for all the points in the 2D plane as parameters.
    Zs = np.array(  [cost(newX,Y.values,np.array([[x],[y]])) for x, y in zip(Xs.reshape(-1), Ys.reshape(-1))])

    #Reshaping the cost values in the dimension (150 x 100)    
    Zs = Zs.reshape(Xs.shape)
    # #Plot the contour
    fig = plt.figure(figsize = (7,7))
    ax = Axes3D(fig)

    #Surface plot. Selecting some points for plotting the graph using stride for efficiency, due to the uniform graph.
    ax.plot_surface(Xs, Ys, Zs, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)

    ax.set_xlabel('theta 1')
    ax.set_ylabel('theta 2')
    ax.set_zlabel('error')

    ax.view_init(30, -75)

    # Create animation, using matplotlib.animate.
    line, = ax.plot([], [], [], 'r-', label = 'Gradient Descent', lw = 1.5)
    point, = ax.plot([], [], [], '*', color = 'red')
    display_value = ax.text(-1, -1, 0, '')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        display_value.set_text('')

        return line, point, display_value

    def animate(i):
        # Animate line
        line.set_data(Thetalist[0][:i], Thetalist[1][:i])
        line.set_3d_properties(Jthetalist[:i])
        
        # Animate points
        point.set_data(Thetalist[0][i], Thetalist[1][i])
        point.set_3d_properties(Jthetalist[i])

        display_value.set_text("Error: "+str(Jthetalist[i]))

        return line, point, display_value

    ax.legend(loc = 1)
    ax.plot(Thetalist[0],Thetalist[1],Jthetalist)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Thetalist[0]), interval=120, repeat_delay=60, blit=True)
    fig.savefig(output_dir+"/q1c.png",dpi = 200)
    mywriter = animation.FFMpegWriter(fps=60)
    anim.save(output_dir+"/q1c.mp4",writer = mywriter)
    #plt.show()