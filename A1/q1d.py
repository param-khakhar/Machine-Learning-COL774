from q1a import LinearRegression

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Function Cost for the features X, targets Y, and the parameters theta.
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

    def init():
        line.set_data([], [])
        point.set_data([], [])
        value_display.set_text('')
        return line, point, value_display

    def animate(i):
        # Animate line
        line.set_data(Thetalist[0][:i], Thetalist[1][:i])
        
        # Animate points
        point.set_data(Thetalist[0][i], Thetalist[1][i])
        value_display.set_text("Error: "+str(Jthetalist[i]))

        return line, point, value_display

    #Setup of meshgrid of theta values. A mesh grid is a 2D grid of values.
    Xs, Ys = np.meshgrid(np.linspace(-1,2,100),np.linspace(-1,1,100))

    #Computing the cost function for all the points in the 2D plane as parameters.
    Zs = np.array([cost(newX,Y.values,np.array([[x],[y]])) for x, y in zip(Xs.reshape(-1), Ys.reshape(-1)) ] )

    #Reshaping the cost values    
    Zs = Zs.reshape(Xs.shape)

    #Plot the contour
    fig, ax = plt.subplots()
    ax.contour(Xs, Ys, Zs, 100, cmap = 'jet')

    ax.set_title("2D Contour Animation for Gradient Descent")
    ax.set_xlabel("Parameter - 1")
    ax.set_ylabel("Parameter - 2")

    # Create animation, using matplotlib animate.
    line, = ax.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
    point, = ax.plot([], [], '*', color = 'red', markersize = 4)
    value_display = ax.text(0.02, 0.02, '', transform=ax.transAxes)

    ax.legend(loc = 1)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Thetalist[0]), interval=100, repeat_delay=60, blit=True)
    ax.plot(Thetalist[0],Thetalist[1])
    fig.savefig(output_dir+"/q1d.png",dpi = 200)
    mywriter = animation.FFMpegWriter(fps=60)
    anim.save(output_dir+"/q1d.mp4",writer = mywriter)
    #plt.show()

