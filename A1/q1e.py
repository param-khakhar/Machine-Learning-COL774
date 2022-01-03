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
    res = np.sum(diff * diff)/2*diff.shape[0]
    return res

if __name__ == '__main__':
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    X = pd.read_csv(data_dir+"/linearX.csv")
    Y = pd.read_csv(data_dir+"/linearY.csv")

    X = (X - X.mean())/X.std()

    newX = np.ones([X.shape[0],X.shape[1]+1])
    newX[:,1:] = newX[:,1:]*X
    lrs = [0.001,0.025,0.1]
    data = []
    for lr in lrs:
        model = LinearRegression(newX.shape[1],lr)
        params,Jthetalist,Thetalist = model.fit(newX,Y.values)
        data.append([params,Jthetalist,Thetalist])

        # def init():
        #     line.set_data([], [])
        #     point.set_data([], [])
        #     value_display.set_text('')
        #     return line, point, value_display

        # def animate(i):
        #     # Animate line
        #     line.set_data(Thetalist[0][:i], Thetalist[1][:i])
            
        #     # Animate points
        #     point.set_data(Thetalist[0][i], Thetalist[1][i])

        #     return line, point, value_display

        # #Setup of meshgrid of theta values. A mesh grid is a 2D grid of values.
        # Xs, Ys = np.meshgrid(np.linspace(-1,2,100),np.linspace(-1,1,100))

        # #Computing the cost function for all the points in the 2D plane as parameters.
        # Zs = np.array([cost(newX,Y.values,np.array([[x],[y]])) for x, y in zip(Xs.reshape(-1), Ys.reshape(-1)) ] )

        # #Reshaping the cost values    
        # Z = Zs.reshape(Xs.shape)

        # #Plot the contour
        # fig1, ax1 = plt.subplots()
        # ax1.contour(Xs, Ys, Z, 100, cmap = 'jet')

        # ax1.set_title("2D Contour Animation for Gradient Descent")
        # ax1.set_xlabel("Parameter - 1")
        # ax1.set_ylabel("Parameter - 2")

        # # Create animation, using matplotlib animate.
        # line, = ax1.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
        # point, = ax1.plot([], [], '*', color = 'red', markersize = 4)
        # value_display = ax1.text(0.02, 0.02, '', transform=ax1.transAxes)

        # ax1.legend(loc = 1)

        # anim1 = animation.FuncAnimation(fig1, animate, init_func=init, frames=len(Thetalist[0]), interval=100, repeat_delay=60, blit=True)

        # mywriter = animation.FFMpegWriter(fps=60)
        # anim1.save(output_dir+"q1e_"+str(lr)+".mp4",writer = mywriter)
        #plt.show()
    #print(len(data))
    n1 = len(data[0][2][0])

    #print(len(data[0][2][0]),len(data[1][2][0]),len(data[2][2][0]))

    n2 = len(data[1][2][0])
    n3 = len(data[2][2][0])
    for i in range(1000-n3):
        data[2][2][0].append(data[2][2][0][-1])
        data[2][2][1].append(data[2][2][1][-1])

    for i in range(1000-n2):
        data[1][2][0].append(data[1][2][0][-1])
        data[1][2][1].append(data[1][2][1][-1])

    #print("Complete")
    #print(len(data[0][2][0]),len(data[1][2][0]),len(data[2][2][0]))

    # while len(data[1][2][0]) < 1000:
    #     data[1][2].append(data[1][2][-1])

    # while len(data[2][2][0]) < 1000:
    #     data[2][2].append(data[2][2][-1])

    def init():
        line.set_data([], [])
        point1.set_data([], [])
        point2.set_data([],[])
        point3.set_data([],[])
        value_display.set_text('')
        return line, point1, point2, point3, value_display

    def animate(i):
        # Animate line
        line.set_data(Thetalist[0][:i], Thetalist[1][:i])
        
        # Animate points
        point1.set_data(data[0][2][0][i], data[0][2][1][i])
        point2.set_data(data[1][2][0][i], data[1][2][1][i])
        point3.set_data(data[2][2][0][i], data[2][2][1][i])

        return line, point1, point2, point3, value_display

    #Setup of meshgrid of theta values. A mesh grid is a 2D grid of values.
    Xs, Ys = np.meshgrid(np.linspace(-1,2,100),np.linspace(-1,1,100))

    #Computing the cost function for all the points in the 2D plane as parameters.
    Zs = np.array([cost(newX,Y.values,np.array([[x],[y]])) for x, y in zip(Xs.reshape(-1), Ys.reshape(-1)) ] )

    #Reshaping the cost values    
    Z = Zs.reshape(Xs.shape)

    #Plot the contour
    fig1, ax1 = plt.subplots()
    ax1.contour(Xs, Ys, Z, 100, cmap = 'jet')

    ax1.set_title("2D Contour Animation for Gradient Descent")
    ax1.set_xlabel("Parameter - 1")
    ax1.set_ylabel("Parameter - 2")

    # Create animation, using matplotlib animate.
    line, = ax1.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
    point1, = ax1.plot([], [], '*', color = 'green', label = '0.001', markersize = 4)
    point2, = ax1.plot([], [], '*', color = 'yellow', label = '0.025', markersize = 4)
    point3, = ax1.plot([], [], '*', color = 'blue', label = '0.1',markersize = 4)
    value_display = ax1.text(0.02, 0.02, '', transform=ax1.transAxes)

    ax1.legend(loc = 1)

    anim1 = animation.FuncAnimation(fig1, animate, init_func=init, frames=len(data[2][2][0]), interval=100, repeat_delay=60, blit=True)

    mywriter = animation.FFMpegWriter(fps=60)
    anim1.save(output_dir+"/q1e"+".mp4",writer = mywriter)
    #plt.show()