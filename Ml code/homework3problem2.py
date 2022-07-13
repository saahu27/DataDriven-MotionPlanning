import numpy as np
import matplotlib.pyplot as plt
#parameters
rad = 10
thk = 5

def generatedata(rad,thk,sep,n,x1=0,y1=0):
    # center of the top semi-circle
    X1 = x1
    Y1 = y1

    # center of the bottom semi-circle
    X2 = X1 + rad + thk / 2
    Y2 = Y1 - sep
    
    # data points in the top semi-circle
    top = []
    # data points in the bottom semi-circle
    bottom = []
    
    # parameters
    r1 = rad + thk
    r2 = rad
    
    cnt = 1
    while(cnt <= n):
        #uniformed generated points
        x = np.random.uniform(-r1,r1)
        y = np.random.uniform(-r1,r1)
        
        d = x**2 + y**2
        if(d >= r2**2 and d <= r1**2):
            if (y > 0):
                top.append([X1 + x,Y1 + y])
                cnt += 1
            else:
                bottom.append([X2 + x,Y2 + y])
                cnt += 1
        else:
            continue

    return top,bottom

def process_data(top,bottom):
    # X1 = [i[0] for i in top]
    # Y1 = [i[1] for i in top]
    # X2 = [i[0] for i in bottom]
    # Y2 = [i[1] for i in bottom]

    #plt.scatter(X1,Y1,s = 1)
    #plt.scatter(X2,Y2,s = 1)
    x1 = [[1] + i + [1] for i in top]
    x2 = [[1] + i + [-1] for i in bottom]
    data = x1 + x2
    
    data = np.array(data)
    np.random.shuffle(data)
    return data

def pla_algorithm(data):
    #x_point = []
    #y_point = []
    weight = [0,0,0]
    count = 0
    for i in range(len(data)):
        point = data[i]
        point.tolist()
        if(np.sign(np.matmul(point[0:3],weight)) != point[3]):
            weight = np.add(weight,(np.dot(point[3],point[0:3])))
            #axes = plt.gca()
            #plt.scatter(X1,Y1,s = 1)
            #plt.scatter(X2,Y2,s = 1)
            #x_h = np.array(axes.get_xlim())
            #y_h = (-(weight[1]/weight[2])*x_h)-(weight[0]/weight[2])
            #plt.plot(x_h,y_h)
            count += 1
    return count

start = 0.2
stop = 5
step = 0.2
sep_x = np.arange(start, stop+step, step)
sep_y = []
for sep in sep_x:
    t,b = generatedata(rad,thk,sep,2000)
    c = pla_algorithm(process_data(t,b))
    sep_y.append(c)
plt.plot(sep_x, sep_y)
plt.show()



