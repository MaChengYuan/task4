from __future__ import division
import numpy as np 
from sympy import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
from scipy import optimize
from numpy.random import rand
import random
import math


x_cor = []
y_cor = []
normal_distribution = np.random.normal(0,1,100)

def noise():
    return np.random.choice(normal_distribution)

def generate():
    for i in range(1,1001,20):
        x = 3*i/1000
        x_cor.append(x)
        z = func(x)
        if(z < -100):
            y_cor.append(-100+noise())
        elif(-100 <= z <= 100):
            y_cor.append(z+noise())
        else:
            y_cor.append(100+noise())
    return

def func(x):
    return 1/(x**2 - 3*x +2)
def goal_func():
    return (a*x + b) / (x**2 + c*x + d) 



def Nelder_Mead(x,y):

    a = symbols('a')
    b = symbols('b')
    c = symbols('c')
    d = symbols('d')
 

    aa = bb = cc = dd = 0
    current_node= [10,10,-9,0]
 
    def fun(array):
        
        n = len(x_cor)

        z = (1/n)*sum([(val)**2 for val in (y -(a * x + b) / (x**2 + c * x + d))])
        result = z.subs([(a,array[0]),(b,array[1]),(c,array[2]),(d,array[3])])
        return result
 
    # define range for input
    r_min, r_max = -5.0, 5.0
    # define the starting point as a random sample from the domain
    pt = r_min + rand(2) * (r_max - r_min)
    # perform the search
    result = minimize(fun, current_node, method='nelder-mead')
    # summarize the result
    # print('Status : %s' % result['message'])
    # print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = fun(solution)
    # print('Solution: f(%s) = %.5f' % (solution, evaluation))
    # print(result)

    x = np.linspace(0,3,30)
    # y = current_node[0] / (1+current_node[1]*x)
    y = (solution[0]*x + solution[1]) / (x**2 + solution[2]*x + solution[3])
    plt.plot(x,y,label='Nelder_Mead')
    print('Nelder_Mead  :initial_node={} , iteration={} , cost={}  ,evalutaion of function={}'\
          .format(current_node,result['nit'],result['fun'], result['nfev']))  

    
def anneal(x,y):

   def cal_Energy(X, nVar):
       n = len(x_cor)

       z = (1/n)*sum([(val)**2 for val in (y -(X[0] * x + X[1] ) / (x**2 + X[2]  * x + X[3]))])
       # z = (1/n)*sum([(val)**2 for val in (y_cor -(a*x_cor - b))])
       # result = z.subs([(a,current_node[0]),(b,current_node[1]),(c,current_node[2]),(d,current_node[3])])
       # result = z.subs([(a,current_node[0]),(b,current_node[1])])
       return z

   # 子程式：模擬退火演算法的引數設定
   def ParameterSetting():
       cName = "funcOpt"           # space name
       nVar = 4                    # number of variables
       xMin = [-10,-10,-10,-10]         # min of search space
       xMax = [10,10,10,10]                    # max of search space
       tInitial = 100.0            # 設定初始退火溫度(initial temperature)
       tFinal  = 1                 # 設定終止退火溫度(stop temperature)
       alfa    = 0.98              # 設定降溫引數，T(k)=alfa*T(k-1)
       meanMarkov = 100            # Markov，iteration number
       scale   = 0.5               # learning_scale
       return cName, nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale


   # 模擬退火演算法
   def OptimizationSSA(nVar,xMin,xMax,tInitial,tFinal,alfa,meanMarkov,scale):
       # ====== initiala point generators ======
       randseed = random.randint(1, 100)
       random.seed(randseed)  
       eva_func = 0

     
       xInitial = np.zeros((nVar))   # initial a vector
       for v in range(nVar):
           # random.uniform(min,max) in [min,max] generate a number
           xInitial[v] = random.uniform(xMin[v], xMax[v])
     
       fxInitial = cal_Energy(xInitial, nVar)
       eva_func += 1

       # initialize
       xNew = np.zeros((nVar))         # 初始化，建立陣列
       xNow = np.zeros((nVar))         # 初始化，建立陣列
       xBest = np.zeros((nVar))        # 初始化，建立陣列
       xNow[:]  = xInitial[:]          # 初始化當前解，將初始解置為當前解
       xBest[:] = xInitial[:]          # 初始化最優解，將當前解置為最優解
       fxNow  = fxInitial              # 將初始解的目標函式置為當前值
       fxBest = fxInitial              # 將當前解的目標函式置為最優值
       # print('x_Initial:{:.6f},{:.6f},\tf(x_Initial):{:.6f}'.format(xInitial[0], xInitial[1], fxInitial))

       recordIter = []                 # 初始化，外迴圈次數
       recordFxNow = []                # 初始化，當前解的目標函式值
       recordFxBest = []               # 初始化，最佳解的目標函式值
       recordPBad = []                 # 初始化，劣質解的接受概率
       kIter = 0                       # 外迴圈迭代次數，溫度狀態數
       totalMar = 0                    # 總計 Markov 鏈長度
       totalImprove = 0                # fxBest 改善次數
       nMarkov = meanMarkov            # 固定長度 Markov鏈

       # ====== simulate ======
       # until max temp 
       tNow = tInitial                 # 初始化當前溫度(current temperature)
       while tNow >= tFinal:           # 外迴圈，直到當前溫度達到終止溫度時結束
           # 在當前溫度下，進行充分次數(nMarkov)的狀態轉移以達到熱平衡
           kBetter = 0                 # 獲得優質解的次數
           kBadAccept = 0              # 接受劣質解的次數
           kBadRefuse = 0              # 拒絕劣質解的次數
     

           # until length of markob
           for k in range(nMarkov):    # 內迴圈，迴圈次數為Markov鏈長度
               totalMar += 1           # 總 Markov鏈長度計數器

               # new points
               # has to be between min and max
               
               xNew[:] = xNow[:]
               v = random.randint(0, nVar-1)   #  [0,nVar-1] generate ranodm umber
               xNew[v] = xNow[v] + scale * (xMax[v]-xMin[v]) * random.normalvariate(0, 1)
               # random.normalvariate(0, 1)：產生服從均值為0、標準差為 1 的正態分佈隨機實數
               xNew[v] = max(min(xNew[v], xMax[v]), xMin[v])  # has to be between min and max

             
               fxNew = cal_Energy(xNew, nVar)
               eva_func += 1
               deltaE = fxNew - fxNow

               # accept new point by possibility
               if fxNew < fxNow:  # 更優解：如果新解的目標函式好於當前解，則接受新解
                   accept = True
                   kBetter += 1
               else:  # 容忍解：如果新解的目標函式比當前解差，則以一定概率接受新解
                   pAccept = math.exp(-deltaE / tNow)  # 計算容忍解的狀態遷移概率
                   if pAccept > random.random():
                       accept = True  
                       kBadAccept += 1
                   else:
                       accept = False  
                       kBadRefuse += 1

              
               if accept == True:  # 如果接受新解，則將新解儲存為當前解
                   xNow[:] = xNew[:]
                   fxNow = fxNew
                   if fxNew < fxBest:  # 如果新解的目標函式好於最優解，則將新解儲存為最優解
                       fxBest = fxNew
                       xBest[:] = xNew[:]
                       totalImprove += 1
                       scale = scale*0.99  # 可變搜尋步長，逐步減小搜尋範圍，提高搜尋精度
                       
           
           # save data
           pBadAccept = kBadAccept / (kBadAccept + kBadRefuse)  # 劣質解的接受概率
           recordIter.append(kIter)  # 當前外迴圈次數
           recordFxNow.append(round(fxNow, 4))  # 當前解的目標函式值
           recordFxBest.append(round(fxBest, 4))  # 最佳解的目標函式值
           recordPBad.append(round(pBadAccept, 4))  # 最佳解的目標函式值

           # if kIter%10 == 0:                           # 模運算，商的餘數
           #     # print('i:{},t(i):{:.2f}, badAccept:{:.6f}, f(x)_best:{:.6f}'.\
           #     # format(kIter, tNow, pBadAccept, fxBest))
           #      print('i:{},t(i):{:.2f},evaluation of function:{:.6f}'.\
           #            format(kIter, tNow, eva_func ))
           

           # 緩慢降溫至新的溫度，降溫曲線：T(k)=alfa*T(k-1)
           tNow = tNow * alfa
           kIter = kIter + 1
           # ====== 結束模擬退火過程 ======

       # print('improve:{:d}'.format(totalImprove))
       print('Simulated Annealing  : iteration={},cost={:.2f},evaluation of function={:.6f}'.\
             format(kIter, tNow, eva_func ))
       return kIter,xBest,fxBest,fxNow,recordIter,recordFxNow,recordFxBest,recordPBad



       
       return 
   [cName, nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale] = ParameterSetting()
   [kIter,xBest,fxBest,fxNow,recordIter,recordFxNow,recordFxBest,recordPBad] \
   = OptimizationSSA(nVar,xMin,xMax,tInitial,tFinal,alfa,meanMarkov,scale)

   return xBest


def LMS(x,y):
    limit_iteration = 1000
    a = symbols('a')
    b = symbols('b')
    c = symbols('c')
    d = symbols('d')
    aa = bb = cc = dd = 15
    initial_node = current_node=[-10,-10,7,0]
    iteration = 0
    lamb = 10
    con =1 
    eva_func = 0
    array = [a,b,c,d]
    n = len(x)
    predict = (a * x + b) / (x**2 + c * x + d)
    cost = (1/n)*sum([(val)**2 for val in (y-predict)])    
    hessianf = hessian(cost,array)
    difa = diff(cost,a) 
    difb = diff(cost,b)
    difc = diff(cost,c)
    difd = diff(cost,d)
    eva_func = eva_func+4
    last_cost =cost.subs([(b,current_node[1]),(a,current_node[0]),(c,current_node[2]),(d,current_node[3])])

    while(con == 1 and iteration < limit_iteration):
        iteration += 1

        identity = np.identity(hessianf.shape[0])
        identity = identity*lamb       
        hessianf = Matrix(hessianf)
        hessianf = hessianf.subs([(b,current_node[1]),(a,current_node[0]),(c,current_node[2]),(d,current_node[3])])
        hessianf = np.array(hessianf).astype(np.float64)
        # hessianf = np.array(hessianf)
        
        hessianf = hessianf + identity
        hessian_inv = np.linalg.inv(hessianf)

   
        gradient = []
        gradient.append(difa.subs([(b,current_node[1]),(a,current_node[0]),(c,current_node[2]),(d,current_node[3])]))
        gradient.append(difb.subs([(b,current_node[1]),(a,current_node[0]),(c,current_node[2]),(d,current_node[3])]))
        gradient.append(difc.subs([(b,current_node[1]),(a,current_node[0]),(c,current_node[2]),(d,current_node[3])]))
        gradient.append(difd.subs([(b,current_node[1]),(a,current_node[0]),(c,current_node[2]),(d,current_node[3])]))
        
        
        current_node , gradient = np.array([current_node,gradient])
        #create new node
        current_node = current_node - np.dot(hessian_inv,gradient)

        current_cost =cost.subs([(b,current_node[1]),(a,current_node[0]),(c,current_node[2]),(d,current_node[3])])
        eva_func += 1

        if(current_cost >= last_cost):
            lamb = lamb*10 # accept less hessian matrix
            # print('lamb up ')
           
        else:
            lamb = lamb/10   # accept more hessian matrix
            # print('lamb down ')
        if(abs(last_cost-current_cost) < 0.001):
            # print('stop')
            con = 0
        last_cost = current_cost
    x = np.linspace(0,3,30)
    y = (current_node[0] * x + current_node[1]) / (x**2 + current_node[2] * x + current_node[3])
    plt.plot(x,y,label='Levenberg- Marquardt algorithm')
    plt.legend()
    
    print('Levenberg- Marquardt algorithm :initial_node={} , iteration={} , cost={} , evaluation of function={}'.format(initial_node,iteration,last_cost,eva_func))  

def partical_swarm(x,y):
    a = symbols('a')
    b = symbols('b')
    c = symbols('c')
    d = symbols('d')
    def func1(current_node):
         
        n = len(x)
    
        z = (1/n)*sum([(val)**2 for val in (y -(a * x + b) / (x**2 + c * x + d))])
        result = z.subs([(a,current_node[0]),(b,current_node[1]),(c,current_node[2]),(d,current_node[3])])
        return result

#--- MAIN ---------------------------------------------------------------------+

    class Particle:
        def __init__(self,x0):
            self.position_i=[]          # particle position
            self.velocity_i=[]          # particle velocity
            self.pos_best_i=[]          # best position individual
            self.err_best_i=-1          # best error individual
            self.err_i=-1               # error individual
    
            for i in range(0,num_dimensions):
                self.velocity_i.append(random.uniform(-1,1))
                self.position_i.append(x0[i])
    
        # evaluate current fitness
        def evaluate(self,costFunc):
            self.err_i=costFunc(self.position_i)
    
            # check to see if the current position is an individual best
            if self.err_i < self.err_best_i or self.err_best_i==-1:
                self.pos_best_i=self.position_i
                self.err_best_i=self.err_i
    
        # update new particle velocity
        def update_velocity(self,pos_best_g):
            w=0.5       # constant inertia weight (how much to weigh the previous velocity)
            c1=1        # cognative constant
            c2=2        # social constant
    
            for i in range(0,num_dimensions):
                r1=random.random()
                r2=random.random()
    
                vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
                vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
                self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social
    
        # update the particle position based off new velocity updates
        def update_position(self,bounds):
            for i in range(0,num_dimensions):
                self.position_i[i]=self.position_i[i]+self.velocity_i[i]
    
                # adjust maximum position if necessary
                if self.position_i[i]>bounds[i][1]:
                    self.position_i[i]=bounds[i][1]
    
                # adjust minimum position if neseccary
                if self.position_i[i] < bounds[i][0]:
                    self.position_i[i]=bounds[i][0]
                    
    class PSO():
        def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
            global num_dimensions
    
            num_dimensions=len(x0)
            err_best_g=-1                   # best error for group
            pos_best_g=[]                   # best position for group
            eva_func = 0
    
            # establish the swarm
            swarm=[]
            for i in range(0,num_particles):
                swarm.append(Particle(x0))
    
            # begin optimization loop
            i=0
            while i < maxiter:
                #print i,err_best_g
                # cycle through particles in swarm and evaluate fitness
                for j in range(0,num_particles):
                    swarm[j].evaluate(costFunc)
                    eva_func += 1
    
                    # determine if current particle is the best (globally)
                    if swarm[j].err_i < err_best_g or err_best_g == -1:
                        pos_best_g=list(swarm[j].position_i)
                        err_best_g=float(swarm[j].err_i)
    
                # cycle through swarm and update velocities and position
                for j in range(0,num_particles):
                    swarm[j].update_velocity(pos_best_g)
                    swarm[j].update_position(bounds)
                i+=1
    
            # print final results
            # print ('FINAL:')
            # print (pos_best_g)
            # print (err_best_g)
            x = np.linspace(0,3,30)
            y = (pos_best_g[0] * x + pos_best_g[1]) / (x**2 + pos_best_g[2] * x + pos_best_g[3])
            plt.plot(x,y,label='partical_swarm')
            print('partical_swarm  :initial_node={} , iteration={} , cost={}  ,evalutaion of function={}'\
                  .format(x0,i,err_best_g,eva_func))  
                
           

    if __name__ == "__PSO__":
        main()
    
    #--- RUN ----------------------------------------------------------------------+
    
    initial=[5,5,5,5]               # initial starting location [x1,x2...]
    bounds=[(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    PSO(func1,initial,bounds=bounds , num_particles=15,maxiter=30)
    




def main():
    generate()
    
    plt.scatter(x_cor,y_cor)
    x = np.array(x_cor)
    y = np.array(y_cor)   
    print('-----------------')
    LMS(x, y)
    print()
    Nelder_Mead(x,y)
    print()
    partical_swarm(x,y)
    print()
    
    anneal_best = anneal(x,y)
    x = np.linspace(0,3,30)
    y = (anneal_best[0] * x + anneal_best[1]) / (x**2 + anneal_best[2] * x + anneal_best[3])
    plt.plot(x,y,label='anneal_simulation')
    
    plt.legend()
    
    

    
    

    
    

if __name__ == "__main__":
    main()