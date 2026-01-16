import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from multiprocessing import Pool, cpu_count

# Torch config
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def SIR(t, y, beta, gamma):
    S = y[0]
    I = y[1]
    R = y[2]
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])

def rk4(f, y0, t, beta, gamma):
    n = len(t)
    y0 = np.array(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n-1):
        h = t[i+1] - t[i]

        k1 = np.array(f(t[i], y[i], beta, gamma))
        k2 = np.array(f(t[i] + h/2, y[i] + h*k1/2, beta, gamma))
        k3 = np.array(f(t[i] + h/2, y[i] + h*k2/2, beta, gamma))
        k4 = np.array(f(t[i] + h, y[i] + h*k3, beta, gamma))

        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    return y

class sir_nn(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(1, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 3))

        self.beta = nn.Parameter(torch.tensor([0.5]))
        self.gamma = nn.Parameter(torch.tensor([0.5]))

    def forward(self, t):
        return nn.Softplus()(self.net(t))
  
# NN Simulation
def train(nHelperPoints, t_real, sol_noisy, nColloc, y_0, beta, gamma, hasWeights):
    model = sir_nn()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Weights
    ic_weight = 10.0
    edo_weight = 40.0
    data_weight = 0.9
    sum_weight = 1.0
    r0_weight = 5.0
    
    # Helper points
    h_points = []
    id_h_points = []
    
    id_h_points = np.linspace(0, n_points - 1, nHelperPoints, dtype = int)
    h_points = t_real[id_h_points].detach().numpy()
    t_h = t_real[id_h_points].clone().detach()
    solH = torch.tensor(sol_noisy[id_h_points])
    
    
    # tColloc
    tColloc = torch.linspace(t_start, t_end, nColloc).reshape(-1, 1)
    tColloc.requires_grad = True
    
    _0_0Tensor = torch.tensor([0.0])
    
    # y0 tensor
    y_0_t = torch.tensor(y_0)

    # Stop limit
    bestLoss = float('inf')
    minEpochs = 2000
    checkEpochs = 200
    patienceCounter = 0
    patienceLimit = 500
    
    # Training loop
    startTime = time.perf_counter()
    for epoch in range(n_ephocs):

        pred = model(tColloc)
        S, I, R = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        
        # Loss IC
        lossIC = torch.mean((model(_0_0Tensor) - y_0_t)**2)

        # Loss EDO
        ds_dt = torch.autograd.grad(S, tColloc, grad_outputs= torch.ones_like(S), create_graph= True)[0]
        di_dt = torch.autograd.grad(I, tColloc, grad_outputs= torch.ones_like(I), create_graph= True)[0]
        dr_dt = torch.autograd.grad(R, tColloc, grad_outputs= torch.ones_like(R), create_graph= True)[0]
    
        diff_s = ds_dt + beta * S * I
        diff_i = di_dt - beta * S * I + gamma * I
        diff_r = dr_dt -  gamma * I
    
        lossEDO = torch.mean(diff_s**2 + diff_i**2 + diff_r**2)

        # Loss Data
        tmpData = model(t_h)
        lossData = torch.mean((tmpData - solH)**2)
        
        # Loss SUM_1
        lossSUM_1 = torch.mean((1.0 - S - I - R)**2)

        # Loss R0
        lossR0 = ((beta/gamma) - (model.beta/model.gamma))**2

        if (hasWeights == True):
            loss = (ic_weight * lossIC + edo_weight * lossEDO + data_weight * lossData + sum_weight * lossSUM_1 + r0_weight * lossR0)
        else:
            loss = lossIC + lossEDO + lossData + lossSUM_1 + lossR0
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if (epoch % 500 == 0):
        #  print (epoch)

        if (epoch % checkEpochs == 0 and epoch >= minEpochs):
            lossValue = loss.item()

            if (lossValue < bestLoss):
                bestLoss = lossValue
                patienceCounter = 0
            else:
                patienceCounter += checkEpochs

            
            if (patienceCounter >= patienceLimit):
                print(f"Stopping at epoch {epoch}")
                break
        
    endTime = time.perf_counter()
    return model, endTime - startTime, id_h_points

def getNoiseBase():
    noiseBase = []
    noiseBase.append(np.random.uniform(0.7, 1.0, size=sol.shape[0])) # S
    noiseBase.append(np.random.uniform(0.7, 1.0, size=sol.shape[0])) # I
    noiseBase.append(np.random.uniform(0.7, 1.0, size=sol.shape[0])) # R
    
    noiseSign = []
    noiseSign.append(np.random.choice([-1, 1], size=sol.shape[0])) # S
    noiseSign.append(np.random.choice([-1, 1], size=sol.shape[0])) # I
    noiseSign.append(np.random.choice([-1, 1], size=sol.shape[0])) # R

    return noiseBase, noiseSign

def getNoisySolution(solution, noiseBase, noiseSign, percentage):
    solNoisy = solution.copy()
    for k in range(3):  # S, I, R
        maxVal = np.max(solution[:, k])
        noiseScaled = noiseBase[k] * noiseSign[k] * (percentage/100.0) * maxVal
        
        solNoisy[:, k] += noiseScaled
        solNoisy[:, k]  = np.abs(solNoisy[:, k])
    solNoisy = np.clip(solNoisy, 0.0, 1.0)

    return solNoisy

# Process
def worker(scenarioId, nPoints, noiseLevels):
    print(f"Scenario {scenarioId}")
    np.random.seed(scenarioId)
    torch.manual_seed(scenarioId)

    results = []
    noiseBase, noiseSign = getNoiseBase()
    for noise in noiseLevels:
        #print(f"\tTraining with noise {noise}")
        solNoisy = getNoisySolution(sol, noiseBase, noiseSign, noise)
        for point in nPoints:
            #print(f"Training with point {point}")
            modelBalanced, trainingTime, idHPoints = train(point, t_real, solNoisy, 50, y_0, beta, gamma, True)

            predBalanced = modelBalanced(t_real).detach().numpy()
            errorS = np.sqrt(np.mean((predBalanced[:, 0] - sol[:, 0])**2))
            errorI = np.sqrt(np.mean((predBalanced[:, 1] - sol[:, 1])**2))
            errorR = np.sqrt(np.mean((predBalanced[:, 2] - sol[:, 2])**2))
            errorTotal = np.sqrt(np.mean((predBalanced[:, 0] - sol[:, 0])**2 + (predBalanced[:, 1] - sol[:, 1])**2 + (predBalanced[:, 2] - sol[:, 2])**2))


            results.append({"points": point,
                            "balanced": "yes",
                            "noise": noise,
                            "error S" : errorS,
                            "error I" : errorI,
                            "error R" : errorR,
                            "error TOTAL": errorTotal,
                            "time": trainingTime,
                            "beta": modelBalanced.beta.item(),
                            "gamma": modelBalanced.gamma.item(),
                            })
            del(modelBalanced)

            modelUnbalanced, trainingTime, idHPoints = train(point, t_real, solNoisy, 50, y_0, beta, gamma, False)
            predUnbalanced = modelUnbalanced(t_real).detach().numpy()
            errorS = np.sqrt(np.mean((predUnbalanced[:, 0] - sol[:, 0])**2))
            errorI = np.sqrt(np.mean((predUnbalanced[:, 1] - sol[:, 1])**2))
            errorR = np.sqrt(np.mean((predUnbalanced[:, 2] - sol[:, 2])**2))
            errorTotal = np.sqrt(np.mean((predUnbalanced[:, 0] - sol[:, 0])**2 + (predUnbalanced[:, 1] - sol[:, 1])**2 + (predUnbalanced[:, 2] - sol[:, 2])**2))

            results.append({"points": point,
                            "balanced": "no",
                            "noise": noise,
                            "error S" : errorS,
                            "error I" : errorI,
                            "error R" : errorR,
                            "error TOTAL": errorTotal,
                            "time": trainingTime,
                            "beta": modelUnbalanced.beta.item(),
                            "gamma": modelUnbalanced.gamma.item(),
                            })
            del(modelUnbalanced)
    
    return results

# Global arguments
t_start = 0.0
t_end = 40.0
n_points = 2000
t_real = torch.linspace(t_start, t_end, n_points).reshape(-1, 1)

# Initial conditions
s_0, i_0, r_0 = 0.95, 0.05, 0.0
y_0 = np.array([s_0, i_0, r_0])

# N Epocas
n_ephocs = 10000

# Beta & gamma
beta = 0.8
gamma = 0.05


# Real RK4 Solution
sol = rk4(SIR, y_0, t_real.detach().numpy(), beta, gamma);
# Learning Rate
lr = 1e-3


if __name__ == "__main__":

    nCores = cpu_count()//2
    print(nCores)
    nPoints = [4, 2, 4, 8]
    noiseLevels = [0.0, 2.5, 5.0]
    nTests = 60

    args = [ (i, nPoints, noiseLevels) for i in range(nTests) ]

    with Pool(processes=nCores) as pool:
        all_results = pool.starmap(worker, args)

    flat_results = [ r for scenario in all_results for r in scenario ]
    df = pd.DataFrame(flat_results)
    print(df)
    df.to_csv("results.csv", index = False)