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

def maki_thompson(t, y, lambd, alpha):
    X = y[0]
    Y = y[1]
    Z = y[2]
    dX = -lambd * X * Y
    dY =  lambd * X * Y - alpha * Y * (1 - X)
    dZ =  alpha * Y * (1 - X)
    return np.array([dX, dY, dZ])

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

class general_nn(nn.Module):
  def __init__(self, time):
    super().__init__()

    self.net = nn.Sequential(
               nn.Linear(1, 32),
               nn.Tanh(),
               nn.Linear(32, 32),
               nn.Tanh(),
               nn.Linear(32, 32),
               nn.Tanh(),
               nn.Linear(32, 3))

    self.lam = nn.Parameter(torch.tensor([0.5]))
    self.delta = nn.Parameter(torch.tensor([0.5]))
    self.alpha = nn.Parameter(torch.tensor([0.5]))

    self.tMax = max(time)

  def forward(self, t):
    t_norm = t / self.tMax
    return nn.Softplus()(self.net(t_norm))

# NN Simulation
def train(t_real, sol_noisy, nColloc, y_0, hasWeights, modelType, idHPoints):
    
    model = general_nn(t_real)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Weights
    ic_weight = 5.0
    edo_weight = 35.0
    data_weight = 0.9
    sum_weight = 1.0
    
    # Helper points
    t_h = t_real[idHPoints].clone().detach()
    solH = torch.tensor(sol_noisy[idHPoints])
    
    
    # tColloc
    tColloc = torch.linspace(t_start, t_end, nColloc).reshape(-1, 1)
    tColloc.requires_grad = True
    
    _0_0Tensor = torch.tensor([0.0])
    
    # y0 tensor
    y_0_t = torch.tensor(y_0)
    
    # Training loop
    startTime = time.perf_counter()
    for epoch in range(n_ephocs):

        pred = model(tColloc)
        X, Y, Z = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        
        # Loss IC
        lossIC = torch.mean((model(_0_0Tensor) - y_0_t)**2)

        # Loss EDO
        dx_dt = torch.autograd.grad(X, tColloc, grad_outputs= torch.ones_like(X), create_graph= True)[0]
        dy_dt = torch.autograd.grad(Y, tColloc, grad_outputs= torch.ones_like(Y), create_graph= True)[0]
        dz_dt = torch.autograd.grad(Z, tColloc, grad_outputs= torch.ones_like(Z), create_graph= True)[0]

        lam =model.lam# torch.abs(model.lam)
        delta = model.delta#torch.abs(model.delta)
        alpha = model.alpha#torch.abs(model.alpha)

        diff_x = dx_dt + lam * X * Y
        diff_y = dy_dt - lam * X * Y + delta * Y + alpha*Y*(1-X)
        diff_z = dz_dt - delta * Y - alpha* Y *(1-X)
        lossEDO = torch.mean(diff_x**2 + diff_y**2 + diff_z**2)


        # Loss Data
        tmpData = model(t_h)
        lossData = torch.mean((tmpData[:, 1] - solH[:, 1])**2)

        # Loss SUM_1
        lossSUM_1 = torch.mean((1.0 - X - Y - Z)**2)

        if (hasWeights == True):
            loss = (ic_weight * lossIC + edo_weight * lossEDO + data_weight * lossData + sum_weight * lossSUM_1)
        else:
            loss = lossIC + lossEDO + lossData + lossSUM_1
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
    endTime = time.perf_counter()
    return model, endTime - startTime, idHPoints

def getNoiseBase(sol):
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
def worker(scenarioId, modelType, nPoints, nNoise, outliersPerc, outliersNoisePerc):
    print(f"Scenario {scenarioId}")
    np.random.seed(scenarioId)
    torch.manual_seed(scenarioId)

    results = []
    if (modelType == "SIR"):
        sol = solSir
    elif (modelType == "MT"):
        sol = solMT
    noiseBase, noiseSign = getNoiseBase(sol)
    solNoisy = getNoisySolution(sol, noiseBase, noiseSign, nNoise)
    
    id_h_points = []
    
    id_h_points = np.linspace(0, n_points - 1, nPoints, dtype = int)

    # Outliers
    
    nOutliers = int(nPoints * outliersPerc/ 100.0)
    selectedOutliers = np.random.choice(id_h_points, nOutliers, replace = False)
    outliersNoiseData = getNoisySolution(sol, noiseBase, noiseSign, outliersNoisePerc)

    solNoisy[selectedOutliers] = outliersNoiseData[selectedOutliers]
    
    #print(f"Training with point {point}")
    modelBalanced, trainingTime, idHPoints = train(t_real, solNoisy, 50, y_0, True, modelType, id_h_points)

    predBalanced = modelBalanced(t_real).detach().numpy()
    errorS = np.sqrt(np.mean((predBalanced[:, 0] - sol[:, 0])**2))
    errorI = np.sqrt(np.mean((predBalanced[:, 1] - sol[:, 1])**2))
    errorR = np.sqrt(np.mean((predBalanced[:, 2] - sol[:, 2])**2))
    errorTotal = np.sqrt(np.mean((predBalanced[:, 0] - sol[:, 0])**2 + (predBalanced[:, 1] - sol[:, 1])**2 + (predBalanced[:, 2] - sol[:, 2])**2))

    t_plot = t_real.detach().numpy()
    
    #plt.plot(t_plot, sol[:, 1], label = "Real")
    #plt.plot(t_plot, predBalanced[:, 1], label ="Balanced")
    #plt.scatter(t_plot[idHPoints], solNoisy[idHPoints, 1], color = "red", label = "Noisy Data")
    
    
    results.append({"points": nPoints,
                "balanced": "yes",
                "noise": nNoise,
                "error 1" : errorS,
                "error 2" : errorI,
                "error 3" : errorR,
                "error TOTAL": errorTotal,
                "time": trainingTime,
                "lambda": modelBalanced.lam.item(),
                "delta": modelBalanced.delta.item(),
                "alpha": modelBalanced.alpha.item(),
                })
    
    
    del(modelBalanced)
    

    modelUnbalanced, trainingTime, idHPoints = train(t_real, solNoisy, 50, y_0, False, modelType, id_h_points)
    predUnbalanced = modelUnbalanced(t_real).detach().numpy()
    errorS = np.sqrt(np.mean((predUnbalanced[:, 0] - sol[:, 0])**2))
    errorI = np.sqrt(np.mean((predUnbalanced[:, 1] - sol[:, 1])**2))
    errorR = np.sqrt(np.mean((predUnbalanced[:, 2] - sol[:, 2])**2))
    errorTotal = np.sqrt(np.mean((predUnbalanced[:, 0] - sol[:, 0])**2 + (predUnbalanced[:, 1] - sol[:, 1])**2 + (predUnbalanced[:, 2] - sol[:, 2])**2))

    
    results.append({"points": nPoints,
                    "balanced": "no",
                    "noise": nNoise,
                    "error 1" : errorS,
                    "error 2" : errorI,
                    "error 3" : errorR,
                    "error TOTAL": errorTotal,
                    "time": trainingTime,
                    "lambda": modelUnbalanced.lam.item(),
                    "delta": modelUnbalanced.delta.item(),
                    "alpha": modelUnbalanced.alpha.item(),
                    })
    
    

    del(modelUnbalanced)


    '''
    plt.plot(t_plot, predUnbalanced[:, 1], label = "Unbalanced")
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.draw()
    plt.show()
    '''
    
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
n_ephocs = 20000

# Beta & gamma
beta = 0.8
gamma = 0.05


# Real RK4 Solution
solSir = rk4(SIR, y_0, t_real.detach().numpy(), beta, gamma)
solMT = rk4(maki_thompson, y_0, t_real.detach().numpy(), beta, gamma)
# Learning Rate
lr = 1e-3


if __name__ == "__main__":

    nCores = cpu_count()//2
    print(f"PC has {nCores} cores")
    
    modelType = "MT"

    args = [(i, modelType, 16, 5.0, 30.0, 25.0) for i in range (20)]
    with Pool(processes = nCores) as pool:
        final_results = pool.starmap(worker, args)

    flat_results = [ r for scenario in final_results for r in scenario ]

    df = pd.DataFrame(flat_results)
    print(df)
    print("\n\n\n\n\n")
    
    summary = df.groupby("balanced").agg({
        "error 1": "mean",
        "error 2": "mean",
        "error 3": "mean",
        "error TOTAL": "mean",
        "lambda": "mean",
        "delta": "mean",
        "alpha": "mean",
        "time": "mean"
        
    }).reset_index()
    

    print(summary.round(6))