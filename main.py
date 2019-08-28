import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt

from data_utils import start_params, experimental_data_TSA, experimental_data_MCA


class Parameterize_ODE():

  def __init__(self):
    self.experimental_data = experimental_data
    self.x_data = np.array([12, 17, 22, 27, 32]) #points in which measurement was taken
    self.xrange = np.array([12, 33]) #time interval on which solution should be calculated
    self.dx = 0.1 #resolution of the solution mesh

    ## INDEXES in p, not  VALUES

    self.r = 0  # Viable cancer cells volume doubling time
    self.k = 1  # Tumor carrying capacity
    self.a = 2  # CTLs' killing rate
    self.d = 3  # Clearance rate of dying cells

    self.l = 4  # Decay rate of effector cells
    self.omega = 5  # Baseline T cell recruitment rate
    self.omega2 = 6  # Fold change in the baseline T cell recruitment rate due to immunogenic cell death
    self.e = 7  # Initial fold change in recruitment of cytotoxic T cells caused by immunotherapy
    self.clr = 8  # 9H10 immunotherapy clearance rate

    self.E = 9
    self.D_prop = 10

    self.SFd20 = 11
    self.SFd8 = 12
    self.SFd6 = 13

    self.AId20 = 14
    self.AId8 = 15
    self.AId6 = 16

  @staticmethod
  def p(V):
    return V / np.sum(V)

  def sfd(self,dose, Xi):
    return {20: Xi[self.SFd20], 8: Xi[self.SFd8], 6: Xi[self.SFd6]}[dose]

  def aid(self,dose, Xi):
    return {20: Xi[self.AId20], 8: Xi[self.AId8], 6: Xi[self.AId6]}[dose]

  def C_dot(self, C, D, I, VE, r, k, a):
    """Eq.4: Growth of viable cancer cells volume
    """
    V = C + D + I
    E = VE/V
    return r * C * (1 - V / k) - a * C * E

  def D_dot(self, C, D, I, VE, r, k, a, d):
    """Eq.5: CCs dying in a non-immunogenic manner
    """
    V = C + D + I
    E = VE/V
    return r * C * V / k + a *C * E - d * D

  def I_dot(self, I, d):
    """Eq.7:
    """
    return np.array([-d * I[0], 0])

  def u(self, t, it, e, clr):
    administered_doses_time = it[it < t]
    return e * np.sum(np.exp(-clr * (t - administered_doses_time)))


  def VE_dot(self, C, D, I, VE, l, e, clr, omega, omega2, it, t):
    V = C + D + I
    u = self.u(t, it, e, clr)
    p = self.p(V)
    return - l * VE + (1 + u) * p * omega * np.sum(V + omega2 * I)

  def ode(self, y, t, it, parameters):
    r_val = parameters[self.r]
    k_val = parameters[self.k]
    a_val = parameters[self.a]
    d_val = parameters[self.d]
    omega_val = parameters[self.omega]
    l_val = parameters[self.l]
    clr_val = parameters[self.clr]
    omega2_val = parameters[self.omega2]
    e_val = parameters[self.e]

    C, D, I, VE = np.split(y, 4) #split into 4 submatrices,

    return np.concatenate(
      [
        self.C_dot(C, D, I, VE, r_val, k_val, a_val),
        self.D_dot(C, D, I, VE, r_val, k_val, a_val, d_val),
        self.I_dot(I, d_val),
        self.VE_dot(C, D, I, VE, l_val, e_val, clr_val, omega_val, omega2_val, it, t)
      ]
    )

  def model(self, it, rt, initial_condition, parameters):
    #now the function returns the solution on the full mesh
    #evaluation on the data points is carried out elsewhere

    # ODEs are integrated on daily pieces
    dead_v = initial_condition[:2] * parameters[self.D_prop]
    VE = initial_condition[:2] * parameters[self.E]

    initial_condition += [-dead_v[0], -dead_v[1], dead_v[0], dead_v[1], 0, 0, VE[0], VE[1]]

    rt = np.asarray(rt)
    if rt.size > 0:
      moments = rt[:,0]
      moments = np.unique(np.concatenate((self.xrange, moments)))
    else:
      moments = self.xrange

    #define output variables
    results = np.array([])
    xmesh = np.array([])

    #previous implementation was ineficient, because lsoda was invoked for [day, day+1] interval
    #now we solve in the whole intervals between radiation doses
    for i in range(len(moments)-1):

      if rt.size > 0 and moments[i] in rt[:,0]:
        rt_dose = np.asscalar(rt[np.where(rt[:,0] == moments[i]),1])
        # Eq.6: Introducing non-continous change to variables due to radiation.
        sfd = self.sfd(rt_dose, parameters)
        aid = self.aid(rt_dose, parameters)
        C = initial_condition[0]
        initial_condition[0] = C * sfd  # C1
        initial_condition[2] = initial_condition[2] + C * (1 - sfd) * (1 - aid)  # D1
        initial_condition[4] = initial_condition[4] + C * (1 - sfd) * aid  # E1

      tmpMesh = np.arange(moments[i],moments[i+1],self.dx)
      resInt = integrate.odeint(
      self.ode, initial_condition, tmpMesh, args=(it, parameters) #, hmin=1e-7
      )
      initial_condition = resInt[-1,:] #take last element of the solution as initial condition
      if results.size == 0:
        results = resInt
        xmesh = tmpMesh
      else:
        results = np.concatenate((results,resInt[1:,:]), axis=0)
        xmesh = np.concatenate((xmesh,tmpMesh[1:]), axis=0)

    return xmesh, results


  def prediction(self, group, initial_condition, parameters):
    it = experimental_data[group]['it']
    rt = experimental_data[group]['rt']

    xmesh, solution = self.model(it, rt, initial_condition, parameters)
    #interpolate solution on the datapoints
    V1 = np.interp(self.x_data, xmesh, np.sum(solution[:, (0, 2, 4)],axis = 1))
    V2 = np.interp(self.x_data, xmesh, np.sum(solution[:, (1, 3, 5)], axis=1))
    return V1, V2, xmesh, solution

  def err(self, x, parameters_init, toFit, returnSolutions = False):
    indx = toFit == 1 #those are being fitted
    parameters = parameters_init.copy()
    parameters[indx] = x*parameters_init[indx]
    V_error = np.zeros(shape=[len(self.experimental_data), len(self.x_data)*2])

    solutions = []

    for i, group in enumerate(self.experimental_data):
      V_data = experimental_data[group]["v"]
      initial_condition = np.array([V_data[0, 0], V_data[0, 1], 0., 0., 0., 0., 0., 0.])
      if returnSolutions:
        V1, V2, xmesh, solution = self.prediction(group, initial_condition, parameters)
        solutions.append([xmesh, solution, group, self.x_data, V_data])
      else:
        V1, V2, _, _ = self.prediction(group, initial_condition, parameters)

      V_error[i, :] = (np.concatenate((V1,V2)) / V_data.flatten('F') - 1)**2


    if returnSolutions:
      return np.sum(V_error.flatten()), solutions
    else:
     return np.sum(V_error.flatten())

  def optimization(self, parameters_init, toFit, lower_bounds, upper_bounds):

    #define bounds for the selected parameters
    indx = toFit > 0
    start_value = parameters_init[indx].copy()
    lb = lower_bounds[indx]
    ub = upper_bounds[indx]

    #because parameters have very different orderds of magnitude it is better to fit relative change from the initial value, i.e.
    #initial guess is x0 = 1 and then the parameter value is p = x*p_init
    return optimize.minimize(
      self.err,
      np.ones(lb.size),
      method='trust-constr',
      options={'disp': True, 'verbose' : 2, 'initial_tr_radius': 1.},
      bounds= [(lb[i]/start_value[i], ub[i]/start_value[i]) for i in range(0,lb.size)],
      args = (parameters_init, toFit)
    )


if __name__ == "__main__":

  #choosing experimental data
  #experimental_data = experimental_data_TSA
  experimental_data = experimental_data_MCA

  ode = Parameterize_ODE()

  parameters_definition = start_params.copy()
  parameters_init = parameters_definition[:,0] #initial parameters values
  toFit = parameters_definition[:,1] #if to fit parameters
  lower_bounds = parameters_definition[:,2] #lower bounds for parameters values
  upper_bounds = parameters_definition[:, 3]  #upper bounds for parameters values

  #perform optimization
  optResults = ode.optimization(parameters_init, toFit, lower_bounds, upper_bounds)

  #grab final error and corresponding solutiuons
  err, solutions = ode.err(optResults.x, parameters_init, toFit, True)

  #calculate final parameters set
  params_final = parameters_init.copy()
  params_final[toFit > 0] = params_final[toFit>0]*optResults.x#update those that were fitted
  print("Final fit error:", err)
  print("Final parameter values", params_final)

  #check if any parameter is close to its bound - if that is the case, please expand the bound
  indx = abs(params_final/upper_bounds - 1.) < .05
  indx = np.logical_or(indx,abs(params_final/lower_bounds - 1.) < .05)
  if any(indx):
    print('Warning: some of the parameters are close to their assumed bound', np.where(indx))

  #plotting optimized solutions
  for i in range(0,len(solutions)):
   solution = solutions[i]
   group = solution[2]
   V_data = solution[4]
   fig, (ax1, ax2) = plt.subplots(2, 1)

   ax1.plot(solution[0], np.sum(solution[1][:, (0, 2, 4)], axis = 1), lw=2, label='C', color='blue')
   ax1.plot(solution[3], V_data[:, 0], 'bo')
   ax1.grid()
   ax1.set_ylabel('left')
   ax1.set_ylim(0, 800 if 'TSA' in group else 1500)

   ax2.plot(solution[0], np.sum(solution[1][:, (1, 3, 5)], axis = 1), lw=2, label='C', color='blue')
   ax2.plot(solution[3], V_data[:, 1], 'bo')
   ax2.grid()
   ax2.set_ylabel('right')
   fig.suptitle(group.replace('_', ' '))
   ax2.set_ylim(0, 800 if 'TSA' in group else 1500)
   plt.savefig('plots/{}.png'.format(group))


