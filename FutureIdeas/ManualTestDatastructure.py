# %%
import FileLoader as FL #data, read_file, Dataset, Data
import EvaluationFunctions as EF
import numpy as np
import scipy
import matplotlib.pyplot as plt
from importlib import reload

reload(EF)
reload(FL)

# Load the data
FL.read_file(
        r"Z:\02_people\Vereijken_Arne\00_PhD\02_LMOKE\2023_0439\2023_0439_1_SCHEDULED_2024-01-10_14-03-06\Scheduled_0067_1440.852s.txt",
        dataformat="MOKE",
    )
# %%

keys = list(FL.data.dataset[0].raw_data.keys())
xdata = FL.data.dataset[0].raw_data[keys[0]]
ydata = FL.data.dataset[0].raw_data[keys[1]]

ydata = EF.hys_norm(xdata, ydata)
fitted_data, params, result = EF.tan_hyseval(xdata, ydata)

plt.figure()
plt.plot(xdata, ydata)
plt.plot(fitted_data['xdata'], fitted_data['ydata'])

#plot results with uncertainty bands
plt.figure()
plt.plot(xdata, ydata, 'b')
plt.plot(fitted_data['xdata'], fitted_data['ydata'], 'r')
uncertainty = result.eval_uncertainty(sigma=3)
plt.fill_between(fitted_data['xdata'], fitted_data['ydata'] - uncertainty, fitted_data['ydata'] + uncertainty, color='gray', alpha=0.8)

# %%

import numpy as np
import matplotlib.pyplot as plt
def asymmetric_arctan(x, a, b):
    return np.where(x >= 0, a * np.arctan(x), b * np.arctan(x))
# Define the range of x values
x = np.linspace(-10, 10, 400)
# Define the coefficients for the asymmetry
a = 1.5
b = 0.2
# Calculate the asymmetric arctan values
y = asymmetric_arctan(x, a, b)
# Plot the original arctan and the asymmetric arctan functions
plt.figure(figsize=(10, 6))
plt.plot(x, np.arctan(x), label='arctan(x)', linestyle='--')
plt.plot(x, y, label=f'asymmetric arctan(x) with a={a}, b={b}')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.title('Asymmetric arctan Function with Different Slopes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# %%

from lmfit import Model, Parameters

def asymm_tan_hys(xdata, a, b, c, d, e, f):
    """
    Takes a list or a single value of xdata and performs two arcus-tanges 
    functions corresponding to two branches of a hysteresis loop

    Parameters
    ----------
    xdata : float, int or list
        input value(s) (typically named x in functions).
    a : float or int
        constant offset.
    b : float or int
        coefficient/amplitude of arctan.
    c : float or int
        coefficient of argument (xdata) inside arctan, mainly steepness.
    d : float or int
        constant offset in xdata (e.g. HEB).
    e : float or int
        branch dependent (in sign) offset in xdata (e.g. HC).
    f : float or int
        asymmetry factor in the slope of the arctan function.

    Returns
    -------
    float, int or list
        calculated value(s).

    """
    # #if arctan of a single value is wanted, average over both branches
    # #make this branch dependent in the future?
    # if type(xdata) in [int, float]:
    #     ydata1 =  a + b * np.arctan(c * (xdata - d + e))
    #     ydata2 =  a + b * np.arctan(c * (xdata - d - e))
    #     return np.mean([np.abs(ydata1), np.abs(ydata2)])
    # else:
    #if xdata is given as a list (hysteresis), split it correspondingly
    #into two branches
    Xdata = np.split(xdata,2)
    ydata1 = np.where(Xdata[0] >= - d + e, a + f * b * np.arctan(c * (Xdata[0] - d + e)), a + b * np.arctan(c * (Xdata[0] - d + e)))
    ydata2 = np.where(Xdata[1] >= - d - e, a + f * b * np.arctan(c * (Xdata[1] - d - e)), a + b * np.arctan(c * (Xdata[1] - d - e)))
    
    #ydata1 = a + b * np.arctan(c * (Xdata[0] - d + e)) + f
    #ydata2 = a + b * np.arctan(c * (Xdata[1] - d - e)) + f
    
    return np.append(ydata1, ydata2) 
    
    
# Create a model from the function
model = Model(asymm_tan_hys)

# Define the parameters
params = Parameters()
params.add('a', value=0.0) # offset
params.add('b', value=-0.7) # amplitude
params.add('c', value=0) # steepness
params.add('d', value=20) # exchange bias field
params.add('e', value=15) # coercive field
params.add('f', value=1) # asymmetry factor

keys = list(FL.data.dataset[0].raw_data.keys())
xdata = FL.data.dataset[0].raw_data[keys[0]]
ydata = EF.hys_norm(xdata, FL.data.dataset[0].raw_data[keys[1]])

# Fit the model to the data
result = model.fit(ydata, params, xdata=xdata)

ydata2 = result.best_fit

#plot results with uncertainty bands
plt.figure()
plt.plot(xdata, ydata, 'b')
plt.plot(xdata, ydata2, 'r')
#uncertainty = result.eval_uncertainty(sigma=3)
#plt.fill_between(xdata, ydata2 - uncertainty, ydata2 + uncertainty, color='gray', alpha=0.8)

#%%
Xdata = np.split(xdata,2)
Ydata = np.split(ydata2,2)

params = result.params
d = params['d'].value
e = params['e'].value

plt.figure()
plt.plot(Xdata[0], Ydata[0], 'b')
plt.plot(Xdata[1], Ydata[1], 'r')

plt.scatter(np.array(Xdata[1])[np.where(Xdata[1] >= - d - e)[0]], np.array(Ydata[1])[np.where(Xdata[1] >= - d - e)[0]], c='g')

# %%


a = 0.001
b = -0.68
c = -0.21
d = -23.77
e = 16.75
f = 1.05

Xdata = np.split(xdata,2)

ydata1 =  a + f * b * np.arctan(c * (Xdata[0] - d + e))
ydata2 =  a + b * np.arctan(c * (Xdata[0] - d + e))

plt.figure()
plt.plot(Xdata[0], ydata1, c='r')
plt.plot(Xdata[0], ydata2, c='b')
plt.scatter(xdata, ydata, c='g')

# %%
