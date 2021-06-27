import autograd.numpy as np
from autograd import grad

class ssine:
    def __init__(self):
        self.xmin, self.xmax = -6.4,1.6
        self.ymin, self.ymax = -2.4,2.4
        self.y_start, self.x_start =0.0, -6.0 # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0., 0., 0.  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = np.abs(-x-np.sin(x*4)*-0.24)
        return z 
class Bukin :
    def __init__(self):
        self.xmin, self.xmax = -15.0,0.5
        self.ymin, self.ymax = -3.0,3.0
        self.y_start, self.x_start = 2.0,-1.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = -10.0, 1.0, 0.0  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = 100.0*(np.abs(y-0.01*x*x)**0.5)+0.01*np.abs(x+10.0)
        return z
        

class be :
    def __init__(self):
        self.xmin, self.xmax = -4.0,1.0
        self.ymin, self.ymax = -1.0,1.0
        self.y_start, self.x_start = 0.05,-3.5  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0.0, 0.0, 0.0  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = np.abs(x)+np.abs(y)
        return z
class McCormick:
    def __init__(self):
        self.xmin, self.xmax = -1.5,4.0
        self.ymin, self.ymax = -3.0,4.0
        self.y_start, self.x_start = 3.0,-1.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = -0.54719, -1.54719, -1.9133  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = np.sin(x+y)+(x-y)**2-1.5*x+2.5*y+1.0
        return z
class Beale:
    def __init__(self):
        self.xmin, self.xmax = -5.12,5.12
        self.ymin, self.ymax = -5.12,5.12
        self.y_start, self.x_start = -4.0,2.0  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 3, 0.5, 0  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = np.log(1+(1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2)/10
        return z

class Rastrigin:
    def __init__(self):
        self.xmin, self.xmax = -5.12,5.12
        self.ymin, self.ymax = -5.12,5.12
        self.y_start, self.x_start =-2.0, 2.0 # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0., 0., 0.  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y)) + 20
        return z
        
class Ackley:
    def __init__(self):
        self.xmin, self.xmax = -5.0,5.0
        self.ymin, self.ymax = -5.0,5.0
        self.y_start, self.x_start =-3.5,-4.5 # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0., 0., 0.  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = -20*np.exp(-0.2*(0.5*(x**2+y**2))**0.5)-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20
        return z
class Rose:
    def __init__(self):
        self.xmin, self.xmax = -5.12,5.12
        self.ymin, self.ymax = -5.12,5.12
        self.y_start, self.x_start =-2.0, 2.0 # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 1., 1., 0.  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = (1.-x)**2 + 100.*(y-x*x)**2
        return z    
class Booth:
    def __init__(self):
        self.xmin, self.xmax = -5.12,5.12
        self.ymin, self.ymax = -5.12,5.12
        self.y_start, self.x_start =0.,-4. # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 1., 3., 0.  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = (x+2*y-7)**2+(2*x+y-5)**2
        return z             
class Himme:
    def __init__(self):
        self.xmin, self.xmax = -5.12,5.12
        self.ymin, self.ymax = -5.12,5.12
        self.y_start, self.x_start =0.,-3. # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0., 0., 0.  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = (x*x+y-11)**2+(x+y**2-7)**2
        return z

class saddle:
    def __init__(self):
        self.xmin, self.xmax = -5.,5.
        self.ymin, self.ymax = -5.,5.
        self.y_start, self.x_start =1e-9, 3.1 # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0., -5., 25.  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = x**2-y**2
        return z