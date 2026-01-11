import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter



x_lims = [-2, 2]
y_lims = [-2, 2]

Lx = x_lims[1] - x_lims[0]
Ly = y_lims[1] - y_lims[0]

Nx = 40
Ny = 40

dx = Lx/Nx
dy = Ly/Ny

xvals = np.linspace(x_lims[0], x_lims[1], Nx+1)
yvals = np.linspace(y_lims[0], y_lims[1], Ny+1)

[X, Y] = np.meshgrid( xvals, yvals, indexing='ij' )

X_flat = X.flatten()
Y_flat = Y.flatten()



## Non-trivial geometry

include = ( (X_flat+0.5)**2 + (Y_flat+0.5)**2 >= 0.5**2 )

# sideLen = 1
# x_obs   = -0.5
# include = ( 0.5*abs(X_flat-x_obs + Y_flat) + 0.5*abs(X_flat-x_obs - Y_flat) > 0.5*sideLen )


Xc = X_flat[include]
Yc = Y_flat[include]


# Adjacenecy of cells
h1 = np.round( abs( Xc[None,:] - Xc[:,None] ) - dx, 3 ) == 0
h2 = np.round( abs( Yc[None,:] - Yc[:,None] ), 3 ) == 0
Adx = h1 * h2

v1 = np.round( abs( Yc[None,:] - Yc[:,None] ) - dy, 3 ) == 0
v2 = np.round( abs( Xc[None,:] - Xc[:,None] ), 3 ) == 0
Ady = v1 * v2


# Number of neighbours for each cell
neighbours_x = np.sum( Adx, 1 )
neighbours_y = np.sum( Ady, 1 )
neighbours   = neighbours_x + neighbours_y

# Logical array, indicates if cell is on the boundary
boundIdx = ( neighbours < 4 )


# Face normal vectors
normal_x = np.round( ( Xc[None,:] - Xc[:,None] ) * Adx / dx, 3)
normal_y = np.round( ( Yc[None,:] - Yc[:,None] ) * Ady / dy, 3)


ParX = 0.5 * ( normal_x - np.diag(np.sum(normal_x, 1)) ) / dx
ParY = 0.5 * ( normal_y - np.diag(np.sum(normal_y, 1)) ) / dy


# Boundary types
scr = np.full_like(X, True, dtype=bool)
scr[-1,1:-1] = False

Ubounds = boundIdx * scr.flatten()[include]    # Dirichlet boundary indices
pbounds = boundIdx * ~Ubounds                  # Neumann   boundary indices


# Boundary outward pointing normal vector components
bnormal_x = - np.sum(normal_x,1)
bnormal_y = - np.sum(normal_y,1)


# Normal derivative operator at the boundary
bound_ND = bnormal_x[:,None] * ParX + bnormal_y[:,None] * ParY



## ---------------------------------------------------------------------------|
## Simulation Tools                                                           |
## ---------------------------------------------------------------------------|

## Timeline setup
tmax = 10
dt   = 0.02

tvals = np.arange( 0, tmax, step=dt )


# Fluid properties
density   = 1
viscosity = 0.002


# Fluid forces
fun1 = np.heaviside( 1/16 - (Xc + 1.25)**2 - Yc**2, 0 )
fun2 = np.heaviside( 1/16 - (Xc - 1.25)**2 - Yc**2, 0 )

# fx = np.zeros_like(Xc)
# fy = 3*(fun1 - fun2)

fx = np.zeros_like(Xc)
fy = np.zeros_like(Xc)


# Construct the Laplacian Matrix
Mx  = Adx - np.diag(neighbours_x)
My  = Ady - np.diag(neighbours_y)
Lap = 1/dx**2 * Mx + 1/dy**2 * My


Lap_ext = np.ones( [Lap.shape[0] + 1]*2 )
Lap_ext[:-1,:-1] = Lap
Lap_ext[-1,-1]   = 0



def plotMatrix(f_col):
    v = np.zeros_like(X_flat)
    v[include] = f_col
    return v.reshape(X.shape)
        

def matrixSelect(Mat, idx1, idx2):
    ind = np.kron( idx1, idx2 )
    Vec = Mat.flatten()
    
    VecSel = Vec[ind]
    
    return VecSel.reshape( [ sum(idx1), sum(idx2) ] )


def makeAdvector(Ux, Uy):
    Un = 0.5 * ( ( Ux[:,None] + Ux[None,:] ) * normal_x / dx
                   + ( Uy[:,None] + Uy[None,:] ) * normal_y / dy )
    
    onDiags = np.diag(np.sum(Un, 1))
    
    return 0.5*(Un + onDiags)
    

def gradient(v):
    return [ ParX @ v, ParY @ v ]


def divergence(Ux, Uy):    
    return ParX @ Ux + ParY @ Uy



def implicitTimeStep( U, Q, S, inds ):
    M0 = matrixSelect( Q, ~inds, inds )
    c  = M0 @ U[inds]
    
    M1 = matrixSelect(Q, ~inds, ~inds)
    M2 = np.identity(M1.shape[0]) - dt * M1
    
    w  = U[~inds] + dt * ( c + S[~inds] )
    
    U[~inds] = np.linalg.solve(M2, w)
    
    return U


def momentumStep(Ux, Uy):
    diffuser = ( viscosity / density ) * Lap
    advector = makeAdvector(Ux, Uy)
    OpMat    = diffuser - advector
    
    Ux = implicitTimeStep(Ux, OpMat, fx/density, Ubounds)
    Uy = implicitTimeStep(Uy, OpMat, fy/density, Ubounds)
    
    return [Ux, Uy]


def PoissonSolver(F, v0):
    ind = pbounds
    M0  = matrixSelect( Lap, ~ind, ind )
    c   = M0 @ v0[ind]
    M1  = matrixSelect( Lap, ~ind, ~ind )
    
    v0[~ind] = np.linalg.solve( M1, -c + F[~ind] )
    
    return v0


def PoissonSolver_ins(F):
    v = np.linalg.solve( Lap_ext, np.append(F, 0) )
    return v[:-1]


def pressureStep(Ux, Uy, p0):
    div = divergence(Ux, Uy)
    
    if all( Ubounds[boundIdx] ):
        p = PoissonSolver_ins(div)
    else:
        p = PoissonSolver(div, p0)
        
    
    [Px, Py] = gradient(p)
    
    Ux += - Px * ~Ubounds
    Uy += - Py * ~Ubounds

    # Extra correction term -- Force insulated boundary
    M0 = matrixSelect( bound_ND, pbounds, ~pbounds )
    cx = M0 @ Ux[~pbounds]
    cy = M0 @ Uy[~pbounds]
    
    M = matrixSelect( bound_ND, pbounds, pbounds )
    
    Ux[pbounds] = np.linalg.solve( M, -cx )
    Uy[pbounds] = np.linalg.solve( M, -cy )
    
    return [Ux, Uy, p]


def fluidStep(Ux, Uy, p0):
    [Ux, Uy]    = momentumStep(Ux, Uy)
    [Ux, Uy, p] = pressureStep(Ux, Uy, p0)
    
    return [Ux, Uy, p]
    
    
# ----------------------------------------------------------------------------|
# Initial conditions                                                          |
# ----------------------------------------------------------------------------|

scr = np.zeros_like(X)
scr[0,1:-1] = 1

Ux = scr.flatten()[include]
Uy = np.zeros_like(Xc)
p0 = np.zeros_like(Xc)

p  = p0


# ----------------------------------------------------------------------------|
# Plot Setup                                                                  |
# ----------------------------------------------------------------------------|

fig, ax = plt.subplots( figsize=[10,10], dpi=60 )

fig.suptitle( 'Fluid Velocity And Pressure: 2D Wind Tunnel' )

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_aspect('equal')

hplot = ax.pcolormesh( X, Y, np.zeros_like(X), cmap='jet', vmin=0, vmax=1 )
# fig.colorbar( hplot, ax=ax )

qplot = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(X), pivot='middle', scale=40)


plt.show()


# ----------------------------------------------------------------------------|
# Run Simulation                                                              |
# ----------------------------------------------------------------------------|

writer = PillowWriter( fps=10, metadata=None )


max_count = 0.1/dt
count = max_count


with writer.saving( fig, '2D-WindTunnel-Sim2.gif', 60 ):
    for t in tvals:
        
        if count+1 < max_count:
            count += 1
        else:
            count = 0
            
            qplot.remove()
            hplot.remove()
            
            Ux_mat = plotMatrix(Ux)
            Uy_mat = plotMatrix(Uy)
            p_mat  = plotMatrix(p)
            
            hplot = ax.pcolormesh( X, Y, p_mat, cmap='jet', vmin=-0.04, vmax=0.04 )
            qplot = ax.quiver(X, Y, Ux_mat, Uy_mat, pivot='middle', scale=40)
            
            writer.grab_frame()
        
        [Ux, Uy, p] = fluidStep(Ux, Uy, p0)

writer.finish()







