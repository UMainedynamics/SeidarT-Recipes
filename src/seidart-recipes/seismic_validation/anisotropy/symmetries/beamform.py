# Calculate the receiver locations for a given radius from the source then pull 
# the values out into an array. 


import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation
from seidart.visualization.slice25d import slicer

# ------------------------------------------------------------------------------
## Initiate the model and domain objects
# project_file = 'isotropic.json' 
project_file = 'orthorhombic.json' 

receiver_file = 'receivers_ref.xyz'

dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)


def build_mesh(nx, ny, nz, dx, dy, dz, origin=(0.0, 0.0, 0.0)):
    """
    Create coordinate arrays for a rectilinear 3D grid with possibly non-uniform spacing.
    Returns X,Y,Z each shaped (nx, ny, nz) in meters.
    origin: (x0,y0,z0) is the coordinate of index (0,0,0).
    """
    x0, y0, z0 = origin
    x = x0 + dx * np.arange(nx, dtype=float)
    y = y0 + dy * np.arange(ny, dtype=float)
    z = z0 + dz * np.arange(nz, dtype=float)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")  # shapes (nx,ny,nz)
    return X, Y, Z


def source_to_index(src_xyz, dx, dy, dz, origin=(0.0,0.0,0.0), rounding="nearest"):
    """
    Convert a source location in meters to integer indices.
    rounding: 'nearest' | 'floor' | 'ceil'
    """
    x0, y0, z0 = origin
    fx = (src_xyz[0] - x0) / dx
    fy = (src_xyz[1] - y0) / dy
    fz = (src_xyz[2] - z0) / dz
    if rounding == "nearest":
        ijk = np.rint([fx, fy, fz]).astype(int)
    elif rounding == "floor":
        ijk = np.floor([fx, fy, fz]).astype(int)
    elif rounding == "ceil":
        ijk = np.ceil([fx, fy, fz]).astype(int)
    else:
        raise ValueError("rounding must be 'nearest', 'floor', or 'ceil'")
    return tuple(ijk)

def index_to_source(src_ijk, dx, dy, dz, origin=(0.0,0.0,0.0)):
    """
    Convert integer indices to a source location in meters (cell center convention).
    """
    x0, y0, z0 = origin
    i, j, k = src_ijk
    return np.array([x0 + i*dx, y0 + j*dy, z0 + k*dz], dtype=float)


def distances_from_source(X, Y, Z, src_xyz, metric="physical", dx=None, dy=None, dz=None):
    """
    Compute distance from source to every node.
    metric:
      - 'physical' -> Euclidean distance in meters using X,Y,Z (requires src_xyz)
      - 'index'    -> Euclidean distance in index space (requires dx,dy,dz to scale or treat as 1)
    Returns D shaped like X (nx,ny,nz).
    """
    if metric == "physical":
        dxv = X - src_xyz[0]
        dyv = Y - src_xyz[1]
        dzv = Z - src_xyz[2]
        D = np.sqrt(dxv*dxv + dyv*dyv + dzv*dzv)
        return D
    elif metric == "index":
        # If spacings provided, normalize coordinates first so distance is in "index units"
        if dx is None or dy is None or dz is None:
            raise ValueError("dx, dy, dz must be provided for metric='index'")
        i_like = (X - X[0,0,0]) / dx
        j_like = (Y - Y[0,0,0]) / dy
        k_like = (Z - Z[0,0,0]) / dz
        # Convert src_xyz to "index-like" space too:
        si = (src_xyz[0] - X[0,0,0]) / dx
        sj = (src_xyz[1] - Y[0,0,0]) / dy
        sk = (src_xyz[2] - Z[0,0,0]) / dz
        D = np.sqrt((i_like - si)**2 + (j_like - sj)**2 + (k_like - sk)**2)
        return D
    else:
        raise ValueError("metric must be 'physical' or 'index'")


def select_shell_nodes(D, radius, tol):
    """
    Boolean mask selecting nodes with radius - tol <= D <= radius + tol.
    D: distance field (nx,ny,nz)
    radius, tol: in same units as D (meters if physical; index units if metric='index')
    Returns:
      - mask (nx,ny,nz) bool
      - flat_indices (N,3) integer index triplets for the selected nodes
    """
    mask = (D >= (radius - tol)) & (D <= (radius + tol))
    I, J, K = np.where(mask)
    idx = np.column_stack([I, J, K])
    return mask, idx


def directions_and_angles(X, Y, Z, src_xyz, idx_list):
    """
    Compute unit direction vectors and (theta, phi) angles from source to each selected node.
    Returns:
      - Rhat: (N,3) unit vectors
      - theta: (N,) inclination from +Z in [0, pi]
      - phi:   (N,) azimuth from +X in XY-plane in [-pi, pi]
    """
    pts = np.array([X[tuple(i)], Y[tuple(i)], Z[tuple(i)]]).T  # (N,3)
    d = pts - src_xyz[None, :]
    r = np.linalg.norm(d, axis=1)
    # Avoid divide-by-zero for the source cell:
    r = np.where(r == 0.0, 1.0, r)
    Rhat = d / r[:, None]
    # angles
    phi = np.arctan2(d[:,1], d[:,0])      # azimuth
    costh = np.clip(d[:,2]/r, -1.0, 1.0)  # for numerical safety
    theta = np.arccos(costh)              # inclination
    return Rhat, theta, phi


def directions_and_angles(X, Y, Z, src_xyz, idx_list):
    """
    Compute unit direction vectors and (theta, phi) angles from source to each selected node.

    Parameters
    ----------
    X, Y, Z : ndarray
        3D coordinate fields from build_mesh().
    src_xyz : array-like, shape (3,)
        Source position in meters.
    idx_list : ndarray, shape (N,3)
        Integer (i,j,k) indices of selected nodes.
    
    Returns
    -------
    Rhat : (N,3) ndarray
        Unit direction vectors from source to each node.
    theta : (N,) ndarray
        Inclination (angle from +Z, range 0–π).
    phi : (N,) ndarray
        Azimuth (angle from +X, range -π–π).
    """
    # Extract coordinates of selected nodes
    pts = np.column_stack([
        X[idx_list[:,0], idx_list[:,1], idx_list[:,2]],
        Y[idx_list[:,0], idx_list[:,1], idx_list[:,2]],
        Z[idx_list[:,0], idx_list[:,1], idx_list[:,2]],
    ])  # shape (N,3)
    
    # Vectors from source to each point
    d = pts - src_xyz[None, :]
    r = np.linalg.norm(d, axis=1)
    
    # Avoid division by zero at the source
    r_safe = np.where(r == 0.0, 1.0, r)
    Rhat = d / r_safe[:, None]
    
    # Compute spherical angles
    phi = np.arctan2(d[:,1], d[:,0])  # azimuth
    costh = np.clip(d[:,2] / r_safe, -1.0, 1.0)
    theta = np.arccos(costh)          # inclination
    
    return Rhat, theta, phi



def ring_around_source(
        nx, ny, nz, dx, dy, dz,
        source,          # dict like {'mode':'xyz','value':(xs,ys,zs)} or {'mode':'ijk','value':(i,j,k)}
        radius, tol,     # in meters if physical metric; in index units if metric='index'
        origin=(0.0,0.0,0.0),
        metric="physical",
        return_angles=True
    ):
    """
    Build mesh, form distances, select a spherical shell around 'source' with a given radius±tol.
    Returns:
      - idx: (N,3) integer indices of selected ring nodes
      - xyz: (N,3) coordinates in meters of the selected nodes
      - D: (N,) distances (same units as radius)
      - (optional) Rhat, theta, phi
    """
    X, Y, Z = build_mesh(nx, ny, nz, dx, dy, dz, origin)
    if source["mode"] == "xyz":
        src_xyz = np.array(source["value"], dtype=float)
    elif source["mode"] == "ijk":
        src_xyz = index_to_source(source["value"], dx, dy, dz, origin)
    else:
        raise ValueError("source['mode'] must be 'xyz' or 'ijk'")
    
    D = distances_from_source(X, Y, Z, src_xyz, metric=metric, dx=dx, dy=dy, dz=dz)
    _, idx = select_shell_nodes(D, radius=radius, tol=tol)
    
    # gather outputs
    if idx.size == 0:
        xyz = np.empty((0,3))
        Dout = np.empty((0,))
        if return_angles:
            return idx, xyz, Dout, (np.empty((0,3)), np.empty((0,)), np.empty((0,)))
        return idx, xyz, Dout
    
    xyz = np.column_stack([X[idx[:,0], idx[:,1], idx[:,2]],
                           Y[idx[:,0], idx[:,1], idx[:,2]],
                           Z[idx[:,0], idx[:,1], idx[:,2]]])
    Dout = D[idx[:,0], idx[:,1], idx[:,2]]
    
    if return_angles:
        Rhat, theta, phi = directions_and_angles(X, Y, Z, src_xyz, idx)
        return idx, xyz, Dout, (Rhat, theta, phi)
    else:
        return idx, xyz, Dout

idx, xyz, Dout = ring_around_source(
    dom.nx, dom.ny, dom.nz, dom.dx, dom.dy, dom.dz,
    {'mode':'xyz','value':(seis.x,seis.y,seis.z)},
    18
)

# ------------------------------------------------------------------------------


def estimate_arrivals(dom, rec_idx, source_idx, C6, rho, fc, cycles=3):
    """
    For each receiver: use ray direction from source to receiver and Christoffel
    phase speeds to predict arrival times. Then build global windows for P,S1,S2.
    """
    dx = dom["dx"]
    src_xyz = np.array(dom["idx_to_xyz"](*source_idx))
    # distances & directions
    Rxyz = np.array([dom["idx_to_xyz"](*tuple(i)) for i in rec_idx])
    dvec = Rxyz - src_xyz
    dist = np.linalg.norm(dvec, axis=1)
    nhat = dvec / dist[:,None]
    
    # speeds per receiver (phase)
    speeds = []
    for n in nhat:
        v, _ = christoffel(C6, rho, n)
        speeds.append(v)  # [vP,vS1,vS2]
    speeds = np.array(speeds)                  # (M,3)
    t_arr = dist[:,None] / speeds              # (M,3)
    
    # global (array-wide) windows centered at earliest arrivals
    # window half-width = cycles/fc
    hw = cycles / fc
    tP0  = np.min(t_arr[:,0])
    tS10 = np.min(t_arr[:,1])
    tS20 = np.min(t_arr[:,2])
    
    winP  = (tP0  - hw, tP0  + hw)
    winS1 = (tS10 - hw, tS10 + hw)
    winS2 = (tS20 - hw, tS20 + hw)
    return t_arr, (winP, winS1, winS2)


def stereonet_grid(theta_step_deg=2.0, phi_step_deg=2.0, lower_hemisphere=True):
    th = np.deg2rad(np.arange(0, 90+1e-6, theta_step_deg) if lower_hemisphere else
                    np.arange(0, 180+1e-6, theta_step_deg))
    ph = np.deg2rad(np.arange(0, 360, phi_step_deg))
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    U = np.column_stack([np.sin(TH).ravel()*np.cos(PH).ravel(),
                         np.sin(TH).ravel()*np.sin(PH).ravel(),
                         np.cos(TH).ravel()])
    return U, TH.shape  # directions, and grid shape

def beamform_power(vx, vy, vz, dt, rec_idx, dom, C6, rho, band, win, mode='P', project_polarization=True):
    """
    Returns P_hat (power) on the stereonet grid (no plotting).
    """
    f_lo, f_hi = band
    U, grid_shape = stereonet_grid()
    dx = dom["dx"]
    x0 = np.array(dom["center_xyz"])
    
    # Assemble traces at receivers
    # Expect vx,vy,vz as full arrays [t,i,j,k]; otherwise adapt a loader.
    traces = []
    for (i,j,k) in rec_idx:
        traces.append(np.column_stack([vx[:,i,j,k], vy[:,i,j,k], vz[:,i,j,k]]))  # [T,3]
    traces = np.array(traces, dtype=float)  # [M,T,3]
    M, T, _ = traces.shape
    
    # Time window
    t_lo, t_hi = win
    i0 = max(int(np.floor(t_lo/dt)), 0)
    i1 = min(int(np.ceil(t_hi/dt)), T)
    tr_win = traces[:, i0:i1, :]   # [M, Tw, 3]
    Tw = tr_win.shape[1]
    
    # FFT along time
    F = np.fft.rfftfreq(Tw, dt)
    mask = (F >= f_lo) & (F <= f_hi)
    X = np.fft.rfft(tr_win, axis=1)  # [M, Fw_all, 3]
    X = X[:, mask, :]                 # [M, Fw, 3]
    Fw = F[mask]
    
    P = np.zeros(U.shape[0], dtype=float)
    
    # Precompute receiver vectors
    Rxyz = np.array([dom["idx_to_xyz"](*tuple(i)) for i in rec_idx])
    rvec = Rxyz - x0  # [M,3]
    
    for ui, u in enumerate(U):
        vph, POL = christoffel(C6, rho, u)  # speeds and pols for this direction
        if   mode == 'P':   v_use, e = vph[0], POL[:,0]
        elif mode == 'S1':  v_use, e = vph[1], POL[:,1]
        elif mode == 'S2':  v_use, e = vph[2], POL[:,2]
        else: raise ValueError("mode must be 'P','S1','S2'")
        
        # optional polarization projection
        if project_polarization:
            # project along e for each receiver
            # X has per-receiver vector components; form scalar by dot
            # (broadcast complex multiply): conj(e)·X_m
            e_col = e.reshape(1,1,3)
            Xs = np.sum(X * e_col.conj(), axis=2)  # [M, Fw]
        else:
            # use a single component, e.g., vx: X[:,:,0]
            Xs = X[:,:,0]
        
        # delays (plane-wave about shell center)
        tau = (rvec @ u) / v_use  # [M]
        phase = np.exp(1j * 2*np.pi * Fw.reshape(1,-1) * tau.reshape(-1,1))  # [M,Fw]
        
        Y = np.sum(Xs * phase, axis=0)  # steered sum per frequency [Fw]
        P[ui] = np.sum(np.abs(Y)**2)    # broadband power
    return P.reshape(grid_shape)


def voigt_to_Cijkl(C6):
    """
    Minimal, dense expansion to C_{ijkl}. Assumes engineering shear in Voigt.
    """
    C = np.zeros((3,3,3,3))
    # Voigt map (ij)->I
    V = {(0,0):0,(1,1):1,(2,2):2,(1,2):3,(0,2):4,(0,1):5}
    for i in range(3):
        for j in range(3):
            I = V[(i,j) if i<=j else (j,i)]
            for k in range(3):
                for l in range(3):
                    J = V[(k,l) if k<=l else (l,k)]
                    # engineering shear normalization
                    fI = 1.0 if I<3 else 1/np.sqrt(2)
                    fJ = 1.0 if J<3 else 1/np.sqrt(2)
                    C[i,j,k,l] = C6[I,J] * fI * fJ
    return C


def christoffel(C6, rho, nhat):
    """
    Returns phase speeds and polarizations for a unit direction nhat (3,).
    """
    C = voigt_to_Cijkl(C6)
    G = np.zeros((3,3))
    n = nhat/np.linalg.norm(nhat)
    for i in range(3):
        for j in range(3):
            # Gamma_ij = C_{ikjl} n_k n_l / rho
            G[i,j] = np.einsum('kl,ikjl->', np.outer(n,n), C[i,:,j,:]) / rho
    
    w, V = eig(G)  # eigenvalues (v^2), eigenvectors (polarizations)
    v = np.sqrt(np.real(w))
    # sort P (largest), S1, S2
    order = np.argsort(v)[::-1]
    v = np.real(v[order])
    V = np.real(V[:,order])  # columns are eigenvectors
    return v, V  # v=(vP, vS1, vS2), V[:,0]=eP, etc.


def run_pipeline(vx, vy, vz, dt, C6, rho, fc, dom_params):
    dom = build_domain(**dom_params)
    # shell radius: with half-core ≈ 17.5 m, use 14 m to keep margin from core boundary
    rec_idx = make_shell_receivers(dom, radius_m=14.0, n_pts=128, keep_lower_hemisphere=True)
    
    # source: given in meters (35,35,35); convert to full-grid indices
    src_idx = dom["xyz_to_idx"](35.0, 35.0, 35.0)
    
    # arrivals + windows
    t_arr, (winP, winS1, winS2) = estimate_arrivals(dom, rec_idx, src_idx, C6, rho, fc, cycles=3)
    
    # pick representative receiver (for quick QA, logging, etc.)
    rep_idx = choose_representative_receiver(vx, vy, vz, dt, rec_idx, winP, band=(0.6*fc, 1.4*fc))
    
    # beamforming maps (arrays ready for plotting later)
    band = (0.6*fc, 1.4*fc)
    Pmap = beamform_power(vx, vy, vz, dt, rec_idx, dom, C6, rho, band, winP,  mode='P',  project_polarization=True)
    S1map= beamform_power(vx, vy, vz, dt, rec_idx, dom, C6, rho, band, winS1, mode='S1', project_polarization=True)
    S2map= beamform_power(vx, vy, vz, dt, rec_idx, dom, C6, rho, band, winS2, mode='S2', project_polarization=True)
    
    return {
        "rec_idx": rec_idx,
        "rep_idx": rep_idx,
        "t_arrivals": t_arr,         # per-receiver predicted [M,3]
        "winP": winP, "winS1": winS1, "winS2": winS2,
        "Pmap": Pmap, "S1map": S1map, "S2map": S2map
    }
