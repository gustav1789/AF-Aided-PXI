import numpy as np
import plotly.graph_objects as go
import healpy as hp

# Lets try to create an explosion pattern using our example data from unit_displacment
# Below are 2 function that have different usecases, as well as a "detector" class

class Detector:
    def __init__(self, plane_size,detector_distance,num_bins,sigma=0,axis=1,detector_eff=1.0):
        self.plane_size = plane_size
        self.distance = detector_distance
        self.num_bins = num_bins
        self.sigma = sigma
        self.axis = axis
        self.detector_eff = detector_eff
        
    def __repr__(self):
        return (f"DetectorParameters(plane_size={self.plane_size}, "
                f"detector_distance={self.distance}, "
                f"num_bins={self.num_bins}, "
                f"sigma={self.sigma}, "
                f"axis={self.axis}, "
                f"detector_eff={self.detector_eff})")
    
import healpy as hp

def bin_plane_xaxis(xyz, detector, mass=None, mass_filter=None):
    """
    Bin unit vectors onto a detector plane at x = +L or x = -L.

    Parameters:
    xyz (numpy.ndarray): (N, 3) array of [x, y, z] coordinates.
    detector (Detector): Detector object containing plane_size, detector_distance, num_bins, and detector_eff.
    mass (numpy.ndarray, optional): Array of masses corresponding to the xyz coordinates.
    mass_filter (tuple, optional): (min_mass, max_mass) to filter the xyz coordinates by mass.

    Returns:
    bins (numpy.ndarray): (num_bins, num_bins) histogram of binned hits.
    hit_coords (numpy.ndarray): (N_hits, 3) array of [x_plane, y_bin_center, z_bin_center] for each hit.
    """
    # Optionally filter by mass
    if mass is not None and mass_filter is not None:
        mass = np.array(mass)
        idx = (mass > mass_filter[0]) & (mass < mass_filter[1])
        xyz = xyz[idx]

    # Could add filter by distance to origin. Something like:
    # min_distance = 
    # xyz = xyz[np.linalg.norm(xyz[:]) > min_distance]
    # Not applicable here though since distances are normalized

    

    # Only keep vectors pointing toward the detector
    L = detector.distance
    if L > 0:
        xyz = xyz[xyz[:, 0] > 0]
    else:
        xyz = xyz[xyz[:, 0] < 0]

    if xyz.shape[0] == 0:
        return np.zeros((detector.num_bins, detector.num_bins), dtype=int), np.empty((0, 3))

    # Project to detector plane at x = L (or -L)
    k = L / xyz[:, 0]
    hits = xyz * k[:, np.newaxis]  # shape (N, 3)
    y = hits[:, 1]
    z = hits[:, 2]

    # Bin edges and centers
    edges = np.linspace(-detector.plane_size/2, detector.plane_size/2, detector.num_bins+1)
    centers = (edges[:-1] + edges[1:]) / 2

    # Digitize y and z to bin indices
    y_idx = np.digitize(y, edges) - 1
    z_idx = np.digitize(z, edges) - 1

    # Mask for hits inside the detector area
    mask = (y_idx >= 0) & (y_idx < detector.num_bins) & (z_idx >= 0) & (z_idx < detector.num_bins)
    y_idx = y_idx[mask]
    z_idx = z_idx[mask]

    # Detector efficiency: randomly keep a fraction of hits
    N = len(y_idx)
    M = int(detector.detector_eff * N)
    keep = np.random.choice(N, M, replace=False)
    y_idx = y_idx[keep]
    z_idx = z_idx[keep]

    # Fill histogram
    bins = np.zeros((detector.num_bins, detector.num_bins), dtype=int)
    np.add.at(bins, (y_idx, z_idx), 1)

    # Return coordinates of each binned hit (center of bin)
    hit_coords = np.stack([np.full_like(y_idx, L), centers[y_idx], centers[z_idx]], axis=1)

    return bins, hit_coords

def build_healpix_full(r_vectors, nside=32, mass=None, mass_filter=None, distance_filter=None, nest=False):
    """
    Bin rays (vectors) into a full-sky HEALPix map (whole 4π solid angle).
    Returns (hpx_map, mask) where mask is True for all pixels.

    r_vectors : (N,3) array-like of ray vectors (need not be unit length).
    nside     : healpy nside
    nest      : healpy nest ordering flag
    """

    # Optionally filter by mass (Added by Ivan)
    if mass is not None and mass_filter is not None:
        mass = np.array(mass)
        idx = (mass > mass_filter[0]) & (mass < mass_filter[1])
        r_vectors = r_vectors[idx]

    r = np.asarray(r_vectors)

    # Optionally filter by distance (Added by Ivan)
    if distance_filter is not None:
        idx = (np.linalg.norm(r, axis = 1) > distance_filter[0]) & (np.linalg.norm(r, axis=1) < distance_filter[1])
        r = r[idx]
    # Could try filter by velocity

    npix = hp.nside2npix(nside)

    # handle empty input
    if r.size == 0 or r.shape[0] == 0:
        return np.zeros(npix, dtype=float), np.ones(npix, dtype=bool)

    # normalize, skip zero-length vectors
    norms = np.linalg.norm(r, axis=1)
    valid = norms > 0
    if not np.all(valid):
        r = r[valid]
        norms = norms[valid]

    r_unit = r / norms[:, None]

    # spherical angles (healpy expects theta = colatitude)
    theta = np.arccos(np.clip(r_unit[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(r_unit[:, 1], r_unit[:, 0]), 2*np.pi)

    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    counts = np.bincount(pix, minlength=npix).astype(float)

    hpx_map = counts
    mask = np.ones(npix, dtype=bool)

    return hpx_map, mask


def plot_healpix_3d_interactive(hpx_map, nside=16):
    """
    Skapar en interaktiv, snurrbar 3D-projektion i webbläsaren.
    """
    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    
    active_idx = hpx_map > 0
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x[active_idx],
        y=y[active_idx],
        z=z[active_idx],
        mode='markers',
        marker=dict(
            size=5,
            color=hpx_map[active_idx], # Färg baserat på värde
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Count")
        )
    )])

    # Ta bort bakgrund och axlar för en ren "rymdkänsla"
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title=dict(text="Interactive 3D Protein Projection", font=dict(color="white"))
    )
    
    return fig


# This is just a helper function
def invert_H_to_r(H, L, side='+'):
    hy = H[:, 1]
    hz = H[:, 2]
    norm = np.sqrt(1 + (hy / L)**2 + (hz / L)**2)
    if side == '+':
        r = np.stack([
            1 / norm,
            hy / (L * norm),
            hz / (L * norm)
        ], axis=1)
    elif side == '-':
        r = np.stack([
            -1 / norm,
            hy / (L * norm),
            hz / (L * norm)
        ], axis=1)
    else:
        raise ValueError("side must be '+' or '-'")
    return r

# lets also define a 2D sum pool function to downsample our binned plane data, this is just a helper function and not strictly necessary for the tutorial
def sum_pool_2d(arr, pool_size):
    """
    Apply 2D sum pooling to a 2D array.

    arr: 2D numpy array to be pooled
    pool_size: size of the pooling window (e.g. 2 for 2x2 pooling)

    Returns:
    pooled_arr: 2D numpy array after sum pooling
    """
    m, n = arr.shape
    new_m = m // pool_size
    new_n = n // pool_size
    pooled_arr = np.zeros((new_m, new_n), dtype=arr.dtype)

    for i in range(new_m):
        for j in range(new_n):
            window = arr[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            pooled_arr[i, j] = np.sum(window)

    return pooled_arr

# Lets also define a gaussian filter function to smooth our data out
def gaussian_filter_2d(arr, sigma):
    """
    Apply a 2D Gaussian filter to a 2D array.

    arr: 2D numpy array to be filtered
    sigma: standard deviation of the Gaussian kernel

    Returns:
    filtered_arr: 2D numpy array after Gaussian filtering
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)

