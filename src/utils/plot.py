import matplotlib.pyplot as plt
import numpy as np
from utils import s3w as s3w_utils, misc as utils

theta = np.linspace(0, 2 * np.pi, 2 * 100)
phi = np.linspace(0, np.pi, 100)
tp = np.array(np.meshgrid(theta, phi, indexing='ij')).transpose([1, 2, 0]).reshape(-1, 2)

def spherical_to_cartesian(sphere_coords):
    """
    Converts spherical coordinates to Cartesian coordinates.

    Parameters:
    - sphere_coords (np.ndarray): Array of spherical coordinates (theta, phi).

    Returns:
    - np.ndarray: Cartesian coordinates.
    """
    theta, phi = sphere_coords.T
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def plot_3d_sphere(ax):
    """
    Plots a 3D sphere on the given axis with meridians and parallels.

    Parameters:
    - ax: Matplotlib 3D axis.
    - n_meridians (int, optional): Number of meridians.
    - n_parallels (int, optional): Number of parallels.
    """
    n_meridians=50
    n_parallels = 100
    
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_parallels*1j]
    x, y, z = spherical_to_cartesian(np.column_stack((u.ravel(), v.ravel()))).T
    x = x.reshape(u.shape)
    y = y.reshape(u.shape)
    z = z.reshape(u.shape)

    ax.plot_surface(x, y, z, color='gray', alpha=0.2)
    ax.plot_wireframe(x, y, z, color="black", alpha=0.1, lw=1)


def plot_scatter_3d(ax, datasets, labels):
    """
    Plots scatter points in 3D space on the given axis.

    Parameters:
    - ax: Matplotlib 3D axis.
    - datasets (list of np.ndarray): List of numpy arrays for plotting.
    - labels (list of str): Corresponding labels for the datasets.
    """
    for data, label in zip(datasets, labels):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=3, label=label)

def plot_all(datasets, labels, view_init=(25, 25)):
    """
    Plots multiple distributions on the 3D sphere and their stereographic projections in 2D.

    Parameters:
    - datasets (list of np.ndarray): A list of numpy arrays; each represents a distribution.
    - labels (list of str): A list of labels corresponding to each dataset.
    - view_init (tuple, optional): Tuple of two ints for 3D plot view initialization.
    - n_meridians (int, optional): Number of meridians.
    - n_parallels (int, optional): Number of parallels.
    """

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    plot_scatter_3d(ax1, datasets, labels)
    plot_3d_sphere(ax1)  
    ax1.view_init(*view_init)
    ax1.legend()

    ax2 = fig.add_subplot(122)
    plot_stereo(ax2, datasets, labels)

    plt.show()

def plot_stereo(ax, datasets, labels):
    """
    Plots stereographic projections of datasets on a 2D plane.

    Parameters:
    - ax: Matplotlib 2D axis object.
    - datasets (list of np.ndarray): List of numpy arrays for plotting.
    - labels (list of str): Corresponding labels for the datasets.
    - n_meridians (int): Number of meridians.
    - n_parallels (int): Number of parallels.
    """
    all_projs = []
    for data in datasets:
        proj = s3w_utils.get_stereo_proj(data)
        all_projs.append(proj)
        ax.scatter(proj[:, 0], proj[:, 1], s=3)

    all_projs = np.concatenate(all_projs, axis=0)
    max_extent = np.max(np.abs(all_projs)) * 0.5
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    
    n_meridians = 50
    n_parallels = max(int(max_extent * 10), 20)

    for i in range(n_meridians):
        meridian_phi = i * 2 * np.pi / n_meridians
        meridian = spherical_to_cartesian(np.array([[theta, meridian_phi] for theta in np.linspace(0, np.pi, 100)]))
        proj_meridian = s3w_utils.get_stereo_proj(meridian)
        ax.plot(proj_meridian[:, 0], proj_meridian[:, 1], color='black', alpha=0.7, lw=0.1)

    for j in range(n_parallels):
        parallel_theta = j * np.pi / n_parallels
        parallel = spherical_to_cartesian(np.array([[parallel_theta, phi] for phi in np.linspace(0, 2 * np.pi, 100)]))
        proj_parallel = s3w_utils.get_stereo_proj(parallel)
        ax.plot(proj_parallel[:, 0], proj_parallel[:, 1], color='black', alpha=0.7, lw=0.1)

    ax.legend(labels)

# Mollweide
def _plot_mollweide(heatmap):
    '''
    https://github.com/clbonet/Spherical_Sliced-Wasserstein/blob/main/lib/utils_sphere.py
    '''
    tt, pp = np.meshgrid(theta - np.pi, phi - np.pi / 2, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.pcolormesh(tt, pp, heatmap, cmap=plt.cm.jet)
    ax.set_axis_off()
    plt.show()

def plot_target_density(target_fn):
    '''
    https://github.com/clbonet/Spherical_Sliced-Wasserstein/blob/main/lib/utils_sphere.py
    '''
    density = target_fn(spherical_to_cartesian(tp))
    heatmap = density.reshape(2 * 100, 100)
    _plot_mollweide(heatmap)

def scatter_mollweide(X_target, target_fn):
    '''
    https://github.com/clbonet/Spherical_Sliced-Wasserstein/blob/main/lib/utils_sphere.py
    '''
    density = target_fn(spherical_to_cartesian(tp))
    heatmap = density.reshape(2 * 100, 100)
    tt, pp = np.meshgrid(theta - np.pi, phi - np.pi / 2, indexing='ij')
    spherical_coords = utils.euclidean_to_spherical(X_target)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.pcolormesh(tt, pp, heatmap, cmap=plt.cm.jet)
    ax.scatter(spherical_coords[:,0] - np.pi, spherical_coords[:,1] - np.pi/2)
    ax.set_axis_off()
    plt.show()

def projection_mollweide(target_fn, ax, vmax=None):
    '''
    https://github.com/clbonet/Spherical_Sliced-Wasserstein/blob/main/lib/utils_sphere.py
    '''
    density = target_fn(spherical_to_cartesian(tp))
    heatmap = density.reshape(2 * 100, 100)
    tt, pp = np.meshgrid(theta - np.pi, phi - np.pi / 2, indexing='ij')
    ax.pcolormesh(tt, pp, heatmap, cmap=plt.cm.jet, vmax=vmax)
    ax.set_axis_off()

def scatter_mollweide_ax(X_target, ax):
    '''
    https://github.com/clbonet/Spherical_Sliced-Wasserstein/blob/main/lib/utils_sphere.py
    '''
    spherical_coords = utils.euclidean_to_spherical(X_target)
    ax.scatter(spherical_coords[:,0] - np.pi, spherical_coords[:,1] - np.pi/2)
    ax.set_axis_off()
