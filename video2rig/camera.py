"""A view's projection: an affine (orthographic) map A, b, or the adapter's own project(X, jac) / depth(X)
(the troll's front panel is a pinhole camera: Seedance drew it with perspective)."""
import numpy as np


def project(view, X, jac=False):
    if 'project' in view:
        return view['project'](X, jac)
    pix = X @ view['A'].T + view['b']
    if jac:
        return pix, np.broadcast_to(view['A'], (len(X), 2, 3))
    return pix


def depth(view, X):
    if 'depth' in view:
        return view['depth'](X)
    return (X - view['location']) @ view['forward']
