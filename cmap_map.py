"""
Obtained from http://wiki.scipy.org/Cookbook/Matplotlib/ColormapTransformations

"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import sys


def cmap_map_p3(function, cmap):
    """
    Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous
    points in a colormap.
    Revised 08/03/2015 - Made Python 3 compliant
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # First get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        length = np.shape(cdict[key])[0]
        step_dict[key] = np.zeros(length)
        for index in np.arange(length):
            step_dict[key][index] = cdict[key][index][0]
    step_list = np.array([step_dict['red'], step_dict['blue'],
                          step_dict['green']]).ravel()
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    old_LUT = []
    new_LUT = []
    for index in np.arange(len(step_list)):
        elements = cmap(step_list[index])[0:3]
        old_LUT.append(elements)
        new_LUT.append(list(function(np.array(elements))))
    old_LUT = np.array(old_LUT)
    new_LUT = np.array(new_LUT)
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(('red', 'green', 'blue')):
        this_colorvector = []
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_colorvector.append([step, new_LUT[j, i], new_LUT[j, i]])
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_colorvector.append([step, new_LUT[j, i], new_LUT[j, i]])
        cdict[key] = this_colorvector
    return LinearSegmentedColormap('colormap', cdict, 1024)


def cmap_map_p2(function, cmap):
    """
    Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous
    points in a colormap.
    Python 2 compliant only
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # First get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    # reduced_cmap = lambda step: np.array(cmap(step)[0:3])

    def reduced_cmap(step): return np.array(cmap(step)[0:3])
    old_LUT = np.array(map(reduced_cmap, step_list))
    new_LUT = np.array(map(function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(('red', 'green', 'blue')):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector
    return LinearSegmentedColormap('colormap', cdict, 1024)


def lighten_cmap(cmap):
    if sys.version_info < (3, 0, 0):
        return cmap_map_p2(lambda x: x/2+0.5, cmap)
    else:
        return cmap_map_p3(lambda x: x/2+0.5, cmap)
