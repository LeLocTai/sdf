"""Provides a function for performing 3D Dual Countouring"""

import numpy as np
from .utils_3d import Quad
from .qef import solve_qef_3d


def adapt(v0, v1):
    """v0 and v1 are numbers of opposite sign. This returns how far you need to interpolate from v0 to v1 to get to 0."""
    assert (v1 > 0) != (v0 > 0), "v0 and v1 do not have opposite sign"

    return (0 - v0) / (v1 - v0)


def dual_contour_3d_find_best_vertex(volume, normals, x, y, z):
    # Evaluate f at each corner
    v = volume[x:x+2, y:y+2, z:z+2]

    # For each edge, identify where there is a sign change.
    # There are 4 edges along each of the three axes
    changes = []
    for dx in (0, 1):
        for dy in (0, 1):
            if (v[dx, dy, 0] > 0) != (v[dx, dy, 1] > 0):
                changes.append(
                    np.array([x + dx, y + dy, z + adapt(v[dx, dy, 0], v[dx, dy, 1])]))

    for dx in (0, 1):
        for dz in (0, 1):
            if (v[dx, 0, dz] > 0) != (v[dx, 1, dz] > 0):
                changes.append(
                    np.array([x + dx, y + adapt(v[dx, 0, dz], v[dx, 1, dz]), z + dz]))

    for dy in (0, 1):
        for dz in (0, 1):
            if (v[0, dy, dz] > 0) != (v[1, dy, dz] > 0):
                changes.append(
                    np.array([x + adapt(v[0, dy, dz], v[1, dy, dz]), y + dy, z + dz]))

    if len(changes) <= 1:
        return None

    # For each sign change location v[i], we find the normal n[i].
    # The error term we are trying to minimize is sum( dot(x-v[i], n[i]) ^ 2)

    # In other words, minimize || A * x - b || ^2 where A and b are a matrix and vector
    # derived from v and n

    ceil_normals = []
    for v in changes:
        lower_vert = np.floor(v).astype('int')
        upper_vert = np.ceil(v).astype('int')

        lower_normal = normals[lower_vert[0], lower_vert[1], lower_vert[2]]
        upper_normal = normals[upper_vert[0], upper_vert[1], upper_vert[2]]

        offset = v - lower_vert
        scale = upper_vert - lower_vert
        t = np.divide(offset,
                      scale,
                      out=np.zeros_like(offset),
                      where=scale != 0)

        n = lower_normal * (1-t) + upper_normal * t

        ceil_normals.append(n)

    return solve_qef_3d(x, y, z, changes, ceil_normals)


def dual_contour(volume, normals):
    xmax = volume.shape[0] - 1
    ymax = volume.shape[1] - 1
    zmax = volume.shape[2] - 1

    # For each cell, find the the best vertex for fitting f
    vert_array = []
    vert_indices = {}
    for x in range(xmax):
        for y in range(ymax):
            for z in range(zmax):
                vert = dual_contour_3d_find_best_vertex(
                    volume, normals, x, y, z)
                if vert is None:
                    continue
                vert_array.append(vert)
                vert_indices[(x, y, z)] = len(vert_array) - 1

    # For each cell edge, emit an face between the center of the adjacent cells if it is a sign changing edge
    faces = []
    for x in range(xmax):
        for y in range(ymax):
            for z in range(zmax):
                if x > 0 and y > 0:
                    solid1 = volume[x, y, z + 0] > 0
                    solid2 = volume[x, y, z + 1] > 0
                    if solid1 != solid2:
                        faces.append(
                            Quad(
                                vert_indices[(x - 1, y - 1, z)],
                                vert_indices[(x - 0, y - 1, z)],
                                vert_indices[(x - 0, y - 0, z)],
                                vert_indices[(x - 1, y - 0, z)],
                            ).swap(solid2))
                if x > 0 and z > 0:
                    solid1 = volume[x, y + 0, z] > 0
                    solid2 = volume[x, y + 1, z] > 0
                    if solid1 != solid2:
                        faces.append(
                            Quad(
                                vert_indices[(x - 1, y, z - 1)],
                                vert_indices[(x - 0, y, z - 1)],
                                vert_indices[(x - 0, y, z - 0)],
                                vert_indices[(x - 1, y, z - 0)],
                            ).swap(solid1))
                if y > 0 and z > 0:
                    solid1 = volume[x + 0, y, z] > 0
                    solid2 = volume[x + 1, y, z] > 0
                    if solid1 != solid2:
                        faces.append(
                            Quad(
                                vert_indices[(x, y - 1, z - 1)],
                                vert_indices[(x, y - 0, z - 1)],
                                vert_indices[(x, y - 0, z - 0)],
                                vert_indices[(x, y - 1, z - 0)],
                            ).swap(solid2))

    vertices = np.empty((len(vert_array), 3), np.float32)
    for i in range(len(vert_array)):
        vertices[i, 0] = vert_array[i].x
        vertices[i, 1] = vert_array[i].y
        vertices[i, 2] = vert_array[i].z

    triangle_faces = np.empty((len(faces)*2, 3), np.int32)
    for i in range(len(faces)):
        quad = faces[i]
        triangle_faces[i*2+0] = np.array([quad.v1, quad.v2, quad.v3])
        triangle_faces[i*2+1] = np.array([quad.v1, quad.v3, quad.v4])

    return (vertices, triangle_faces)
