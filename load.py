from myshapes import Sphere, Triangle, Plane
import json
import numpy as np
import random

def loadScene(scene_fn):
    with open(scene_fn) as f:
        data = json.load(f)

    spheres = []

    for sphere in data["Spheres"]:
        spheres.append(
            Sphere(sphere["Center"], sphere["Radius"],
                   sphere["Mdiff"], sphere["Mspec"], sphere["Mgls"], sphere["Refl"],
                   sphere["Kd"], sphere["Ks"], sphere["Ka"]))

    triangles = []

    for triangle in data["Triangles"]:
        triangles.append(
            Triangle(triangle["A"], triangle["B"], triangle["C"],
                     triangle["Mdiff"], triangle["Mspec"], triangle["Mgls"], triangle["Refl"],
                     triangle["Kd"], triangle["Ks"], triangle["Ka"]))

    planes = []

    for plane in data["Planes"]:
        planes.append(
            Plane(plane["Normal"], plane["Distance"],
                  plane["Mdiff"], plane["Mspec"], plane["Mgls"], plane["Refl"],
                  plane["Kd"], plane["Ks"], plane["Ka"]))

    objects = spheres + triangles + planes

    camera = {
        "LookAt": np.array(data["Camera"]["LookAt"], ),
        "LookFrom": np.array(data["Camera"]["LookFrom"]),
        "Up": np.array(data["Camera"]["Up"]),
        "FieldOfView": data["Camera"]["FieldOfView"]
    }

    light = {
        "DirectionToLight": np.array(data["Light"]["DirectionToLight"]),
        "LightColor": np.array(data["Light"]["LightColor"]),
        "AmbientLight": np.array(data["Light"]["AmbientLight"]),
        "BackgroundColor": np.array(data["Light"]["BackgroundColor"]),
    }

    return camera, light, objects

def generate_rays(camera, resw, resh):

    r_0 = camera["LookFrom"]
    pat = camera["LookAt"]
    up = camera["Up"]
    fov = camera["FieldOfView"]

    # Gram-Schmidt
    e3 = (pat - r_0) / np.linalg.norm(pat - r_0)
    e1 = np.cross(e3, up) / np.linalg.norm(np.cross(e3, up))
    e2 = np.cross(e1, e3) / np.linalg.norm(np.cross(e1, e3))

    # Window calculations
    fovx, fovy = np.deg2rad(fov), np.deg2rad(fov)
    dist = np.linalg.norm(pat - r_0)
    u_max = dist * np.tan(fovx/2)
    v_max = dist * np.tan(fovy/2)
    u_min = -u_max
    v_min = -v_max

    # Pixel distances
    dist_u = (u_max - u_min) / (resw + 1)
    dist_v = (v_max - v_min) / (resh + 1)

    # Create empty matrix to store ray direction values
    rd_matrix = np.empty((resw, resh, 3), dtype=np.float32)

    # Populate matrix
    w_range = int(resw/2)
    h_range = int(resh/2)
    for i in range(-w_range, w_range):
        for j in range(-h_range, h_range):
            # S
            s = pat + (dist_u * (j + 0.5) * e1) + (dist_v * (i + 0.5) * e2)
            # Ray distance
            rd = (s - r_0) / np.linalg.norm(s - r_0)
            rd_matrix[i + w_range][j + h_range] = rd
    return rd_matrix

# Attempt at distributed sampling, not currently working
def generate_dist_rays(camera, resw, resh):

    r_0 = camera["LookFrom"]
    pat = camera["LookAt"]
    up = camera["Up"]
    fov = camera["FieldOfView"]

    # Gram-Schmidt
    e3 = (pat - r_0) / np.linalg.norm(pat - r_0)
    e1 = np.cross(e3, up) / np.linalg.norm(np.cross(e3, up))
    e2 = np.cross(e1, e3) / np.linalg.norm(np.cross(e1, e3))

    # Window calculations
    fovx, fovy = np.deg2rad(fov), np.deg2rad(fov)
    dist = np.linalg.norm(pat - r_0)
    u_max = dist * np.tan(fovx/2)
    v_max = dist * np.tan(fovy/2)
    u_min = -u_max
    v_min = -v_max

    # Pixel distances
    dist_u = (u_max - u_min) / (resw + 1)
    dist_v = (v_max - v_min) / (resh + 1)

    # Create empty matrix to store ray direction values
    rd_matrix = np.empty((resw, resh, 4, 3), dtype=np.float32)

    # Populate matrix
    w_range = int(resw/2)
    h_range = int(resh/2)
    for i in range(-w_range, w_range):
        for j in range(-h_range, h_range):
            # S
            s = pat + (dist_u * (j + 0.5) * e1) + (dist_v * (i + 0.5) * e2)
            # Ray distance
            rd = (s - r_0) / np.linalg.norm(s - r_0)
            # Top Left
            rd_matrix[i + w_range][j + h_range][0] = rd + np.array([random.uniform(0.0, (dist_u/2.0)), random.uniform(-(dist_v/2.0), 0.0), 0.0])
            # Top Right
            rd_matrix[i + w_range][j + h_range][1] = rd + np.array([random.uniform(0.0, (dist_u/2.0)), random.uniform(0.0, (dist_v / 2)), 0.0])
            # Bottom Left
            rd_matrix[i + w_range][j + h_range][2] = rd + np.array([random.uniform(-(dist_u / 2.0), 0.0), random.uniform(-(dist_v/2.0), 0.0), 0.0])
            # Bottom Right
            rd_matrix[i + w_range][j + h_range][3] = rd + np.array([random.uniform(-(dist_u / 2.0), 0.0), random.uniform(0.0, (dist_v / 2)), 0.0])
    return rd_matrix



