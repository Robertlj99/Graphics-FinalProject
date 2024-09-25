import numpy as np
import matplotlib.pyplot as plt
from load import loadScene, generate_rays, generate_dist_rays
from phong import diffuse, specular, final_color

resh = 1024
resw = 1024
recursive_depth = 64

scene_fn = r'..\scenes\my_scene.json'
image_fn = r'..\images\my_scene-512-16.png'

camera, light, objects = loadScene(scene_fn)
rdmatrix = generate_rays(camera, resw, resh)
#dist_rdmatrix = generate_dist_rays(camera, resw, resh)
mamb = light["AmbientLight"]
camb = light["AmbientLight"] * light["LightColor"]
r_0 = camera["LookFrom"]
l = light["DirectionToLight"]
lhat = l / np.linalg.norm(l)
s = light["LightColor"]

def cast_ray(r0, rd, closest_obj=None):
    # Covers the initial casting of the rays
    if closest_obj is None:
        # Initialize t_min to -1
        tmin = -1
        # Boolean to set first t-min
        first = True
        # Closest object tracker
        closest = [-1]
        # Loop through[] objects in scene
        for obj in objects:
            # Get t-value
            t = obj.intersect(r0, rd)
            # Check t
            if t > 0:
                if first:
                    tmin = t
                    closest.append(obj)
                    first = False
                elif t < tmin:
                    tmin = t
                    closest.append(obj)

        return tmin, closest[-1]

    # Covers recasting of the rays (shadow & reflection)
    else:
        # Initialize t_min to -1
        tmin = -1
        # Boolean to set first t-min
        first = True
        # Closest object tracker
        closest = [-1]
        # Exclude object casting ray from object iteration
        objs = [x for x in objects if x != closest_obj]
        # Loop through rest of objects in scene
        for obj in objs:
            # Get t-value
            t = obj.intersect(r0, rd)
            # Check t
            if t > 0:
                if first:
                    tmin = t
                    closest.append(obj)
                    first = False
                elif t < tmin:
                    tmin = t
                    closest.append(obj)

        return tmin, closest[-1]

# This function recursively traces reflection rays, at least I think
def reflection_rays(r_d, p, closest_obj, depth):
    v = -r_d
    vhat = v / np.linalg.norm(v)
    cdiff, nhat = diffuse(p, closest_obj, lhat, s)
    cspec = specular(vhat, nhat, closest_obj, lhat, s)
    rehat = 2 * np.sum(nhat * vhat) * nhat - vhat  # reflection angle
    t_min, temp = cast_ray(p, rehat, closest_obj)
    if t_min > 0:
        if depth > 0:
            depth = depth - 1
            return final_color(cspec, cdiff, reflection_rays(r_d, t_min, temp, depth), closest_obj, camb)
        else:
            print('got here')
            return final_color(cspec, cdiff, temp.getDiffuse(), closest_obj, camb)
    else:
        return final_color(cspec, cdiff, light["BackgroundColor"], closest_obj, camb)

def generate_image():
    pixel_values = np.empty((resw, resh, 3), dtype=np.float32)
    i = resw - 1
    j = 0
    for row in rdmatrix:
        for r_d in row:
            # Cast the initial rays
            t_min, closest_obj = cast_ray(r_0, r_d)
            if t_min < 0:
                # The ray doesn't intersect with any objects
                pixel_values[i][j] = light["BackgroundColor"]
            else:
                # The ray does intersect with an object
                p = r_0 + r_d * t_min
                # Cast shadow ray
                t_min, temp = cast_ray(p, lhat, closest_obj)
                if t_min > 0:
                    # Point is in a shadow
                    pixel_values[i][j] = light["AmbientLight"]
                else:
                    # Preform recursive reflection testing
                    pixel_values[i][j] = reflection_rays(r_d, p, closest_obj, depth=recursive_depth)
            j = j + 1
        i = i - 1
        j = 0

    return pixel_values

# My attempt at distributed sampling, this currently does not work
def generate_dist_image():
    pixel_values = np.empty((resw, resh, 4, 3), dtype=np.float32)
    i = resw - 1
    j = 0
    k = 0
    for column in dist_rdmatrix:
        for row in column:
            for r_d in row:
                # Cast the initial rays
                t_min, closest_obj = cast_ray(r_0, r_d)
                if t_min < 0:
                    # The ray doesn't intersect with any objects
                    pixel_values[i][j][k] = light["BackgroundColor"]
                else:
                    # The ray does intersect with an object
                    p = r_0 + r_d * t_min
                    # Cast shadow ray
                    t_min, temp = cast_ray(p, lhat, closest_obj)
                    if t_min > 0:
                        # Point is in a shadow
                        pixel_values[i][j][k] = light["AmbientLight"]
                    else:
                        # Preform recursive reflection testing
                        pixel_values[i][j][k] = reflection_rays(r_d, p, closest_obj, depth=recursive_depth)
                k = k + 1
            k = 0
            j = j + 1
        i = i - 1
        j = 0

    return pixel_values


if __name__ == "__main__":
    # No need to do anything here
    image = generate_image()
    #image = generate_dist_image()
    plt.imsave(image_fn, image)