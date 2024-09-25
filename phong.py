from myshapes import Sphere
import numpy as np

def diffuse(p, closest_obj, lhat, s):
    mdiff = closest_obj.getDiffuse()

    if isinstance(closest_obj, Sphere):
        n = closest_obj.getNormal(p)
    else:
        n = closest_obj.getNormal()

    nhat = n / np.linalg.norm(n)
    dot_product = np.sum(nhat * lhat)

    if dot_product < 0:
        dot_product = 0

    cdiff = (s * mdiff) * dot_product
    return cdiff, nhat

def specular(vhat, nhat, closest_obj, lhat, s):
    mspec = closest_obj.getSpecular()
    mgls = closest_obj.getGloss()
    rhat = 2*np.sum(lhat*nhat)*nhat - lhat
    dot_product = np.sum(vhat*rhat)
    if dot_product < 0:
        dot_product = 0

    cspec = (s*mspec)*dot_product**mgls
    return cspec

def final_color(cspec, cdiff, crefl, closest_obj, camb):
    kd = closest_obj.getKd()
    ks = closest_obj.getKs()
    ka = closest_obj.getKa()
    kr = closest_obj.getRefl()
    return kd*cdiff + ks*cspec + ka*camb + kr*crefl

