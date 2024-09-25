import numpy as np

class myshape:

	def __init__(self, diffuse, specular, gloss, refl, Kd, Ks, Ka):
		self.diffuse = np.array(diffuse, dtype=np.float32)
		self.specular = np.array(specular, dtype=np.float32)
		self.gloss = gloss
		self.refl = refl
		self.Kd = Kd
		self.Ks = Ks
		self.Ka = Ka

	def setDiffuse(self, r, g, b):
		self.diffuse = np.array([r,g,b], dtype=np.float32)
	
	def getDiffuse(self):
		return self.diffuse
		
	def setSpecular(self, r, g, b):
		self.specular = np.array([r,g,b], dtype=np.float32)
	
	def getSpecular(self):
		return self.specular
	
	def setGloss(self, c):
		self.gloss = c
		
	def getGloss(self):
		return self.gloss
	
	def setKd(self, c):
		self.Kd = c

	def getKd(self):
		return self.Kd
	
	def setKs(self, c):
		self.Ks = c

	def getKs(self):
		return self.Ks
	
	def setKa(self, c):
		self.Ka = c

	def getKa(self):
		return self.Ka
	
	def setRefl(self, c):
		self.refl = c

	def getRefl(self):
		return self.refl
	
	def intersect(self, ray_org, ray_dir):
		''' Returns the t value of the intersection point '''
		return -1

	def __repr__(self):
		return "Diffuse: " + str(self.diffuse) + "\n" +  \
				"Specular: " + str(self.specular) + "\n" + \
				"Gloss: " + str(self.gloss) + "\n" + \
				"Refl: " + str(self.refl) + "\n" +  \
				"Kd: " + str(self.Kd) + "\n" +  \
				"Ks: " + str(self.Ks) + "\n" + \
				"Ka: " + str(self.Ka) + "\n"
		
class Plane(myshape):

	def __init__(self, normal, d, diffuse, specular, gloss, refl, Kd, Ks, Ka):
		
		super().__init__(diffuse, specular, gloss, refl, Kd, Ks, Ka)
		self.normal = np.array(normal, dtype=np.float32)
		self.d = d

	def setNormal(self, normal):
		self.normal = normal

	def getNormal(self):
		return self.normal
	
	def setD(self, d):
		self.d = d

	def getD(self):
		return self.d
	
	def intersect(self, ray_org, ray_dir):
		denom = np.sum(self.normal*ray_dir)

		# Divide by zero check
		if abs(denom) < 1e-6:
			return -1
		
		# Normal Correction
		if denom > 0:
			return -1
			#self.normal = -self.normal

		t = (-1 * np.sum(self.normal*ray_org) + self.d)/denom

		return t

	def __repr__(self):
		return "Plane\n" + \
				"normal: " + str(self.normal) + "\n" +  \
				"d: " + str(self.d) + "\n" + \
				super().__repr__()



class Triangle(myshape):

	def __init__(self, a, b, c, diffuse, specular, gloss, refl, Kd, Ks, Ka):
		
		super().__init__(diffuse, specular, gloss, refl, Kd, Ks, Ka)
		self.a = np.array(a, dtype=np.float32)
		self.b = np.array(b, dtype=np.float32)
		self.c = np.array(c, dtype=np.float32)

	def setA(self, a):
		self.a = a

	def getA(self):
		return self.a
	
	def setB(self, b):
		self.b = b

	def getB(self):
		return self.b
	
	def setC(self, c):
		self.c = c

	def getC(self):
		return self.c
	
	def getNormal(self):
		normal = np.cross(self.b-self.a,self.c-self.a)
		return normal/np.linalg.norm(normal)

	def intersect(self, ray_org, ray_dir):
		# Moller-Trumbore Algorithm (Generated with CoPilot)
		edge1 = self.b - self.a
		edge2 = self.c - self.a
		h = np.cross(ray_dir, edge2)
		a = np.sum(edge1*h)
		
		if a > -1e-6 and a < 1e-6:
			return -1
		
		f = 1/a
		s = ray_org - self.a
		u = f*np.sum(s*h)

		if u < 0 or u > 1:
			return -1
		
		q = np.cross(s, edge1)
		v = f*np.sum(ray_dir*q)
		
		if v < 0 or u + v > 1:
			return -1
		
		t = f*np.sum(edge2*q)

		if t > 1e-6:
			return t
		else:
			return -1

	def __repr__(self):
		return "Triangle\n" + \
				"A: " + str(self.a) + "\n" +  \
				"B: " + str(self.b) + "\n" + \
				"C: " + str(self.c) + "\n" + \
				super().__repr__()


class Sphere(myshape):

	def __init__(self, center, radius, diffuse, specular, gloss, refl, Kd, Ks, Ka):
		
		super().__init__(diffuse, specular, gloss, refl, Kd, Ks, Ka)
		self.center = center
		self.radius = radius
	
	def getCenter(self):
		return self.center
	
	def setCenter(self, center):
		self.center = center
        
	def getRadius(self):
		return self.radius
	
	def setRadius(self, radius):
		self.radius = radius

	def getNormal(self, p):
		return (p-self.center)/self.radius

	def intersect(self, ray_org, ray_dir):
		
		# Quadratic Formula
		a = np.sum(ray_dir*ray_dir)
		b = 2*np.sum(ray_dir*(ray_org-self.center))
		c = np.sum((ray_org-self.center)*(ray_org-self.center)) - self.radius*self.radius

		discriminant = b*b - 4*a*c

		if discriminant < 0:
			return -1
		
		t1 = (-b + np.sqrt(discriminant))/(2*a)
		t2 = (-b - np.sqrt(discriminant))/(2*a)

		if t1 < 0 and t2 < 0:
			return -1
		elif t1 < 0:
			return t2
		elif t2 < 0:
			return t1
		else:
			return min(t1,t2)

	def __repr__(self):
		return "Sphere\n" + \
				"Center: " + str(self.center) + "\n" +  \
				"Radius: " + str(self.radius) + "\n" + \
				super().__repr__()
