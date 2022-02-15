import numpy as np
from stl import mesh
import cv2 as cv

max_size=(150,150)
max_height=1.5
min_height=0


img = cv.imread('Imagenes_PhotoBox/Instantanea.jpg')
img = cv.resize(img,max_size,cv.INTER_AREA)

img_g = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

img_g = cv.bitwise_not(img_g)
cv.imshow('gray',img_g)
cv.waitKey(0)
imageNp = np.array(img_g)
maxPix=imageNp.max()
minPix=imageNp.min()

(W,H)=img_g.shape

print(f"W,H={W},{H}")
vertices=[]
faces=[]

DX=1
DY=1

for Y in range(0, H, DY):
    for X in range(0,W,DX):
        pixelIntensity = imageNp[Y][X]
        Z = (pixelIntensity * max_height) / maxPix
        vertices.append((X,Y,Z))

vert = np.array(vertices)
        
for X in range(0, W-1, DX):
    for Y in range(0, H-1, DY):
        face_v1= X+Y*W
        face_v2=X + 1 + Y* W
        face_v3=X + 1 + (Y+1) * W
        
        faces.append((face_v1,face_v2,face_v3))
        
        face_v1= X+Y*W
        face_v2=X  + (Y+1)* W
        face_v3=X + 1 + (Y+1) * W
        
        faces.append((face_v1,face_v2,face_v3))

face = np.array(faces)


cube = mesh.Mesh(np.zeros(face.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(face):
    for j in range(3):
        cube.vectors[i][j] = vert[f[j],:]

# Write the mesh to file "cube.stl"
cube.save('photo.stl')