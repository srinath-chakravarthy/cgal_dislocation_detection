# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 09:06:35 2017
CGAL read xyz file and tesselate this file
need point set processing and then Delaunay_triangulate 3

@author: srinath
"""
from __future__ import print_function
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Kernel import  cross_product
from CGAL.CGAL_Kernel import Segment_3
import numpy as np

from CGAL.CGAL_Point_set_processing_3 import *
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3
# ---- cell and vertex handles
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3_Cell_handle as cell_handle
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3_Vertex_handle as vertex_handle
# ---- Iterators
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3_Finite_vertices_iterator as vertex_iterator
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3_Finite_edges_iterator as edge_iterator
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3_Finite_facets_iterator as facet_iterator
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3_Finite_cells_iterator as cell_iterator


from CGAL.CGAL_Triangulation_3 import Ref_Locate_type_3
from CGAL.CGAL_Triangulation_3 import VERTEX
from CGAL.CGAL_Kernel import Ref_int


lattice_constant = 4.032
a0 = 4.032

# ---
def min_arg(v, setL):
    i = 0
    min = 1000000.0
    index = 0
    for l in setL:
        diff = v-l
        ss = diff.squared_length()
        if (ss < min):
            min = ss
            index =i
        i+= 1
    print ("Index found = ", index, setL[index], v)
    return setL[index]

class Rotation:
    import numpy as np
    v1 = Vector_3
    v2 = Vector_3
    v3 = Vector_3

    def __init__(self,sample, primitive_cell):
        self.sample = sample
        self.primitive_cell = primitive_cell
        self.R = np.zeros((3,3))
        i = 0
        for p in primitive_cell:
            j = 0
            for s in sample:
                # print (i,j, p,s)
                self.R[i,j] = p*s
                j += 1
            i += 1
    def setR(self,x,y,z):
        self.R[:,0] = np.transpose(np.array([x.x(),x.y(),x.z()]))
        self.R[:,1] = np.transpose(np.array([y.x(), y.y(), y.z()]))
        self.R[:,2] = np.transpose(np.array([z.x(), z.y(), z.z()]))

        # self.R[0,:] = np.array([x.x(),x.y(),x.z()])
        # self.R[1,:] = np.array([y.x(), y.y(), y.z()])
        # self.R[2,:] = np.array([z.x(), z.y(), z.z()])


    def rotate(self,v):
        vv = np.array([v.x(),v.y(),v.z()])
        out = np.transpose(self.R).dot(vv)
        outv = Vector_3(out[0],out[1],out[2])
        return outv
    def rotate_inverse(self,v):
        vv = np.array([v.x(),v.y(),v.z()])
        out = self.R.dot(vv)
        outv = Vector_3(out[0],out[1],out[2])
        return outv

def write_to_vtk(filename,T):
    f = open(filename,'w')
    f.write('# vtk DataFile Version 2.0 \n')
    f.write('CGAL mesh \n')
    f.write('ASCII \n')
    f.write('DATASET UNSTRUCTURED_GRID \n')
    print("POINTS ",T.number_of_vertices(), "float", file=f)
    mapv = {}
    i = 0
    for vertex in T.finite_vertices():
        p = T.point(vertex)
        print('{0:.5f}    {1:.5f}    {2:.5f}'.format(p.x(),p.y(),p.z()), file=f)
        mapv[str(p)] = i
        i+=1
        # print(mapv[p])
    print ("",file=f)
    print("CELLS ", T.number_of_finite_cells(), T.number_of_finite_cells()*5, file=f)
    for cell in T.finite_cells():
        v0 = cell.vertex(0)
        v1 = cell.vertex(1)
        v2 = cell.vertex(2)
        v3 = cell.vertex(3)
        if (T.is_infinite(v0) or T.is_infinite(v1) or T.is_infinite(v1) or T.is_infinite(v3)):
            continue
        p0 = T.point(v0)
        p1 = T.point(v1)
        p2 = T.point(v2)
        p3 = T.point(v3)
        print ("4    {0:d}   {1:d}   {2:d}   {3:d} ".format(mapv[str(p0)], mapv[str(p1)], mapv[str(p2)], mapv[str(p3)]), file=f)
    print ("",file=f)
    print("CELL_TYPES {0:d}".format(T.number_of_finite_cells()),file=f)
    for cell in T.finite_cells():
        print ("10",file=f)
    print("CELL_DATA {0:d}".format(T.number_of_finite_cells()), file=f)
    print("SCALARS DB integer",file=f)
    print("LOOKUP_TABLE default", file=f)
    for cell in T.finite_cells():
        print("0", file=f)


a6 = 1.0/6.0
a62 = 2.0*a6
# ---- Contains the burgers vector and ideal lattice vectors
# ---- Ideal lattice vectors burgers vectors for fcc----
L_fcc= [Vector_3( 0.5,  0.5,  0.0),
        Vector_3( 0.0,  0.5,  0.5),
        Vector_3( 0.5,  0.0,  0.5),
        Vector_3(-0.5, -0.5,  0.0),
        Vector_3( 0.0, -0.5, -0.5),
        Vector_3(-0.5,  0.0, -0.5),
        Vector_3(-0.5,  0.5,  0.0),
        Vector_3( 0.0, -0.5,  0.5),
        Vector_3(-0.5,  0.0,  0.5),
        Vector_3( 0.5, -0.5,  0.0),
        Vector_3( 0.0,  0.5, -0.5),
        Vector_3( 0.5,  0.0, -0.5),
        Vector_3(a6,a6,a62), Vector_3(a6,a62,a6), Vector_3(a62,a6,a6),
        Vector_3(a6,a6,-a62), Vector_3(a6,a62,-a6), Vector_3(a62,a6,-a6),
        Vector_3(a6,-a6,a62), Vector_3(a6,-a62,a6), Vector_3(a62,-a6,a6),
        Vector_3(a6,-a6,-a62), Vector_3(a6,-a62,-a6), Vector_3(a62,-a6,-a6),
        Vector_3(-a6,a6,a62), Vector_3(-a6,a62,a6), Vector_3(-a62,a6,a6),
        Vector_3(-a6,a6,-a62), Vector_3(-a6,a62,-a6), Vector_3(-a62,a6,-a6),
        Vector_3(-a6,-a6,a62), Vector_3(-a6,-a62,a6), Vector_3(-a62,-a6,a6),
        Vector_3(-a6,-a6,-a62), Vector_3(-a6,-a62,-a6), Vector_3(-a62,-a6,-a6)]
        # Vector_3(1.0,0.0,0.0),Vector_3(-1.0,0.0,0.0),
        # Vector_3(0.0,1.0,0.0),Vector_3(0.0,-1.0,0.0),
        # Vector_3(0.0,0.0,1.0),Vector_3(0.0,0.0,-1.0)]


primitive_cell = [Vector_3( 0.0,  0.5,  0.5),
                 Vector_3( 0.5,  0.0,  0.5),
                 Vector_3( 0.5,  0.5,  0.0)]

for p in primitive_cell:
    p.normalize()

sample_x = Vector_3(1.0,1.0,-2.0)
sample_x.normalize()
sample_y = Vector_3(1.0,1.0,1.0)
sample_y.normalize()
sample_z = cross_product(sample_x, sample_y)
sample = [sample_x, sample_y, sample_z]

R = Rotation(sample, primitive_cell)
R.setR(sample_x, sample_y, sample_z)
L_fcc_R = []
print ("Burgers vectors are ")
for lat in L_fcc:
    l1 = R.rotate(lat)
    print(l1*a0)
    L_fcc_R.append(l1*a0)


print ('test')


# ---- Read xyz data file from file
datafile = '/home/srinath/cgal_testing/lammps_single_disloc/new1.xyz'
points = []
print("Reading xyz points ...")
read_xyz_points(datafile,points)
print(len(points), " points read")

# --- Now try to tessellate these points
print("Tessellating domain points ...")
T = Delaunay_triangulation_3(points)
T.number_of_vertices()
print("Delaunay finite cells ", T.number_of_finite_cells(), " created")
# --- Now triangulation is done

write_to_vtk('test.vtk',T)

L_burgers = []
L_index = []
i = 0
for cell in T.finite_cells():
    if i > 0:
        break
    else:
        i+=1

    a = T.point(cell.vertex(0))
    b = T.point(cell.vertex(1))
    c = T.point(cell.vertex(2))
    d = T.point(cell.vertex(3))

    # --- Edge vectors
    v_ab = Vector_3(a,b)
    v_bc = Vector_3(b,c)
    v_ca = Vector_3(c,a)
    v_bd = Vector_3(b,d)
    v_cd = Vector_3(c,d)
    v_ad = Vector_3(a,d)

    v_ba = -v_ab
    v_cb = -v_bc
    v_ac = -v_ca
    v_db = -v_bd
    v_dc = -v_cd
    v_da = -v_ad

    L_ab = (min_arg(v_ab,L_fcc_R))
    L_ba = -L_ab
    L_bc = (min_arg(v_bc,L_fcc_R))
    L_cb = -L_bc
    L_ca = (min_arg(v_ca,L_fcc_R))
    L_ac = -L_ca
    L_bd = (min_arg(v_bd,L_fcc_R))
    L_db = -L_bd
    L_cd = (min_arg(v_cd,L_fcc_R))
    L_dc = -L_cd
    L_ad = (min_arg(v_ad,L_fcc_R))
    L_da = -L_ad

    b_abc = L_ab + L_bc + L_ca
    b_cbd = L_cb + L_bd + L_dc
    b_acd = L_ac + L_cd + L_da
    b_adb = L_ad + L_db + L_ba


    print (i,'ABC = ',b_abc, L_ab, L_bc, L_ca)
    print (i,'CBD = ',b_cbd, L_cb, L_bd, L_dc)
    print (i,'ACD = ',b_acd, L_ac, L_cd, L_da)
    print (i,'ADB = ',b_adb, L_ad, L_db, L_ba)
    print(i, "Total burgers = ", b_abc + b_cbd + b_acd + b_adb)
    i+= 1
print ("test")
# i = 0;
# for edge in T.finite_edges():
#     s = T.segment(edge)
#     v = s.to_vector()
#     ll = min_arg(v,L_fcc_R)
#     print (i, ll)
#     L_burgers.append(ll)
#     if i > 1:
#         break
#     i+=1
