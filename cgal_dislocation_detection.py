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
    # print ("Index found = ", index, setL[index])
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
    def rotate(self,v):
        vv = np.array([v.x(),v.y(),v.z()])
        out = np.transpose(self.R).dot(vv)
        outv = Vector_3(out[0],out[1],out[2])
        return outv




lattice_constant = 4.032
a0 = 4.032
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
        Vector_3(-a6,-a6,-a62), Vector_3(-a6,-a62,-a6), Vector_3(-a62,-a6,-a6),
        Vector_3(1.0,0.0,0.0),Vector_3(-1.0,0.0,0.0),
        Vector_3(0.0,1.0,0.0),Vector_3(0.0,-1.0,0.0),
        Vector_3(0.0,0.0,1.0),Vector_3(0.0,0.0,-1.0)]


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
L_fcc_R = []
for lat in L_fcc:
    l1 = R.rotate(lat)
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
print("Delaunay finite cells ", T.number_of_finite_cells(), " created")
# --- Now triangulation is done

L_burgers = []
L_index = []
i = 0
for cell in T.finite_cells():
    if i < 1000 or i > 1005:
        i+=1
        continue
    a = T.point(cell.vertex(3))
    b = T.point(cell.vertex(0))
    c = T.point(cell.vertex(1))
    d = T.point(cell.vertex(2))
    # --- Edge vectors
    v_ab = Vector_3(a,b)
    v_bc = Vector_3(b,c)
    v_ca = Vector_3(c,a)
    v_bd = Vector_3(b,d)
    v_cd = Vector_3(c,d)
    v_ad = Vector_3(a,d)

    # v_ba = -v_ab
    # v_cb = -v_bc
    # v_ac = -v_ca
    # v_db = -v_bd
    # v_dc = -v_cd
    # v_da = -v_ad

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


    print (i,'ABC = ',b_abc)#, L_ab, L_bc, L_ca)
    print (i,'CBD = ',b_cbd)#, L_cb, L_bd, L_dc)
    print (i,'ACD = ',b_acd)#, L_ac, L_cd, L_da)
    print (i,'ADB = ',b_adb)#, L_ad, L_db, L_ba)
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
