import numpy as np

weights = [1.0, 9.0, 45.0, 70.0, 45.0, 9.0, 1.0]

smooth_norm = 5832000.0 # Based on separate calculations

#define DER1_3 (1. / 60.)
#define DER1_2 (-3. / 20.)
#define DER1_1 (3. / 4.)
#define DER1_0 (0)

#Stencil derx {
#    [0][0][-3] = -AC_inv_dsx * DER1_3,
#    [0][0][-2] = -AC_inv_dsx * DER1_2,
#    [0][0][-1] = -AC_inv_dsx * DER1_1,
#    [0][0][1]  = AC_inv_dsx * DER1_1,
#    [0][0][2]  = AC_inv_dsx * DER1_2,
#    [0][0][3]  = AC_inv_dsx * DER1_3
#}

#Calculate normalization coefficient
smooth_norm = 0.0
for k, weightsk in enumerate(weights):
    for j, weightsj in enumerate(weights):
        for i, weightsi in enumerate(weights):
            smooth_norm += weightsi * weightsj * weightsk

coord_deg = ["M3","M2","M1","C0","P1","P2","P3"]
coord_axx = ["X", "X", "X", "X", "X", "X", "X" ]
coord_axy = ["Y", "Y", "Y", "Y", "Y", "Y", "Y" ]
coord_axz = ["Z", "Z", "Z", "Z", "Z", "Z", "Z" ]

#for k, weightsk in enumerate(weights):
#    for j, weightsj in enumerate(weights):
#        for i, weightsi in enumerate(weights):
#            label_x = coord_axx[i] 
#            label_y = coord_axy[j] 
#            label_z = coord_axz[k] 
#            deg_x   = coord_deg[i] 
#            deg_y   = coord_deg[j] 
#            deg_z   = coord_deg[k] 
#            iver    = -3 + i;
#            jver    = -3 + j;
#            kver    = -3 + k;
#
#            smooth_factor = (weightsi * weightsj * weightsk) / smooth_norm;
#
#            stencil_index  = "[%i][%i][%i]" % (kver, jver, iver)
#            coeff_fraction = "%.1f / %.1f" % (weightsi*weightsj*weightsk, smooth_norm)
#            coeff_tot      = "%e" % (smooth_factor)
#            coeff_label    = "SMOOTH_%s%s_%s%s_%s%s" % (label_x, deg_x, label_y, deg_y, label_z, deg_z)
#
#            print(stencil_index + " = " + coeff_fraction + " = " + coeff_tot + ", " + coeff_label)
#
#            define_line = "#define " + coeff_label + " ( " + coeff_fraction + " ) "
#            print(define_line)
#
#            stencil_line = stencil_index + " = " + coeff_label
#            print(stencil_line)

print("")
print("")
print("")

#Print out stencil
for k, weightsk in enumerate(weights):
    for j, weightsj in enumerate(weights):
        for i, weightsi in enumerate(weights):
            label_x = coord_axx[i] 
            label_y = coord_axy[j] 
            label_z = coord_axz[k] 
            deg_x   = coord_deg[i] 
            deg_y   = coord_deg[j] 
            deg_z   = coord_deg[k] 
            iver    = -3 + i;
            jver    = -3 + j;
            kver    = -3 + k;

            coeff_fraction = "%.1f / %.1f" % (weightsi*weightsj*weightsk, smooth_norm)
            coeff_label    = "SMOOTH_%s%s_%s%s_%s%s" % (label_x, deg_x, label_y, deg_y, label_z, deg_z)

            define_line = "#define " + coeff_label + " ( " + coeff_fraction + " ) "
            print(define_line)

print("")
print("")
print("")

print("Stencil smooth_kernel {")
for k, weightsk in enumerate(weights):
    for j, weightsj in enumerate(weights):
        for i, weightsi in enumerate(weights):
            label_x = coord_axx[i] 
            label_y = coord_axy[j] 
            label_z = coord_axz[k] 
            deg_x   = coord_deg[i] 
            deg_y   = coord_deg[j] 
            deg_z   = coord_deg[k] 
            iver    = -3 + i;
            jver    = -3 + j;
            kver    = -3 + k;

            stencil_index  = "[%i][%i][%i]" % (kver, jver, iver)
            coeff_label    = "SMOOTH_%s%s_%s%s_%s%s" % (label_x, deg_x, label_y, deg_y, label_z, deg_z)

            if (i == 6) and (j == 6) and (k ==6):
                stencil_line = stencil_index + " = " + coeff_label 
            else:
                stencil_line = stencil_index + " = " + coeff_label + ","
            print(stencil_line)

print("}")
print("")
print("")
print("")

#These approahes do not work beceaseu compilation time becomes unfeasible and if freezes over at the tartup optimization stage.
####for k, deg_z in enumerate(coord_deg):
####    for j, deg_y in enumerate(coord_deg):
####        for i, deg_x in enumerate(coord_deg):
####            label_x = coord_axx[i] 
####            label_y = coord_axy[j] 
####            label_z = coord_axz[k] 
####            iver    = -3 + i;
####            jver    = -3 + j;
####            kver    = -3 + k;
####
####            stencil_index  = "[%i][%i][%i]" % (kver, jver, iver)
####            coeff_label    = "Stencil value_%s%s_%s%s_%s%s {" % (label_x, deg_x, label_y, deg_y, label_z, deg_z)
####
####            print(coeff_label)
####            print(stencil_index + " = 1")
####            print("}")
####
####print("")
####print("")
####print("")

####for k, deg_z in enumerate(coord_deg):
####    for j, deg_y in enumerate(coord_deg):
####        for i, deg_x in enumerate(coord_deg):
####            label_x = coord_axx[i] 
####            label_y = coord_axy[j] 
####            label_z = coord_axz[k] 
####            iver    = -3 + i;
####            jver    = -3 + j;
####            kver    = -3 + k;
####            
####            cross_test = iver*jver*kver
####            if jver != 0:
####                ijdiag = abs(float(iver)/float(jver))
####            else:
####                ijdiag = 666.0
####            if kver != 0:
####                jkdiag = abs(float(jver)/float(kver))
####                ikdiag = abs(float(iver)/float(kver))
####            else:
####                jkdiag = 666.0
####                ikdiag = 666.0
####
####            if  (( (ijdiag == 1.0) and (jkdiag == 1.0) and (ikdiag == 1.0)) or 
####                 ( (iver == 0) and (jver == 0) ) or
####                 ( (iver == 0) and (kver == 0) ) or
####                 ( (jver == 0) and (kver == 0) )):
####
####                stencil_index  = "[%i][%i][%i]" % (kver, jver, iver)
####                coeff_label    = "Stencil value_%s%s_%s%s_%s%s {" % (label_x, deg_x, label_y, deg_y, label_z, deg_z)
####
####                print(coeff_label)
####                print(stencil_index + " = 1")
####                print("}")
####
####print("")
####print("")
####print("")

#Cube maximum stencil
if 0: 
    print("Max Stencil max5 {")
    for k, deg_z in enumerate(coord_deg):
        for j, deg_y in enumerate(coord_deg):
            for i, deg_x in enumerate(coord_deg):
                iver    = -3 + i;
                jver    = -3 + j;
                kver    = -3 + k;
    
                stencil_index  = "[%i][%i][%i]" % (kver, jver, iver)
    
                if (i == 6) and (j == 6) and (k ==6):
                    print(stencil_index + " = 1,")
                else:
                    print(stencil_index + " = 1")
    
    print("}")
    
    print("")
    print("")
    print("")

#Sphere maximum stencil 
if 1: 
    print("Max Stencil max5 {")
    for k, deg_z in enumerate(coord_deg):
        for j, deg_y in enumerate(coord_deg):
            for i, deg_x in enumerate(coord_deg):
                iver    = -3 + i;
                jver    = -3 + j;
                kver    = -3 + k;
    
                irr = np.float(iver*iver + jver*jver + kver*kver)
                ir  = np.sqrt(irr)
                
                if ir <= 3.0:
                    stencil_index  = "[%i][%i][%i]" % (kver, jver, iver)
    
                    #REMOVE THE LAST COMMA BY HAND!!!
                    print(stencil_index + " = 1,")
    
    print("}")
    
    print("")
    print("")
    print("")


#please pipe the result with > to get the stencil file. 

