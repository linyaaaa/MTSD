import math

d_eye = 65

b = 65/(2*2000)
y = 2*math.atan(b)
final_alpha = y*180/math.pi
print(final_alpha)

alpha = final_alpha
disparity = -1
beta = alpha - disparity
vergence_distance = d_eye/(2*math.tan(beta/2*(math.pi/180)))
finla_vergence = vergence_distance/1000
print(finla_vergence)

# 1 dioter = 0.57 degree

# 23 28 83 84