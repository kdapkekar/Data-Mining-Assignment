import numpy as np
import matplotlib.pyplot as plt

L = np.array([[5,-2,-3,0,0,0],[-2,5,-3,0,0,0],[-3,-3,10,-4,0,0],[0,0,-4,24,-10,-10],[0,0,0,-10,110,-100],[0,0,0,-10,-100,110]])
L_s = np.array([[1,-0.4,-0.42,0,0,0],[-0.4,1,-0.42,0,0,0],[-0.42,-0.42,1,-0.25,0,0],[0,0,-0.25,1,-0.19,-0.19],[0,0,0,-0.19,1,-0.90],[0,0,0,-0.19,-0.90,1]])
print(L)
print(L_s)

eigen_Val_L, eigen_Vect_L = np.linalg.eig(L)
eigen_Val_L_s, eigen_Vect_L_s = np.linalg.eig(L_s)
print("Eigen values for L",eigen_Val_L)
print("Eigen Vectors for L",eigen_Vect_L)
print("Eigen values for L_s",eigen_Val_L_s)
print("Eigen Vectors for L_s",eigen_Vect_L_s)

ind_L = np.argmin(eigen_Val_L)
ind_L_s = np.argmin(eigen_Val_L_s)
print(eigen_Val_L[ind_L],eigen_Val_L_s[ind_L_s])
u = eigen_Vect_L[ind_L]
u_s = eigen_Vect_L_s[ind_L_s]
print("Eigen vector corresponding to smallest eigen value",u)
print("Eigen vector corresponding to smallest eigen value",u_s)

x1= list(range(1,7))
print(x1)
plt.plot(x1,eigen_Vect_L[ind_L] , label="For Laplacian matrix")
plt.plot(x1,eigen_Vect_L_s[ind_L_s] , label="For Symmetric Laplacian matrix")
plt.title('Eigen vectors for smallest eigen value')
plt.legend()
plt.show()