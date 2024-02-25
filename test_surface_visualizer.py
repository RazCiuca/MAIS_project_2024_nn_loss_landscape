import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(0.1, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
theta = np.arctan(Y/X)
r = np.sqrt(X**2 + Y**2)
Z = (1.0/(1.0+theta**2))*(r - 3)**2 + 0.5*(theta-1)**2

# Plot the surface.
plt.contour(X, Y, Z, cmap=cm.coolwarm, levels=30)

# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()