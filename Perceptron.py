import numpy as np
from matplotlib import pyplot as plt

X = np.array([[-2,4,-1], [4,1,-1], [1,6,-1], [2,4,-1], [6,2,-1]])
Y = np.array([-1, -1, 1, 1, 1])

'''
X = np.array([[0,0,-1], [0,1,-1], [2,0,-1], [2,1,-1], [2,2,-1]])
Y = np.array([-1, -1, 1, 1, 1])
'''

def perceptron_sgd_plot(X, Y):
	w = np.zeros(len(X[0]))
	eta = 1
	n = 30
	errors = []

	for t in range(n):
		total_error = 0
		for i, x in enumerate(X):
			if (np.dot(X[i], w) * Y[i]) <= 0:
				total_error += (np.dot(X[i], w) * Y[i])
				w += eta * X[i] * Y[i]
		errors.append(-total_error)

	plt.plot(errors)
	plt.xlabel('Epoch')
	plt.ylabel('Total Loss')
	plt.show()

	return w

w = perceptron_sgd_plot(X, Y)
print(w)


for d, sample in enumerate(X):
# Plot the negative samples
	if d < 2:
		plt.scatter(sample[0], sample[1], s = 120, marker = '_', linewidths = 2, color = "green")
# Plot the positive samples
	else:
		plt.scatter(sample[0], sample[1], s = 120, marker = '+', linewidths = 2, color = "red")

# Add our test samples
plt.scatter(2, 2, s = 120, marker = '_', linewidths = 2, color = 'yellow')
plt.scatter(4, 3, s = 120, marker = '+', linewidths = 2, color = 'blue')

'''
plt.scatter(-1, 2, s = 120, marker = '_', linewidths = 2, color = 'yellow')
plt.scatter(4, 3, s = 120, marker = '+', linewidths = 2, color = 'blue')
'''

# Print the hyperplane calculated by perceptron_sgd_plot()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]
x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X, Y , U, V, scale = 1, color = 'blue')
plt.show()