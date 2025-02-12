import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "./data_nonlinear.csv"
df = pd.read_csv(file_path)
df.head()

x = df['X'].values
y = df['Y'].values

plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Y = a * X^3 + b * X^2 + c * X + d

X = np.array(df['X'].values)
Y = np.array(df['Y'].values)

a, b, c, d = 0.0, 0.0, 0.0, 0.0 # initialize all coefficients to zero

learning_rate = 1e-6
epochs = 10000

for epoch in range(epochs): #each little test
  Y_pred = a * X**3 + b * X**2 + c * X + d #our "guestimate" function

  # compute MSE loss for each iteration
  loss = np.mean((Y_pred - Y)**2)

  # compute gradients with respect to a,b,c,d
  grad_a = 2 * np.mean((Y_pred - Y) * X**3) # derivatives of MSE * X^3 (respect to coeff a), iterates through every x and y value/prediciton
  grad_b = 2 * np.mean((Y_pred - Y) * X**2) # derivatives of MSE * X^2 (respect to coeff b)
  grad_c = 2 * np.mean((Y_pred - Y) * X)
  grad_d = 2 * np.mean((Y_pred - Y))

  # update coefficients
  a -= learning_rate * grad_a #adjust by the gradient value * set learning rate
  b -= learning_rate * grad_b #same as above
  c -= learning_rate * grad_c #same as above
  d -= learning_rate * grad_d #same as above

  # Debug
  if epoch <= 5:
    print(f"Epoch {epoch}: a={a} b={b} c={c} d={d}, Loss={loss}")

print("Optimal solution:")
print(f"{a}x^3 + {b}x^2 + {c}x + {d}") #print out the best coefficients after all epochs

X_range = np.linspace(min(X), max(X), 100)
Y_range = a * X_range**3 + b * X_range**2 + c * X_range + d

plt.scatter(X, Y, label='Data')
plt.plot(X_range, Y_range, color='red', label='Spline')
plt.xlabel('X')
plt.ylabel('Y')

file_path = "./data_two_variables.csv"

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv(file_path)
df.head()

x1 = df['X1'].values
x2 = df['X2'].values
y = df['Y'].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y)

X1 = np.array(df['X1'].values) # create np array for x1 values
X2 = np.array(df['X2'].values) # create np array for x2 values
Y = np.array(df['Y'].values) # create np array for x3 values

m1, m2, m3 = 0.0, 0.0, 0.0 # initialize coefficients to 0.0

learning_rate = 1e-3 #set learning rate
epochs = 10000 #set epochs

for epoch in range(epochs): #iterate through each "test"
  Y_pred = m1 * X1 + m2 * X2 + m3 #set prediction model

  loss = np.mean((Y_pred - Y)**2) #calculate loss each test

  grad_m1 = 2 * np.mean((Y_pred - Y) * X1) #calculate gradient for m1
  grad_m2 = 2 * np.mean((Y_pred - Y) * X2) #calculate gradient for m2
  grad_m3 = 2 * np.mean((Y_pred - Y)) # calculate gradient for m3

  m1 -= learning_rate * grad_m1 #subtract m1 by learning rate * gradient
  m2 -= learning_rate * grad_m2 #same as above
  m3 -= learning_rate * grad_m3 #same as above

  #Debug
  if epoch <= 5:
    print(f"Epoch {epoch}: m1={m1}, m2={m2}, m3={m3}, Loss={loss}")


print("Although the 5th epoch does not reach the value shown above in the description, I do end up converging on the correct optimal coefficients.")
print("Optimal coefficients:") #print out all the optimal coefficients
print(f"m1={m1}, m2={m2}, m3={m3}")