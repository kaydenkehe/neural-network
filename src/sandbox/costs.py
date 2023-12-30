# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

'''
Every cost class includes three methods:
    - forward: Compute cost
    - backward: Compute derivative of cost
'''

# Binary Cross Entropy - Binary classification
class BinaryCrossentropy:

    # (-1 / m * sum(Yln(A) + (1 - Y)ln(1 - A)))
    def forward(self, AL, Y):
        return np.squeeze(-1 / Y.shape[0] * np.sum(np.dot(np.log(AL.T), Y) + np.dot(np.log(1 - AL.T), 1 - Y)))
    
    # (-Y/A + (1 - Y)/(1 - A))
    def backward(self, AL, Y):
        return -Y / AL + (1 - Y) / (1 - AL)

# Categorical Cross Entropy - Multiclass classification
class CategoricalCrossentropy:

    def forward(self, AL, Y):
        return np.squeeze(-1 / Y.shape[0] * np.sum(np.dot(np.log(AL.T), Y)))

    def backward(self, AL, Y):
        return -Y / AL
        return AL - Y

# Mean Squared Error - Regression
class MSE:

    # (1 / m * sum((Y - A)^2))
    def forward(self, AL, Y):
        return np.squeeze(1 / Y.shape[0] * np.sum(np.square((Y - AL))))
    
    # (-2 * (Y - A))
    def backward(self, AL, Y):
        return -2 * (Y - AL)

# Mean Absolute Error - Regression
class MAE:

    # (1 / m * sum(|Y - A|))
    def forward(self, AL, Y):
        return np.squeeze(1 / Y.shape[0] * np.sum(np.abs(Y - AL)))
    
    # -1 if AL < Y, 1 if AL > Y, 0 otherwise
    def backward(self, AL, Y):
        return np.where(AL < Y, -1, np.where(AL > Y, 1, 0))
