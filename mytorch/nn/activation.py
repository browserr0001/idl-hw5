import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        self.Z=Z
        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            A_moved = np.moveaxis(self.A, self.dim, -1)
            dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
            original_shape = A_moved.shape
            self.A = A_moved.reshape(-1, C)
            dLdA = dLdA_moved.reshape(-1, C)
        batch_size = self.A.shape[0]
        dLdZ = np.zeros_like(self.A)

        for i in range(batch_size):
            a_i = self.A[i:i+1].T
            dLdA_i = dLdA[i:i+1].T
            outer_prod = -np.dot(a_i, a_i.T)
            diag_indices = np.arange(C)
            outer_prod[diag_indices, diag_indices] += a_i.flatten()
            dLdZ_i = np.dot(outer_prod, dLdA_i)
            dLdZ[i] = dLdZ_i.T


        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            dLdZ_reshaped = dLdZ.reshape(original_shape)
            self.A = np.moveaxis(self.A.reshape(original_shape), -1, self.dim)
            dLdZ = dLdZ = np.moveaxis(dLdZ_reshaped, -1, self.dim)

        return dLdZ
 

    