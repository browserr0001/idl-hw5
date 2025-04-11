import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """
    Create a boolean mask for padded positions in a batch of sequences.

    Args:
        padded_input (Tensor): Padded sequences of shape (N, T, ...) or (N, T)
        input_lengths (Tensor): Actual lengths of the sequences (N,)

    Returns:
        Tensor: A boolean mask of shape (N, T), with True in padded positions.
    """
    N, T = padded_input.shape[:2]  # Get batch size and sequence length
    device = padded_input.device

    # Create a range tensor [0, 1, 2, ..., T-1] shaped (1, T)
    range_tensor = torch.arange(T, device=device).unsqueeze(0)

    # Compare each position to the corresponding input length
    mask = range_tensor >= input_lengths.unsqueeze(1)

    return mask  # shape: (N, T), True for padding


''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """
    Create a causal attention mask that prevents attending to future positions.

    Args:
        padded_input (Tensor): Input tensor of shape (N, T, ...) or (N, T)

    Returns:
        Tensor: A boolean mask of shape (T, T), with True above the diagonal (future positions).
    """
    T = padded_input.shape[1]
    device = padded_input.device

    # Generate an upper triangular matrix with True above the diagonal
    mask = torch.triu(torch.ones((T, T), device=device), diagonal=1).bool()

    return mask  # shape: (T, T), True = block

