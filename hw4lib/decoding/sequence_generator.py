import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        batch_size = x.size(0)
        current_sequences = x.clone().to(self.device)
        scores = torch.zeros(batch_size, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Generate tokens up to max_length
        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break
            
            # Get logits for next token
            next_logits = self.score_fn(current_sequences)  # (batch_size, vocab_size)
            
            # Apply repeat penalty if requested
            if repeat_penalty != 1.0:
                next_logits = self._apply_repeat_penalty(next_logits, current_sequences, repeat_penalty)
            
            # Apply temperature scaling
            next_logits = next_logits / temperature
            
            # Convert to log probabilities
            log_probs = torch.log_softmax(next_logits, dim=-1)
            
            # Greedy selection: take the most likely token
            next_tokens = torch.argmax(log_probs, dim=-1)  # (batch_size,)
            
            # Get token scores for tracking
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)  # (batch_size,)
            
            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)
            
            # Append next tokens to sequences
            current_sequences = torch.cat([current_sequences, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for EOS token
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos
        
        return current_sequences, scores

        
        # TODO: Implement greedy search
        #raise NotImplementedError # Remove once implemented
    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int = 3,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to maintain
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
            
        Returns:
            Tuple of tensors: (sequences, scores)
            - sequences is of shape (batch_size, beam_width, sequence_length)
            - scores is of shape (batch_size, beam_width)
        """
        # Input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device
        vocab_size = self.tokenizer.vocab_size
        
        # Initialize scores and flags
        scores = torch.zeros(batch_size, beam_width, device=device)
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)
        
        # Compute initial logits and probabilities
        logits = self.score_fn(x)  # (batch_size, vocab_size)
        
        # Process logits
        if repeat_penalty != 1.0:
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        logits = logits / temperature
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)
        
        # Select top beam_width tokens
        next_scores, next_tokens = log_probs.topk(beam_width, dim=-1)  # (batch_size, beam_width)
        scores = next_scores
        
        # Expand x along beam dimension
        x = x.unsqueeze(1).expand(batch_size, beam_width, seq_len)  # (batch_size, beam_width, seq_len)
        
        # Append next_tokens to x
        next_tokens = next_tokens.unsqueeze(-1)  # (batch_size, beam_width, 1)
        x = torch.cat([x, next_tokens], dim=-1)  # (batch_size, beam_width, seq_len+1)
        
        # Update finished flags where EOS token is encountered
        finished = (next_tokens.squeeze(-1) == self.tokenizer.eos_id)  # (batch_size, beam_width)
        
        # Continue generating until max length
        for t in range(1, self.max_length - seq_len):
            # If all sequences are finished, break
            if finished.all():
                break
            
            # Compute logits for next tokens
            next_token_scores = []
            
            for b in range(batch_size):
                batch_next_scores = []
                for k in range(beam_width):
                    if finished[b, k]:
                        # For finished beams, create dummy logits
                        dummy_logits = torch.full((1, vocab_size), float('-inf'), device=device)
                        dummy_logits[0, self.tokenizer.eos_id] = 0
                        batch_next_scores.append(dummy_logits)
                    else:
                        # Get logits for current sequence
                        curr_x = x[b, k].unsqueeze(0)  # (1, seq_len+t)
                        
                        # Create batch tensor with this sequence in position b
                        batch_x = torch.full((batch_size, curr_x.size(1)), self.tokenizer.pad_id, 
                                            dtype=torch.long, device=device)
                        batch_x[b] = curr_x[0]
                        
                        # Get logits for this batch, but only keep position b
                        curr_logits = self.score_fn(batch_x)[b:b+1]  # (1, vocab_size)
                        
                        # Apply temperature and repetition penalty
                        if repeat_penalty != 1.0:
                            curr_logits = self._apply_repeat_penalty(curr_logits, curr_x, repeat_penalty)
                        curr_logits = curr_logits / temperature
                        
                        batch_next_scores.append(curr_logits)
                next_token_scores.append(torch.cat(batch_next_scores, dim=0))
            
            # Process each batch separately
            next_x = []
            next_scores = []
            next_finished = []
            
            for b in range(batch_size):
                # Get scores for this batch
                batch_token_scores = next_token_scores[b]  # (beam_width, vocab_size)
                
                # Convert to log probabilities
                batch_log_probs = torch.log_softmax(batch_token_scores, dim=-1)  # (beam_width, vocab_size)
                
                # Compute cumulative scores
                batch_cum_scores = scores[b].unsqueeze(-1) + batch_log_probs  # (beam_width, vocab_size)
                
                # Flatten for beam selection
                flat_cum_scores = batch_cum_scores.view(-1)  # (beam_width * vocab_size)
                
                # Select top beam_width candidates
                batch_next_scores, indices = flat_cum_scores.topk(beam_width)  # (beam_width)
                
                # Re-map to get beam indices and token ids
                beam_indices = indices // vocab_size  # (beam_width)
                token_indices = indices % vocab_size  # (beam_width)
                
                # Create new sequences
                batch_next_x = []
                batch_next_finished = []
                
                for i in range(beam_width):
                    # Get parent beam and token
                    beam_idx = beam_indices[i].item()
                    token_idx = token_indices[i].item()
                    
                    # If parent beam was finished, propagate the finished state
                    if finished[b, beam_idx]:
                        new_seq = x[b, beam_idx].clone()
                        batch_next_x.append(new_seq)
                        batch_next_finished.append(True)
                    else:
                        # Create new sequence by appending token
                        parent_seq = x[b, beam_idx]
                        new_seq = torch.cat([parent_seq, torch.tensor([token_idx], device=device)])
                        batch_next_x.append(new_seq)
                        
                        # Check if sequence is now finished
                        if token_idx == self.tokenizer.eos_id:
                            batch_next_finished.append(True)
                        else:
                            batch_next_finished.append(False)
                
                # Pad sequences to same length if needed
                max_seq_len = max(seq.size(0) for seq in batch_next_x)
                padded_batch_next_x = []
                
                for seq in batch_next_x:
                    if seq.size(0) < max_seq_len:
                        padding = torch.full((max_seq_len - seq.size(0),), self.tokenizer.pad_id, 
                                            dtype=torch.long, device=device)
                        padded_seq = torch.cat([seq, padding])
                    else:
                        padded_seq = seq
                    padded_batch_next_x.append(padded_seq)
                
                # Add results to lists
                next_x.append(torch.stack(padded_batch_next_x))
                next_scores.append(batch_next_scores)
                next_finished.append(torch.tensor(batch_next_finished, dtype=torch.bool, device=device))
            
            # Ensure all batches have same sequence length
            max_batch_seq_len = max(batch_seqs.size(-1) for batch_seqs in next_x)
            for b in range(batch_size):
                if next_x[b].size(-1) < max_batch_seq_len:
                    padding = torch.full((beam_width, max_batch_seq_len - next_x[b].size(-1)), 
                                        self.tokenizer.pad_id, dtype=torch.long, device=device)
                    next_x[b] = torch.cat([next_x[b], padding], dim=-1)
            
            # Update x, scores, and finished
            x = torch.stack(next_x)  # (batch_size, beam_width, seq_len+t+1)
            scores = torch.stack(next_scores)  # (batch_size, beam_width)
            finished = torch.stack(next_finished)  # (batch_size, beam_width)
        
        # Post-process sequences to ensure they have EOS tokens
        for b in range(batch_size):
            for k in range(beam_width):
                # If sequence doesn't end with EOS, find first pad token and replace it with EOS
                # or append EOS if no pad tokens
                if not finished[b, k]:
                    pad_indices = (x[b, k] == self.tokenizer.pad_id).nonzero(as_tuple=True)[0]
                    if pad_indices.nelement() > 0:
                        # Replace first pad with EOS
                        x[b, k, pad_indices[0]] = self.tokenizer.eos_id
                    else:
                        # Append EOS - should not happen with our padding mechanism, but just in case
                        eos_tensor = torch.tensor([self.tokenizer.eos_id], device=device)
                        x[b, k] = torch.cat([x[b, k], eos_tensor])
        
        return x, scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
