import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO: Create two maps as described in the docstring above.
    # It's best if you also sort the chars before assigning indices, so that
    # they're in lexical order.
    # ====== YOUR CODE: ======
    chars_set = set(text)
    sorted_chars = sorted(chars_set)
    idx_to_char = { i : sorted_chars[i] for i in range(0, len(sorted_chars)) }
    char_to_idx = {v: k for k, v in idx_to_char.items()}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    dd = {ord(c) : None for c in chars_to_remove}
    text_clean = text.translate(dd)
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    indices = torch.LongTensor([char_to_idx[char] for char in text]).view(-1, 1)
    result = torch.zeros(len(text), len(char_to_idx), dtype = torch.int8)
    result = result.scatter_(1, indices, 1)
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    indices = embedded_text.argmax(dim=1)
    result = ''.join([idx_to_char[idx.item()] for idx in indices])
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO: Implement the labelled samples creation.
    # 1. Embed the given text.
    # 2. Create the samples tensor by splitting to groups of seq_len.
    #    Notice that the last char has no label, so don't use it.
    # 3. Create the labels tensor in a similar way and convert to indices.
    # Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    embedded_text = chars_to_onehot(text, char_to_idx).to(device)
    samples = torch.stack(embedded_text[0:-1, :].split(seq_len)[0:-1])
    labels = torch.stack(embedded_text[1:,:].split(seq_len)[0:-1]).argmax(dim=2)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    scaled_input = y * (1/temperature)
    result = torch.softmax(scaled_input, dim=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO: Implement char-by-char text generation.
    # 1. Feed the start_sequence into the model.
    # 2. Sample a new char from the output distribution of the last output
    #    char. Convert output to probabilities first.
    #    See torch.multinomial() for the sampling part.
    # 3. Feed the new char into the model.
    # 4. Rinse and Repeat.
    #
    # Note that tracking tensor operations for gradient calculation is not
    # necessary for this. Best to disable tracking for speed.
    # See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        h = None
        embedded_start_sequence = chars_to_onehot(start_sequence, char_to_idx).unsqueeze(dim=0).to(dtype=torch.float).to(device)
        embedded, h = model(embedded_start_sequence, h)
        embedded = embedded[:, len(start_sequence) -1, :]
        for i in range(n_chars - len(start_sequence)):
            prob = hot_softmax(embedded.squeeze(), temperature=T)
            generated_char_idx = torch.multinomial(input=prob, num_samples=1)
            generated_char = idx_to_char[generated_char_idx.item()]
            out_text += generated_char
            embedded_generated_char = chars_to_onehot(generated_char, char_to_idx).unsqueeze(dim=0).to(dtype=torch.float).to(device)
            embedded, h = model(embedded_generated_char, h)


    # ========================

    return out_text


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model.
        # To implement the affine transforms you can use either nn.Linear
        # modules (recommended) or create W and b tensor pairs directly.
        # Create these modules or tensors and save them per-layer in
        # the layer_params list.
        # Important note: You must register the created parameters so
        # they are returned from our module's parameters() function.
        # Usually this happens automatically when we assign a
        # module/tensor as an attribute in our module, but now we need
        # to do it manually since we're not assigning attributes. So:
        #   - If you use nn.Linear modules, call self.add_module() on them
        #     to register each of their parameters as part of your model.
        #   - If you use tensors directly, wrap them in nn.Parameter() and
        #     then call self.register_parameter() on them. Also make
        #     sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        for i in range(self.n_layers):
            input_size = in_dim if i == 0 else h_dim
            layer = {}
            layer['xz'] = nn.Linear(input_size, h_dim, bias=False)
            layer['hz'] = nn.Linear(h_dim, h_dim, bias=True)

            layer['xr'] = nn.Linear(input_size, h_dim, bias=False)
            layer['hr'] = nn.Linear(h_dim, h_dim, bias=True)

            layer['xg'] = nn.Linear(input_size, h_dim, bias=False)
            layer['hg'] = nn.Linear(h_dim, h_dim, bias=True)

            for key, value in layer.items():                
                self.add_module("layer {} {}".format(i, key), value)

            layer['dropout'] = nn.Dropout(dropout)
            self.layer_params.append(layer)

        self.output_layer = nn.Linear(h_dim, out_dim, bias=True)

        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor=None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None
        # TODO: Implement the model's forward pass.
        # You'll need to go layer-by-layer from bottom to top (see diagram).
        # Tip: You can use torch.stack() to combine multiple tensors into a
        # single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        layer_input = layer_input.permute(1 ,0, 2)
        prev_layer_output = layer_input
        for i in range(self.n_layers):
            layer = self.layer_params[i]
            hiden_states = []
            hts = []
            ht = None
            for xt in prev_layer_output:
                zt = torch.sigmoid(layer['xz'](xt) + layer['hz'](layer_states[i]))
                rt = torch.sigmoid(layer['xr'](xt) + layer['hr'](layer_states[i]))
                gt = torch.tanh(layer['xg'](xt) + layer['hg'](rt * layer_states[i]))
                layer_states[i] = zt * layer_states[i] + (1 - zt) * gt
                hts.append(layer_states[i])
            prev_layer_output = torch.stack(hts)
            prev_layer_output = layer['dropout'](prev_layer_output)
        layer_output = self.output_layer(prev_layer_output)
        layer_output = layer_output.permute(1 ,0 ,2)
        hidden_state = torch.stack(layer_states).permute(1 ,0 ,2)
        # ========================
        return layer_output, hidden_state
