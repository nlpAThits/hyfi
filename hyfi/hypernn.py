# adapted from https://github.com/dalab/hyperbolic_nn and https://github.com/ferrine/hyrnn/blob/master/hyrnn/nets.py

import torch
import torch.nn as nn
import geoopt as gt
from hyfi.constants import DEVICE, DEFAULT_DTYPE
import itertools
import torch.nn
import torch.nn.functional
import geoopt.manifolds.stereographic.math as pmath

MIN_NORM = 1e-15

    
class MobiusRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(MobiusRNN, self).__init__()

        self.ball = gt.PoincareBall()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # k = (1 / hidden_size)**0.5
        k_w = (6 / (self.hidden_size + self.hidden_size)) ** 0.5  # xavier uniform
        k_u = (6 / (self.input_size + self.hidden_size)) ** 0.5  # xavier uniform
        self.w = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k_w, k_w))
        self.u = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k_u, k_u))
        bias = torch.randn(hidden_size) * 1e-5
        self.b = gt.ManifoldParameter(pmath.expmap0(bias, k=self.ball.k), manifold=self.ball)
        
    def transition(self, x, h):
        """
        :param x: batch x input
        :param h: hidden x hidden
        :return: batch x hidden
        """
        W_otimes_h = pmath.mobius_matvec(self.w, h, k=self.ball.k)
        U_otimes_x = pmath.mobius_matvec(self.u, x, k=self.ball.k)
        Wh_plus_Ux = pmath.mobius_add(W_otimes_h, U_otimes_x, k=self.ball.k)
        
        return pmath.mobius_add(Wh_plus_Ux, self.b, k=self.ball.k)

    def init_rnn_state(self, batch_size, hidden_size):
        return torch.zeros((batch_size, hidden_size), dtype=DEFAULT_DTYPE, device=DEVICE)

    def forward(self, inputs):
        """
        :param inputs: batch x seq_len x embed_dim
        :return: batch x seq_len x hidden_size
        """
        hidden = self.init_rnn_state(inputs.shape[0], self.hidden_size)     # batch x hidden_size
        outputs = []
        for x in inputs.transpose(0, 1):            # seq_len x batch x dim transposes in order to iterate through words
            hidden = self.transition(x, hidden)     # of the whole batch at each step
            outputs += [hidden]
        return torch.stack(outputs).transpose(0, 1)    # batch x seq_len x hidden_size


class EuclRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EuclRNN, self).__init__()

        self.manifold = gt.Euclidean()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # k = (1 / hidden_size)**0.5
        k_w = (6 / (self.hidden_size + self.hidden_size)) ** 0.5  # xavier uniform
        k_u = (6 / (self.input_size + self.hidden_size)) ** 0.5   # xavier uniform
        self.w = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k_w, k_w))
        self.u = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k_u, k_u))
        bias = torch.randn(hidden_size) * 1e-5
        self.b = gt.ManifoldParameter(bias, manifold=self.manifold)

    def transition(self, x, h):
        """
        :param x: batch x input
        :param h: hidden x hidden
        :return: batch x hidden
        """
        W_otimes_h = h.matmul(self.w)
        U_otimes_x = x.matmul(self.u)
        return W_otimes_h + U_otimes_x + self.b

    def init_rnn_state(self, batch_size, hidden_size):
        return torch.zeros((batch_size, hidden_size), dtype=DEFAULT_DTYPE, device=DEVICE)

    def forward(self, inputs):
        """
        :param inputs: batch x seq_len x embed_dim
        :return: batch x seq_len x hidden_size
        """
        hidden = self.init_rnn_state(inputs.shape[0], self.hidden_size)  # batch x hidden_size
        outputs = []
        for x in inputs.transpose(0, 1):  # seq_len x batch x dim transposes in order to iterate through words
            hidden = self.transition(x, hidden)  # of the whole batch at each step
            outputs += [hidden]
        return torch.stack(outputs).transpose(0, 1)  # batch x seq_len x hidden_size


class MobiusGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlin=None, hyperbolic_input=True,
                 hyperbolic_hidden_state0=True, c=1.0):
        super().__init__()
        self.ball = gt.PoincareBall(c=c)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.weight_ih = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size))
             for i in range(num_layers)]
        )
        self.weight_hh = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size)) for _ in range(num_layers)]
        )
        if bias:
            biases = []
            for i in range(num_layers):
                bias = torch.randn(3, hidden_size) * 1e-5
                bias = gt.ManifoldParameter(pmath.expmap0(bias, k=self.ball.k), manifold=self.ball)
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)
        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.reset_parameters()

    def reset_parameters(self):
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            rows, cols = weight.size()
            stdv = (6 / (rows / 3 + cols)) ** 0.5  # xavier uniform
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0=None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input[:2]
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)
        if h0 is None:
            h0 = input.new_zeros(
                self.num_layers, max_batch_size, self.hidden_size, requires_grad=False
            )
        h0 = h0.unbind(0)
        if self.bias is not None:
            biases = self.bias
        else:
            biases = (None,) * self.num_layers
        outputs = []
        last_states = []
        out = input
        for i in range(self.num_layers):
            out, h_last = mobius_gru_loop(
                input=out,
                h0=h0[i],
                weight_ih=self.weight_ih[i],
                weight_hh=self.weight_hh[i],
                bias=biases[i],
                k=self.ball.k,
                hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
                hyperbolic_input=self.hyperbolic_input or i > 0,
                nonlin=self.nonlin,
                batch_sizes=batch_sizes,
            )
            outputs.append(out)
            last_states.append(h_last)
        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack(last_states)
        # default api assumes
        # out: (seq_len, batch, num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        # if packed:
        # out: (sum(seq_len), num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        return out, ht

    def extra_repr(self):
        return (
            "{input_size}, {hidden_size}, {num_layers}, bias={bias}, "
            "hyperbolic_input={hyperbolic_input}, "
            "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
            "c={self.ball.c}"
        ).format(**self.__dict__, self=self, bias=self.bias is not None)


class EuclGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlin=None):
        super().__init__()
        self.manifold = gt.Euclidean()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.weight_ih = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size))
             for i in range(num_layers)]
        )
        self.weight_hh = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size)) for _ in range(num_layers)]
        )
        if bias:
            biases = []
            for i in range(num_layers):
                bias = torch.randn(3, hidden_size) * 1e-5
                bias = gt.ManifoldParameter(bias, manifold=self.manifold)
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)
        self.nonlin = nonlin
        self.reset_parameters()

    def reset_parameters(self):
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            rows, cols = weight.size()
            stdv = (6 / (rows / 3 + cols)) ** 0.5  # xavier uniform
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0=None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input[:2]
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)
        if h0 is None:
            h0 = input.new_zeros(
                self.num_layers, max_batch_size, self.hidden_size, requires_grad=False
            )
        h0 = h0.unbind(0)
        if self.bias is not None:
            biases = self.bias
        else:
            biases = (None,) * self.num_layers
        outputs = []
        last_states = []
        out = input
        for i in range(self.num_layers):
            out, h_last = eucl_gru_loop(
                input=out,
                h0=h0[i],
                weight_ih=self.weight_ih[i],
                weight_hh=self.weight_hh[i],
                bias=biases[i],
                nonlin=self.nonlin,
                batch_sizes=batch_sizes,
            )
            outputs.append(out)
            last_states.append(h_last)
        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack(last_states)
        # default api assumes
        # out: (seq_len, batch, num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        # if packed:
        # out: (sum(seq_len), num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        return out, ht

    def extra_repr(self):
        return ("{input_size}, {hidden_size}, {num_layers}, bias={bias}, "
                "").format(**self.__dict__, self=self, bias=self.bias is not None)


class MobiusLinear(nn.Linear):
    def __init__(self, *args, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball = gt.PoincareBall(c=c)
        if self.bias is not None:
            if hyperbolic_bias:
                self.bias = gt.ManifoldParameter(self.bias, manifold=self.ball)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() * 1e-3, k=self.ball.k))
        with torch.no_grad():
            fin, fout = self.weight.size()
            k = (6 / (fin + fout)) ** 0.5  # xavier uniform
            self.weight.uniform_(-k, k)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            k=self.ball.k,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += ", hyperbolic_input={}".format(self.hyperbolic_input)
        if self.bias is not None:
            info += ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info


class MobiusConcat(nn.Module):
    def __init__(self, output_dim, input_dims, second_input_dim=None, third_input_dim=None, nonlin=None):
        super(MobiusConcat, self).__init__()
        b_input_dims = second_input_dim if second_input_dim is not None else input_dims

        self.lin_a = MobiusLinear(input_dims, output_dim, bias=False, nonlin=nonlin)
        self.lin_b = MobiusLinear(b_input_dims, output_dim, bias=False, nonlin=nonlin)

        if third_input_dim:
            self.lin_c = MobiusLinear(third_input_dim, output_dim, bias=False, nonlin=nonlin)

        self.ball = gt.PoincareBall()
        b = torch.randn(output_dim) * 1e-5
        self.bias = gt.ManifoldParameter(pmath.expmap0(b, k=self.ball.k), manifold=self.ball)

    def forward(self, input_a, input_b, third_input=None):
        """
        :param input_a: batch x * x input_dim_a
        :param input_b: batch x * x input_dim_b
        :return: batch x output_dim
        """
        out_a = self.lin_a(input_a)
        out_b = self.lin_b(input_b)
        out_total = self.add(out_a, out_b)

        if third_input is not None:
            out_c = self.lin_c(third_input)
            out_total = self.add(out_total, out_c)

        out_total = self.add(out_total, self.bias)
        return out_total

    def add(self, a, b):
        out = pmath.mobius_add(a, b, k=self.ball.k)
        return pmath.project(out, k=self.ball.k)


class EuclConcat(nn.Module):
    def __init__(self, output_dim, input_dims, second_input_dim=None, third_input_dim=None, nonlin=None):
        super(EuclConcat, self).__init__()
        b_input_dims = second_input_dim if second_input_dim is not None else input_dims

        self.lin_a = nn.Linear(input_dims, output_dim, bias=False)
        self.lin_b = nn.Linear(b_input_dims, output_dim, bias=False)

        if third_input_dim:
            self.lin_c = nn.Linear(third_input_dim, output_dim, bias=False)

        self.manifold = gt.Euclidean()
        self.bias = gt.ManifoldParameter(torch.randn(output_dim) * 1e-5, manifold=self.manifold)
        self.nonlin = nonlin if nonlin is not None else lambda x: x

    def forward(self, input_a, input_b, third_input=None):
        """
        :param input_a: batch x * x input_dim_a
        :param input_b: batch x * x input_dim_b
        :return: batch x output_dim
        """
        out_a = self.nonlin(self.lin_a(input_a))
        out_b = self.nonlin(self.lin_b(input_b))
        out_total = out_a + out_b

        if third_input is not None:
            out_c = self.nonlin(self.lin_c(third_input))
            out_total += out_c

        out_total += self.bias
        return out_total


class MobiusMLR(nn.Module):
    """
    Multinomial logistic regression in the Poincare Ball

    It is based on formulating logits as distances to margin hyperplanes.
    In Euclidean space, hyperplanes can be specified with a point of origin
    and a normal vector. The analogous notion in hyperbolic space for a
    point $p \in \mathbb{D}^n$ and
    $a \in T_{p} \mathbb{D}^n \backslash \{0\}$ would be the union of all
    geodesics passing through $p$ and orthogonal to $a$. Given $K$ classes
    and $k \in \{1,...,K\}$, $p_k \in \mathbb{D}^n$,
    $a_k \in T_{p_k} \mathbb{D}^n \backslash \{0\}$, the formula for the
    hyperbolic MLR is:

    \begin{equation}
        p(y=k|x) f\left(\lambda_{p_k} \|a_k\| \operatorname{sinh}^{-1} \left(\frac{2 \langle -p_k \oplus x, a_k\rangle}
                {(1 - \| -p_k \oplus x \|^2)\|a_k\|} \right) \right)
    \end{equation}

    """

    def __init__(self, in_features, out_features, c=1.0):
        """
        :param in_features: number of dimensions of the input
        :param out_features: number of classes
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = gt.PoincareBall(c=c)
        points = torch.randn(out_features, in_features) * 1e-5
        points = pmath.expmap0(points, k=self.ball.k)
        self.p_k = gt.ManifoldParameter(points, manifold=self.ball)

        tangent = torch.Tensor(out_features, in_features)
        stdv = (6 / (out_features + in_features)) ** 0.5  # xavier uniform
        torch.nn.init.uniform_(tangent, -stdv, stdv)
        self.a_k = torch.nn.Parameter(tangent)

    def forward(self, input):
        """
        :param input: batch x space_dim: points (features) in the Poincaré ball
        :return: batch x classes: logit of probabilities for 'out_features' classes
        """
        input = input.unsqueeze(-2)     # batch x aux x space_dim
        distance, a_norm = self._dist2plane(x=input, p=self.p_k, a=self.a_k, c=self.ball.c, k=self.ball.k, signed=True)
        result = 2 * a_norm * distance
        return result

    def _dist2plane(self, x, a, p, c, k, keepdim: bool = False, signed: bool = False, dim: int = -1):
        """
        Taken from geoopt and corrected so it returns a_norm and this value does not have to be calculated twice
        """
        sqrt_c = c ** 0.5
        minus_p_plus_x = pmath.mobius_add(-p, x, k=k, dim=dim)
        mpx_sqnorm = minus_p_plus_x.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
        mpx_dot_a = (minus_p_plus_x * a).sum(dim=dim, keepdim=keepdim)
        if not signed:
            mpx_dot_a = mpx_dot_a.abs()
        a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
        num = 2 * sqrt_c * mpx_dot_a
        denom = (1 - c * mpx_sqnorm) * a_norm
        return pmath.arsinh(num / denom.clamp_min(MIN_NORM)) / sqrt_c, a_norm

    def extra_repr(self):
        return "in_features={in_features}, out_features={out_features}".format(**self.__dict__) + f" k={self.ball.k}"


class EuclMLR(nn.Module):
    """Euclidean Multinomial logistic regression"""

    def __init__(self, in_features, out_features):
        """
        :param in_features: number of dimensions of the input
        :param out_features: number of classes
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = gt.Euclidean()
        points = torch.randn(out_features, in_features) * 1e-5
        self.p_k = gt.ManifoldParameter(points, manifold=self.manifold)

        tangent = torch.Tensor(out_features, in_features)
        stdv = (6 / (out_features + in_features)) ** 0.5  # xavier uniform
        torch.nn.init.uniform_(tangent, -stdv, stdv)
        self.a_k = torch.nn.Parameter(tangent)

    def forward(self, input):
        """
        :param input: batch x space_dim: points (features) in the Poincaré ball
        :return: batch x classes: logit of probabilities for 'out_features' classes
        """
        x = input.unsqueeze(dim=-2)
        minus_p_plus_x = -self.p_k + x
        result = (minus_p_plus_x * self.a_k).sum(dim=-1, keepdim=False)
        return 4 * result

    def extra_repr(self):
        return "in_features={in_features}, out_features={out_features}".format(**self.__dict__)


def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None,
    k=-1.0,
):
    if hyperbolic_input:
        output = pmath.mobius_matvec(weight, input, k=k)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = pmath.expmap0(output, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, k=k)
        output = pmath.mobius_add(output, bias, k=k)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, k=k)
    output = pmath.project(output, k=k)
    return output


def one_hyperb_rnn_transform(W, h, U, x, b, k):
    W_otimes_h = pmath.mobius_matvec(W, h, k=k)
    U_otimes_x = pmath.mobius_matvec(U, x, k=k)
    Wh_plus_Ux = pmath.mobius_add(W_otimes_h, U_otimes_x, k=k)
    return pmath.mobius_add(Wh_plus_Ux, b, k=k)


def one_eucl_rnn_transform(W, h, U, x, b):
    W_otimes_h = torch.tensordot(h, W, dims=([-1], [1]))
    U_otimes_x = torch.tensordot(x, U, dims=([-1], [1]))
    return W_otimes_h + U_otimes_x + b


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    k: torch.Tensor,
    nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = pmath.logmap0(one_hyperb_rnn_transform(W_hz, hx, W_iz, input, b_z, k), k=k).sigmoid()
    r_t = pmath.logmap0(one_hyperb_rnn_transform(W_hr, hx, W_ir, input, b_r, k), k=k).sigmoid()

    rh_t = pmath.mobius_pointwise_mul(r_t, hx, k=k)
    h_tilde = one_hyperb_rnn_transform(W_hh, rh_t, W_ih, input, b_h, k)

    if nonlin is not None:
        h_tilde = pmath.mobius_fn_apply(nonlin, h_tilde, k=k)
    delta_h = pmath.mobius_add(-hx, h_tilde, k=k)
    h_out = pmath.mobius_add(hx, pmath.mobius_pointwise_mul(z_t, delta_h, k=k), k=k)
    return h_out


def eucl_gru_cell(
        input: torch.Tensor,
        hx: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias: torch.Tensor,
        nonlin=None
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = one_eucl_rnn_transform(W_hz, hx, W_iz, input, b_z).sigmoid()
    r_t = one_eucl_rnn_transform(W_hr, hx, W_ir, input, b_r).sigmoid()

    rh_t = r_t * hx
    h_tilde = one_eucl_rnn_transform(W_hh, rh_t, W_ih, input, b_h)

    if nonlin is not None:
        h_tilde = nonlin(h_tilde)

    delta_h = -hx + h_tilde
    h_out = hx + z_t * delta_h
    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    k: torch.Tensor,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
    nonlin=None,
):
    if not hyperbolic_hidden_state0:
        hx = pmath.expmap0(h0, k=k)
    else:
        hx = h0
    if not hyperbolic_input:
        input = pmath.expmap0(input, k=k)
    outs = []
    if batch_sizes is None:
        input_unbinded = input.unbind(0)
        for t in range(input.size(0)):
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                k=k,
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
            hx = mobius_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                k=k,
            )
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t+1]], hx[batch_sizes[t+1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last


def eucl_gru_loop(
        input: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias: torch.Tensor,
        batch_sizes=None,
        nonlin=None):
    hx = h0
    outs = []
    if batch_sizes is None:
        input_unbinded = input.unbind(0)
        for t in range(input.size(0)):
            hx = eucl_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t]:]
            hx = eucl_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin
            )
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t + 1]], hx[batch_sizes[t + 1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last


def poincare2klein(x, c=1.0, dim=-1):
    r"""
    Maps points from Poincare model to Klein model
    Parameters:
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    dim : int
        reduction dimension for operations
    Returns
    -------
    tensor
        points in Klein model, :math:`\mathbf{x}_{\mathbb{K}}=\frac{2 \mathbf{x}_{\mathbb{D}}}{1+c\left\|\mathbf{x}_{\mathbb{D}}\right\|^{2}}`
    """
    denom = 1.0 + c * x.pow(2).sum(dim, keepdim=True)
    return 2.0 * x / denom


def klein2poincare(x, c=1.0, dim=-1):
    r"""
    Maps points from Klein model to Poincare model
    Parameters:
    ----------
    x : tensor
        point on the Klein model
    c : float|tensor
        ball negative curvature
    dim : int
        reduction dimension for operations
    Returns
    -------
    tensor
        points in Poincare: \mathbf{x}_{\mathbb{D}}=\frac{\mathbf{X}_{\mathbb{K}}}{1+\sqrt{1-c\left\|\mathbf{x}_{\mathbb{K}}\right\|^{2}}}
    """
    denom = 1.0 + torch.sqrt(1.0 - c * x.pow(2).sum(dim, keepdim=True))
    return x / denom


def lorentz_factor(x, c=1.0, dim=-1, keepdim=False):
    """
    Parameters
    ----------
    x : tensor
        point on Klein disk
    c : float
        negative curvature
    dim : int
        dimension to calculate Lorenz factor
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Lorentz factor
    """
    return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))


def einstein_midpoint(x, c=1.0):
    r"""
    Finds the Einstein midpoint, analogue of finding average over features in Euclidean space
    Parameters:
    ----------
    x : tensor
        point on the Poincare ball. The points are assumed to be in the last dim
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        midpoint
    """
    x = poincare2klein(x, c)
    factors = lorentz_factor(x, c=c, keepdim=True)
    midpoint = torch.sum(factors * x, dim=1, keepdim=True) / torch.sum(factors, dim=1, keepdim=True)
    midpoint = klein2poincare(midpoint, c)
    return midpoint.squeeze(dim=1)


def weighted_einstein_midpoint(x, w, c=1.0):
    r"""
    Finds the Einstein midpoint, analogue of finding average over features in Euclidean space.
    The input and output of this function are points in the Poincare ball, but internally the points are converted
    to the klein model of hyperbolic space to calculate the einstein midpoint.
    Parameters:
    ----------
    x : tensor
        point on the Poincare ball. The points are assumed to be in the last dim: B x seq_len x space_dim
    w : tensor
        weights for midpoint. B x seq_len
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        midpoint
    """
    x = poincare2klein(x, c)
    factors = lorentz_factor(x, c=c, keepdim=True)  # batch x seq_len
    weighted_factors = factors * w
    midpoint = torch.sum(weighted_factors * x, dim=1, keepdim=True) / torch.sum(weighted_factors, dim=1, keepdim=True)
    midpoint = klein2poincare(midpoint, c)
    return midpoint.squeeze(dim=1)


def mobius_midpoint(x, c=1.0):
    r"""
    Finds the Mobius midpoint, analogue of finding average over features in Euclidean space
    Parameters:
    ----------
    x : tensor
        point on the Poincare ball. The points are assumed to be in the last dim
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        midpoint
    """
    sq_factors = lorentz_factor(x, c=c, keepdim=True) ** 2  # batch x seq_len
    midpoint = torch.sum(sq_factors * x, dim=1, keepdim=True) / torch.sum(sq_factors - 0.5, dim=1, keepdim=True)
    midpoint = pmath.mobius_scalar_mul(torch.Tensor([0.5]).to(x.device), midpoint, k=torch.Tensor([-c]).to(x.device))
    return midpoint.squeeze(dim=1)


def weighted_mobius_midpoint(x, w, c=1.0):
    r"""
    Finds the Mobius midpoint, analogue of finding weighted average over features in Euclidean space
    Parameters:
    ----------
    x : tensor
        point on the Poincare ball. The points are assumed to be in the last dim
    w : tensor
        weights for midpoint. B x seq_len
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        midpoint
    """
    sq_factors = lorentz_factor(x, c=c, keepdim=True) ** 2  # batch x seq_len
    weighted_factors = w * sq_factors
    midpoint = torch.sum(weighted_factors * x, dim=1, keepdim=True) / torch.sum(w * (sq_factors - 0.5), dim=1, keepdim=True)
    midpoint = pmath.mobius_scalar_mul(torch.Tensor([0.5]).to(x.device), midpoint, k=torch.Tensor([-c]).to(x.device))
    return midpoint.squeeze(dim=1)


def euclidean_midpoint(x):
    """
    :param x: b x seq x dim
    :return: b x dim
    """
    total = x.size(1)
    return x.sum(dim=1) / total


def weighted_euclidean_midpoint(x, w):
    """
    :param x: b x seq x dim
    :param w: b x seq x 1
    :return: b x dim
    """
    total = w.sum(dim=1)
    return (x * w).sum(dim=1) / total
