import torch
GRAVITY_VEC = torch.tensor([ 0.,  0., -1.],
                           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                           dtype = torch.float)

@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """通过四元数的逆旋转一个向量，沿 q 和 v 的最后一个维度进行操作。

    Args:
        q (torch.Tensor): 四元数，形状为 (..., 4)，表示为 (w, x, y, z)。
        v (torch.Tensor): 要旋转的向量，形状为 (..., 3)，表示为 (x, y, z)。

    Returns:
        torch.Tensor: 旋转后的向量，形状为 (..., 3)，表示为 (x, y, z)。

    详细步骤：
    1. **提取四元数的分量**:
       - `q_w`: 四元数的实部 w，形状为 (..., )。
       - `q_vec`: 四元数的虚部 (x, y, z)，形状为 (..., 3)。

    2. **计算旋转公式中的各个部分**:
       - `a`: 计算公式的第一项 \( v \cdot (2q_w^2 - 1) \)。这里使用了广播机制来确保形状匹配。
       - `b`: 计算公式中的第二项 \( 2 \cdot q_w \cdot (\text{cross}(q_{\text{vec}}, v)) \)。这里使用了 `torch.cross` 来计算向量叉积，并乘以 `q_w`。
       - `c`: 计算公式中的第三项 \( 2 \cdot q_{\text{vec}} \cdot (\text{dot}(q_{\text{vec}}, v)) \)。这里根据 `q_vec` 的维度选择不同的实现方式：
         - 如果 `q_vec` 是二维张量（即批次大小为第一维），使用 `torch.bmm` 进行批量矩阵乘法，因为其效率更高。
         - 否则，使用 `torch.einsum` 进行爱因斯坦求和，适用于任意维度的张量。

    3. **组合结果**:
       - 最终结果为 \( a - b + c \)，即旋转后的向量。

    公式解释：
    四元数的逆旋转公式为：
    \[
    v' = v + 2 \cdot q_w \cdot (\text{cross}(q_{\text{vec}}, v)) + 2 \cdot \text{dot}(q_{\text{vec}}, v) \cdot q_{\text{vec}}
    \]
    其中 \( q_{\text{vec}} \) 是四元数的虚部，\( q_w \) 是四元数的实部，\( v \) 是要旋转的向量。

    注意：
    - 四元数的逆可以通过将虚部取反得到，即 \( q^{-1} = (w, -x, -y, -z) \)。
    - 该函数实现了四元数逆旋转的高效版本，避免直接构造逆四元数并进行两次旋转操作。
    """
    q_w = q[..., 0]  # 提取四元数的实部
    q_vec = q[..., 1:]  # 提取四元数的虚部

    # 计算公式的第一项：v * (2 * q_w^2 - 1)
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)

    # 计算公式中的第二项：2 * q_w * cross(q_vec, v)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0

    # 根据 q_vec 的维度选择不同的实现方式计算公式中的第三项
    if q_vec.dim() == 2:
        # 对于二维张量，使用 torch.bmm 进行批量矩阵乘法
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        # 对于其他维度的张量，使用 torch.einsum 进行爱因斯坦求和
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0

    # 返回最终的旋转结果
    return a - b + c