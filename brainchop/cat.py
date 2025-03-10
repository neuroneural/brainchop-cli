from tinygrad.tensor import Tensor
from typing import List

def concatenate(l: List[Tensor], axis: int, eager: bool = True) -> Tensor:
    """Concatenate tensors along specified axis using binary tree partitioning"""
    def cat_pair(a: Tensor, b: Tensor) -> Tensor:
        """Concatenate two tensors along the specified axis"""
        assert axis < len(a.shape) and axis < len(b.shape), "joined axis must be within tensor shape"
        ashift = [(0, b.shape[axis]) if i == axis else (0, 0) for i, _ in enumerate(a.shape)]
        bshift = [(a.shape[axis], 0) if i == axis else (0, 0) for i, _ in enumerate(a.shape)]
        result = a.pad(ashift) + b.pad(bshift)
        return result.realize() if eager else result
    
    def tree_concat(tensors: List[Tensor]) -> Tensor:
        """Recursively concatenate using divide-and-conquer"""
        n = len(tensors)
        if n == 1:
            return tensors[0]
        if n == 2:
            return cat_pair(tensors[0], tensors[1])
        mid = n // 2
        return cat_pair(tree_concat(tensors[:mid]), tree_concat(tensors[mid:]))
    
    return tree_concat(l)

# Example usage
if __name__ == "__main__":
    l = [Tensor.randn(1, 256, 256, 7) for i in range(256)]
    print(concatenate(l, 0, True).shape)  # (256, 256, 256, 7)
