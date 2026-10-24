"""The shared CSV dataset for both absorber families."""

from .data_des import MyDataset

Dataset = MyDataset
metal = MyDataset

__all__ = ["Dataset", "MyDataset", "metal"]
