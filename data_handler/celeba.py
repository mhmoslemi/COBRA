# import csv
# import os
# from collections import namedtuple
# from typing import Any, Callable, List, Optional, Tuple, Union, TypeVar, Iterable
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torch

# T = TypeVar("T", str, bytes)
# CSV = namedtuple("CSV", ["header", "index", "data"])


# class CelebA_train(Dataset):
#     """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

#     Args:
#         root (string): Root directory where images are downloaded to.
#         split (string): One of {'train', 'valid', 'test', 'all'}.
#             Accordingly dataset is selected.
#         target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
#             or ``landmarks``. Can also be a list to output a tuple with all specified target types.
#             The targets represent:

#                 - ``attr`` (Tensor shape=(40,) dtype=int): binary (0, 1) labels for attributes
#                 - ``identity`` (int): label for each person (data points with the same identity are the same person)
#                 - ``bbox`` (Tensor shape=(4,) dtype=int): bounding box (x, y, width, height)
#                 - ``landmarks`` (Tensor shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
#                   righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

#             Defaults to ``attr``. If empty, ``None`` will be returned as target.

#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.PILToTensor``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#     """

#     def __init__(
#             self,
#             target_label_idx: int,
#             sensitive_label_idx: int,
#             root: str,
#             split: str = "train",
#             target_type: Union[List[str], str] = "attr",
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             download: bool = False,
#     ) -> None:
#         super(CelebA_train, self).__init__()
#         self.target_label_idx = target_label_idx
#         self.sensitive_label_idx = sensitive_label_idx
#         self.root = root
#         self.split = split
#         if isinstance(target_type, list):
#             self.target_type = target_type
#         else:
#             self.target_type = [target_type]

#         self.transform = transform
#         self.target_transform = target_transform
#         if not self.target_type and self.target_transform is not None:
#             raise RuntimeError("target_transform is specified but target_type is empty")

#         split_map = {
#             "train": 0,
#             "valid": 1,
#             "test": 2,
#             "all": None,
#         }
#         split_ = split_map[split]
#         splits = self._load_csv("list_eval_partition.txt")
#         identity = self._load_csv("identity_CelebA.txt")
#         bbox = self._load_csv("list_bbox_celeba.txt", header=1)
#         landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
#         attr = self._load_csv("list_attr_celeba.txt", header=1)

#         mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
#         if mask == slice(None):  # if split == "all"
#             self.filename = splits.index
#         else:
#             self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
#         # self.identity = identity.data[mask]
#         self.bbox = bbox.data[mask]
#         self.landmarks_align = landmarks_align.data[mask]
#         self.attr = attr.data[mask]
#         # map from {-1, 1} to {0, 1}
#         self.attr = torch.floor_divide(self.attr + 1, 2)
#         self.attr_names = attr.header

#     def _load_csv(
#             self,
#             filename: str,
#             header: Optional[int] = None,
#     ) -> CSV:
#         with open(os.path.join(self.root, filename)) as csv_file:
#             data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

#         if header is not None:
#             headers = data[header]
#             data = data[header + 1:]
#         else:
#             headers = []
#         indices = [row[0] for row in data]
#         data = [row[1:] for row in data]
#         data_int = [list(map(int, i)) for i in data]

#         return CSV(headers, indices, torch.tensor(data_int))

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         X = Image.open(os.path.join(self.root, "", self.filename[index]))

#         target: Any = []
#         for t in self.target_type:
#             if t == "attr":
#                 target.append(self.attr[index, :])
#             elif t == "identity":
#                 target.append(self.identity[index, 0])
#             elif t == "bbox":
#                 target.append(self.bbox[index, :])
#             elif t == "landmarks":
#                 target.append(self.landmarks_align[index, :])
#             else:
#                 # TODO: refactor with utils.verify_str_arg
#                 raise ValueError(f'Target type "{t}" is not recognized.')

#         if self.transform is not None:
#             X = self.transform(X)

#         if target:
#             target = tuple(target) if len(target) > 1 else target[0]

#             if self.target_transform is not None:
#                 target = self.target_transform(target)
#         else:
#             target = None

#         # return X,target[2],target[20] # 有无吸引力/性别
#         # return X,target[2],target[20] # 有无吸引力/性别
#         return X, target[self.target_label_idx], target[self.sensitive_label_idx]

#     def __len__(self) -> int:
#         return len(self.attr)


# import numpy as np
# import random


# class CelebA_test(Dataset):
#     """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

#     Args:
#         root (string): Root directory where images are downloaded to.
#         split (string): One of {'train', 'valid', 'test', 'all'}.
#             Accordingly dataset is selected.
#         target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
#             or ``landmarks``. Can also be a list to output a tuple with all specified target types.
#             The targets represent:

#                 - ``attr`` (Tensor shape=(40,) dtype=int): binary (0, 1) labels for attributes
#                 - ``identity`` (int): label for each person (data points with the same identity are the same person)
#                 - ``bbox`` (Tensor shape=(4,) dtype=int): bounding box (x, y, width, height)
#                 - ``landmarks`` (Tensor shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
#                   righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

#             Defaults to ``attr``. If empty, ``None`` will be returned as target.

#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.PILToTensor``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#     """

#     def __init__(
#             self,
#             target_label_idx: int,
#             sensitive_label_idx: int,
#             root: str,
#             split: str = "train",
#             target_type: Union[List[str], str] = "attr",
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             download: bool = False,
#     ) -> None:
#         super(CelebA_test, self).__init__()
#         self.target_label_idx = target_label_idx
#         self.sensitive_label_idx = sensitive_label_idx
#         self.root = root
#         self.split = split
#         if isinstance(target_type, list):
#             self.target_type = target_type
#         else:
#             self.target_type = [target_type]

#         self.transform = transform
#         self.target_transform = target_transform
#         if not self.target_type and self.target_transform is not None:
#             raise RuntimeError("target_transform is specified but target_type is empty")

#         split_map = {
#             "train": 0,
#             "valid": 1,
#             "test": 2,
#             "all": None,
#         }
#         split_ = split_map[split]
#         splits = self._load_csv("list_eval_partition.txt")

#         identity = self._load_csv("identity_CelebA.txt")
#         bbox = self._load_csv("list_bbox_celeba.txt", header=1)
#         landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
#         attr = self._load_csv("list_attr_celeba.txt", header=1)

#         mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

#         if mask == slice(None):  # if split == "all"
#             self.filename = splits.index
#         else:
#             self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

#         # self.identity = identity.data[mask]
#         self.bbox = bbox.data[mask]
#         self.landmarks_align = landmarks_align.data[mask]
#         self.attr = attr.data[mask]
#         # self.identity = identity[mask]

#         # map from {-1, 1} to {0, 1}
#         self.attr = torch.floor_divide(self.attr + 1, 2)
#         self.attr_names = attr.header

#         lab_idx_dict = {}
#         col_idx_dict = {}

#         lab = self.attr[:, self.target_label_idx]
#         col = self.attr[:, self.sensitive_label_idx]

#         for lab_id in np.unique(lab):
#             lab_idx_dict[lab_id] = [idx for idx, c in enumerate(lab) if lab_id == c]
#         for col_id in np.unique(col):
#             col_idx_dict[col_id] = [idx for idx, c in enumerate(col) if col_id == c]

#         test_idx = []
#         min_intersection = 1e10
#         for i in range(len(lab_idx_dict)):
#             for j in range(len(col_idx_dict)):
#                 intersection = list(set(lab_idx_dict[i]) & set(col_idx_dict[j]))
#                 min_intersection = min(min_intersection, len(intersection))

#         for i in range(len(lab_idx_dict)):
#             for j in range(len(col_idx_dict)):
#                 intersection = list(set(lab_idx_dict[i]) & set(col_idx_dict[j]))
#                 min_intersection = min(min_intersection, len(intersection))
#                 # print(i, j, min_intersection, len(intersection))
#                 select_idx = random.sample(intersection, min(len(intersection), min_intersection))
#                 test_idx.extend(select_idx)
#                 lab_idx_dict[i] = list(set(lab_idx_dict[i]) - set(select_idx))
#                 col_idx_dict[j] = list(set(col_idx_dict[j]) - set(select_idx))

#         self.attr = self.attr[test_idx]
#         temp = []
#         for i in test_idx:
#             temp.append(self.filename[i])
#         self.filename = temp
#         self.bbox = self.bbox[test_idx]
#         self.landmarks_align = self.landmarks_align[test_idx]
#         # self.identity = self.identity[test_idx]

#     def _load_csv(
#             self,
#             filename: str,
#             header: Optional[int] = None,
#     ) -> CSV:
#         with open(os.path.join(self.root, filename)) as csv_file:
#             data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

#         if header is not None:
#             headers = data[header]
#             data = data[header + 1:]
#         else:
#             headers = []
#         indices = [row[0] for row in data]
#         data = [row[1:] for row in data]
#         data_int = [list(map(int, i)) for i in data]

#         return CSV(headers, indices, torch.tensor(data_int))

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         X = Image.open(os.path.join(self.root, "", self.filename[index]))

#         target: Any = []
#         for t in self.target_type:
#             if t == "attr":
#                 target.append(self.attr[index, :])
#             elif t == "identity":
#                 target.append(self.identity[index, 0])
#             elif t == "bbox":
#                 target.append(self.bbox[index, :])
#             elif t == "landmarks":
#                 target.append(self.landmarks_align[index, :])
#             else:
#                 # TODO: refactor with utils.verify_str_arg
#                 raise ValueError(f'Target type "{t}" is not recognized.')

#         if self.transform is not None:
#             X = self.transform(X)

#         if target:
#             target = tuple(target) if len(target) > 1 else target[0]

#             if self.target_transform is not None:
#                 target = self.target_transform(target)
#         else:
#             target = None

#         # return X, target[2], target[20]  # 有无吸引力/性别
#         # return X,target[31],target[20] # 微笑/性别
#         return X, target[self.target_label_idx], target[self.sensitive_label_idx]

#     def __len__(self) -> int:
#         return len(self.attr)


# def iterable_to_str(iterable: Iterable) -> str:
#     return "'" + "', '".join([str(item) for item in iterable]) + "'"


# def CelebA(target_label_idx, sensitive_label_idx, data_dir="/mnt/DatasetCondensation-master/data/celeba"):
#     mean = (0.5063, 0.4258, 0.3832)
#     std = (0.2676, 0.2453, 0.2410)
#     from torchvision import transforms
#     transform = transforms.Compose([
#         transforms.CenterCrop(178),
#         transforms.Resize(64),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std), ])
#     train_dataset = CelebA_train(target_label_idx, sensitive_label_idx, root=data_dir, split='train',
#                                  transform=transform)
#     test_dataset = CelebA_test(target_label_idx, sensitive_label_idx, root=data_dir, split='test', transform=transform)
#     return train_dataset, test_dataset, mean, std


# if __name__ == "__main__":
#     # target_label_idx = 31  # 微笑
#     # target_label_idx = 39  # 年轻
#     target_label_idx = 33  # 卷发
#     # target_label_idx=2 # 有无吸引力
#     sensitive_label_idx = 20  # 性别

#     for i in range(0,40):

#         target_label_idx=i
#         print(target_label_idx)
#         train_dataset, test_dataset, mean, std = CelebA(target_label_idx, sensitive_label_idx,
#                                                         data_dir="/mnt/DatasetCondensation-master/data/celeba")
#         # dataloader = DataLoader(train_dataset, batch_size=32)
#         # for x, tra, m in dataloader:
#         #     print(x.shape)
#         #     print(tra, m)

#         train_target=train_dataset.attr[:, target_label_idx]
#         train_sensitive=train_dataset.attr[:, sensitive_label_idx]

#         test_target=test_dataset.attr[:, target_label_idx]
#         test_sensitive=test_dataset.attr[:, sensitive_label_idx]

#         aa=0
#         ab=0
#         ba=0
#         bb=0

#         for index in range(len(train_target)):
#             target=train_target[index]
#             sensitive=train_sensitive[index]
#             if target==0 and sensitive==0:
#                 aa=aa+1
#             if target==0 and sensitive==1:
#                 ab=ab+1
#             if target==1 and sensitive==0:
#                 ba=ba+1
#             if target==1 and sensitive==1:
#                 bb=bb+1

#         print((aa,ab,ba,bb))

#         aa = 0
#         ab = 0
#         ba = 0
#         bb = 0

#         for index in range(len(test_target)):
#             target=test_target[index]
#             sensitive=test_sensitive[index]
#             if target==0 and sensitive==0:
#                 aa=aa+1
#             if target==0 and sensitive==1:
#                 ab=ab+1
#             if target==1 and sensitive==0:
#                 ba=ba+1
#             if target==1 and sensitive==1:
#                 bb=bb+1

#         print((aa,ab,ba,bb))
#         print("--------------------------")







# # ########## !!!!!!! # ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!
# # ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!
# # ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!
# # ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!# ########## !!!!!!!

# # celeba_cached_full.py
# # Goal:
# # 1) First run: read all CelebA images once, preprocess (crop, resize), store as a single .pt cache.
# # 2) Later runs: load the .pt once (no per-image disk reads).
# # 3) Training: either (A) CPU RAM + pinned memory + async GPU transfer (recommended),
# #    or (B) preload the whole split to GPU RAM if it fits.

# # import os
# # import csv
# # import random
# # from collections import namedtuple
# # from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

# # import torch
# # from torch.utils.data import Dataset, DataLoader
# # from PIL import Image

# # CSV = namedtuple("CSV", ["header", "index", "data"])


# # # -----------------------------
# # # CSV + split helpers
# # # -----------------------------
# # def _load_csv(root: str, filename: str, header: Optional[int] = None) -> CSV:
# #     path = os.path.join(root, filename)
# #     with open(path) as f:
# #         data = list(csv.reader(f, delimiter=" ", skipinitialspace=True))

# #     if header is not None:
# #         headers = data[header]
# #         data = data[header + 1 :]
# #     else:
# #         headers = []

# #     indices = [row[0] for row in data]
# #     rows = [row[1:] for row in data]
# #     rows_int = [list(map(int, r)) for r in rows]
# #     return CSV(headers, indices, torch.tensor(rows_int, dtype=torch.int64))


# # def _get_split_filenames_and_attr(root: str, split: str) -> Tuple[List[str], torch.Tensor]:
# #     split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
# #     if split not in split_map:
# #         raise ValueError(f"split must be one of {list(split_map.keys())}, got: {split}")

# #     split_ = split_map[split]
# #     splits = _load_csv(root, "list_eval_partition.txt")
# #     attr = _load_csv(root, "list_attr_celeba.txt", header=1)

# #     if split_ is None:
# #         filenames = splits.index
# #         attr_tensor = attr.data
# #     else:
# #         mask = (splits.data == split_).squeeze()
# #         idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
# #         filenames = [splits.index[i] for i in idx]
# #         attr_tensor = attr.data[mask]

# #     # map {-1, 1} -> {0, 1}
# #     attr_tensor = torch.floor_divide(attr_tensor + 1, 2).to(torch.int64)
# #     return filenames, attr_tensor


# # # -----------------------------
# # # Cache builder
# # # -----------------------------
# # class _CelebAReader(Dataset):
# #     """
# #     Reads images from disk and applies cache_transform.
# #     Used only while building the cache.
# #     """
# #     def __init__(self, root: str, filenames: List[str], cache_transform: Callable):
# #         self.root = root
# #         self.filenames = filenames
# #         self.cache_transform = cache_transform

# #     def __len__(self) -> int:
# #         return len(self.filenames)

# #     def __getitem__(self, idx: int) -> torch.Tensor:
# #         img_path = os.path.join(self.root, "", self.filenames[idx])
# #         img = Image.open(img_path).convert("RGB")
# #         x = self.cache_transform(img)  # uint8 tensor [3,H,W]
# #         return x


# # def build_or_load_celeba_cache(
# #     root: str,
# #     split: str,
# #     cache_dir: str,
# #     image_size: int = 64,
# #     center_crop: int = 178,
# #     batch_size: int = 256,
# #     num_workers: int = 8,
# # ) -> dict:
# #     """
# #     Cache format:
# #       {
# #         "images_u8": uint8 tensor [N,3,H,W] (0..255),
# #         "attr":      int64 tensor [N,40] (0/1),
# #         "meta":      dict
# #       }
# #     """
# #     os.makedirs(cache_dir, exist_ok=True)
# #     cache_path = os.path.join(
# #         cache_dir, f"celeba_{split}_c{center_crop}_s{image_size}_uint8.pt"
# #     )

# #     if os.path.exists(cache_path):
# #         cache = torch.load(cache_path, map_location="cpu")
# #         meta = cache.get("meta", {})
# #         ok = (
# #             meta.get("split") == split
# #             and meta.get("image_size") == image_size
# #             and meta.get("center_crop") == center_crop
# #         )
# #         if not ok:
# #             raise RuntimeError(
# #                 f"Cache exists but meta mismatches requested settings.\n"
# #                 f"Cache: {cache_path}\nMeta: {meta}"
# #             )
# #         return cache

# #     # lazy import to keep file self-contained
# #     from torchvision import transforms

# #     filenames, attr = _get_split_filenames_and_attr(root, split)

# #     cache_transform = transforms.Compose(
# #         [
# #             transforms.CenterCrop(center_crop),
# #             transforms.Resize(image_size),
# #             transforms.PILToTensor(),  # uint8 [C,H,W]
# #         ]
# #     )

# #     reader = _CelebAReader(root, filenames, cache_transform)
# #     loader = DataLoader(
# #         reader,
# #         batch_size=batch_size,
# #         shuffle=False,
# #         num_workers=num_workers,
# #         pin_memory=False,
# #         persistent_workers=(num_workers > 0),
# #     )

# #     n = len(reader)
# #     images_u8 = torch.empty((n, 3, image_size, image_size), dtype=torch.uint8)

# #     offset = 0
# #     for batch in loader:
# #         b = batch.size(0)
# #         images_u8[offset : offset + b].copy_(batch)
# #         offset += b

# #     cache = {
# #         "images_u8": images_u8.contiguous(),
# #         "attr": attr.contiguous(),
# #         "meta": {
# #             "split": split,
# #             "image_size": image_size,
# #             "center_crop": center_crop,
# #             "dtype": "uint8",
# #         },
# #     }
# #     torch.save(cache, cache_path)
# #     return cache


# # # -----------------------------
# # # Fast cached dataset
# # # -----------------------------
# # def _make_balanced_test_indices_like_original(
# #     attr: torch.Tensor,
# #     target_label_idx: int,
# #     sensitive_label_idx: int,
# #     seed: int = 0,
# # ) -> torch.Tensor:
# #     """
# #     Mimics your CelebA_test behavior: for each (target, sensitive) cell,
# #     sample the same number equal to the minimum intersection size.
# #     Assumes binary target and binary sensitive.
# #     """
# #     y = attr[:, target_label_idx]
# #     s = attr[:, sensitive_label_idx]

# #     # Build index lists like your code (keys 0 and 1)
# #     lab_idx = {0: [], 1: []}
# #     col_idx = {0: [], 1: []}

# #     y_list = y.tolist()
# #     s_list = s.tolist()

# #     for i, v in enumerate(y_list):
# #         lab_idx[int(v)].append(i)
# #     for i, v in enumerate(s_list):
# #         col_idx[int(v)].append(i)

# #     # find min intersection
# #     min_intersection = 10**18
# #     for i in [0, 1]:
# #         for j in [0, 1]:
# #             inter = list(set(lab_idx[i]) & set(col_idx[j]))
# #             min_intersection = min(min_intersection, len(inter))

# #     if min_intersection <= 0 or min_intersection == 10**18:
# #         raise RuntimeError("Could not build balanced test indices (empty intersection).")

# #     rng = random.Random(seed)

# #     test_idx: List[int] = []
# #     for i in [0, 1]:
# #         for j in [0, 1]:
# #             inter = list(set(lab_idx[i]) & set(col_idx[j]))
# #             k = min(len(inter), min_intersection)
# #             picked = rng.sample(inter, k)
# #             test_idx.extend(picked)

# #             # remove picked, like your code
# #             lab_idx[i] = list(set(lab_idx[i]) - set(picked))
# #             col_idx[j] = list(set(col_idx[j]) - set(picked))

# #     return torch.tensor(test_idx, dtype=torch.int64)


# # class CachedCelebA(Dataset):
# #     """
# #     Reads from cache tensors.
# #     Optionally preloads normalized data to GPU RAM.
# #     """
# #     def __init__(
# #         self,
# #         cache: dict,
# #         target_label_idx: int,
# #         sensitive_label_idx: int,
# #         mean: Tuple[float, float, float],
# #         std: Tuple[float, float, float],
# #         make_balanced_test: bool = False,
# #         balanced_seed: int = 0,
# #         preload_device: Optional[str] = None,   # "cuda:0" to keep whole split on GPU
# #         preload_dtype: torch.dtype = torch.float16,
# #     ):
# #         self.target_label_idx = target_label_idx
# #         self.sensitive_label_idx = sensitive_label_idx

# #         images_u8 = cache["images_u8"]  # uint8 [N,3,H,W]
# #         attr = cache["attr"]            # int64 [N,40]

# #         if make_balanced_test:
# #             idx = _make_balanced_test_indices_like_original(
# #                 attr, target_label_idx, sensitive_label_idx, seed=balanced_seed
# #             )
# #             images_u8 = images_u8[idx]
# #             attr = attr[idx]

# #         self.images_u8 = images_u8.contiguous()
# #         self.attr = attr.contiguous()

# #         self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
# #         self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

# #         self.preload_device = preload_device
# #         self.preload_X = None

# #         if preload_device is not None:
# #             # Precompute normalized float tensor once and store on GPU.
# #             X = self.images_u8.to(device=preload_device, non_blocking=True)
# #             X = X.to(torch.float32).div_(255.0)
# #             mean_t = self.mean.to(preload_device)
# #             std_t = self.std.to(preload_device)
# #             X = (X - mean_t) / std_t
# #             self.preload_X = X.to(preload_dtype)

# #             self.attr = self.attr.to(preload_device, non_blocking=True)

# #     def __len__(self) -> int:
# #         return self.attr.size(0)

# #     def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# #         if self.preload_X is not None:
# #             x = self.preload_X[index]
# #         else:
# #             x = self.images_u8[index].to(torch.float32).div(255.0)
# #             x = (x - self.mean) / self.std

# #         y = self.attr[index, self.target_label_idx]
# #         s = self.attr[index, self.sensitive_label_idx]
# #         return x, y, s


# # # -----------------------------
# # # Convenience factory
# # # -----------------------------
# # def CelebA_cached(
# #     target_label_idx: int,
# #     sensitive_label_idx: int,
# #     data_dir: str,
# #     cache_dir: str,
# #     image_size: int = 64,
# #     center_crop: int = 178,
# #     cache_batch_size: int = 256,
# #     cache_num_workers: int = 8,
# #     preload_device: Optional[str] = None,  # "cuda:0" if you want full split on GPU
# # ) -> Tuple[Dataset, Dataset, Tuple[float, float, float], Tuple[float, float, float]]:
# #     mean = (0.5063, 0.4258, 0.3832)
# #     std = (0.2676, 0.2453, 0.2410)

# #     # Build caches once (or load)
# #     train_cache = build_or_load_celeba_cache(
# #         root=data_dir,
# #         split="train",
# #         cache_dir=cache_dir,
# #         image_size=image_size,
# #         center_crop=center_crop,
# #         batch_size=cache_batch_size,
# #         num_workers=cache_num_workers,
# #     )
# #     test_cache = build_or_load_celeba_cache(
# #         root=data_dir,
# #         split="test",
# #         cache_dir=cache_dir,
# #         image_size=image_size,
# #         center_crop=center_crop,
# #         batch_size=cache_batch_size,
# #         num_workers=cache_num_workers,
# #     )

# #     train_dataset = CachedCelebA(
# #         cache=train_cache,
# #         target_label_idx=target_label_idx,
# #         sensitive_label_idx=sensitive_label_idx,
# #         mean=mean,
# #         std=std,
# #         make_balanced_test=False,
# #         preload_device=preload_device,
# #     )
# #     test_dataset = CachedCelebA(
# #         cache=test_cache,
# #         target_label_idx=target_label_idx,
# #         sensitive_label_idx=sensitive_label_idx,
# #         mean=mean,
# #         std=std,
# #         make_balanced_test=True,      # matches your CelebA_test sampling idea
# #         balanced_seed=0,
# #         preload_device=preload_device,
# #     )
# #     return train_dataset, test_dataset, mean, std


# # # -----------------------------
# # # Example main: reproduce your counts fast
# # # -----------------------------
# # if __name__ == "__main__":
# #     data_dir = "/mnt/DatasetCondensation-master/data/celeba"
# #     cache_dir = os.path.join(data_dir, "celeba_cache")

# #     sensitive_label_idx = 20  # gender in your code

# #     # Option A (recommended): keep in CPU RAM, use pin_memory + async copies in your training loop
# #     preload_device = None

# #     # Option B: keep whole split on GPU RAM (only if it fits)
# #     # preload_device = "cuda:0"

# #     # Build caches once up front
# #     _ = build_or_load_celeba_cache(data_dir, "train", cache_dir, image_size=64, center_crop=178)
# #     _ = build_or_load_celeba_cache(data_dir, "test",  cache_dir, image_size=64, center_crop=178)

# #     for target_label_idx in range(40):
# #         print(target_label_idx)

# #         train_dataset, test_dataset, mean, std = CelebA_cached(
# #             target_label_idx=target_label_idx,
# #             sensitive_label_idx=sensitive_label_idx,
# #             data_dir=data_dir,
# #             cache_dir=cache_dir,
# #             image_size=64,
# #             center_crop=178,
# #             cache_batch_size=256,
# #             cache_num_workers=8,
# #             preload_device=preload_device,
# #         )

# #         # Fast counts using cached attr (no image reads)
# #         train_attr = train_dataset.attr if hasattr(train_dataset, "attr") else None
# #         test_attr = test_dataset.attr if hasattr(test_dataset, "attr") else None

# #         train_target = train_attr[:, target_label_idx].to("cpu")
# #         train_sensitive = train_attr[:, sensitive_label_idx].to("cpu")

# #         test_target = test_attr[:, target_label_idx].to("cpu")
# #         test_sensitive = test_attr[:, sensitive_label_idx].to("cpu")

# #         def count_4cells(t, s):
# #             aa = int(((t == 0) & (s == 0)).sum().item())
# #             ab = int(((t == 0) & (s == 1)).sum().item())
# #             ba = int(((t == 1) & (s == 0)).sum().item())
# #             bb = int(((t == 1) & (s == 1)).sum().item())
# #             return (aa, ab, ba, bb)

# #         print(count_4cells(train_target, train_sensitive))
# #         print(count_4cells(test_target, test_sensitive))
# #         print("--------------------------")

# #     # Minimal training loader example (CPU RAM -> GPU batches)
# #     # If preload_device is not None, use num_workers=0 (CUDA tensors inside dataset).
# #     # device = "cuda:0" if torch.cuda.is_available() else "cpu"
# #     # target_label_idx = 33
# #     # train_dataset, _, _, _ = CelebA_cached(target_label_idx, sensitive_label_idx, data_dir, cache_dir, preload_device=None)
# #     # train_loader = DataLoader(
# #     #     train_dataset,
# #     #     batch_size=256,
# #     #     shuffle=True,
# #     #     num_workers=8,
# #     #     pin_memory=True,
# #     #     persistent_workers=True,
# #     # )
# #     # for x, y, s in train_loader:
# #     #     x = x.to(device, non_blocking=True)
# #     #     y = y.to(device, non_blocking=True)
# #     #     s = s.to(device, non_blocking=True)
# #     #     break

# import csv
# import os
# import torch
# import random
# import numpy as np
# from collections import namedtuple
# from typing import Any, Callable, List, Optional, Tuple, Union, TypeVar, Iterable
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from tqdm import tqdm  # Progress bars (pip install tqdm)

# T = TypeVar("T", str, bytes)
# CSV = namedtuple("CSV", ["header", "index", "data"])

# class CelebA_train(Dataset):
#     """
#     Optimized CelebA Train Dataset
#     - Reads all images once, transforms them, and saves to a .pt file.
#     - Subsequent runs load directly from the .pt file into RAM/VRAM.
#     """

#     def __init__(
#             self,
#             target_label_idx: int,
#             sensitive_label_idx: int,
#             root: str,
#             split: str = "train",
#             target_type: Union[List[str], str] = "attr",
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             download: bool = False,
#             device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
#     ) -> None:
#         super(CelebA_train, self).__init__()
#         self.target_label_idx = target_label_idx
#         self.sensitive_label_idx = sensitive_label_idx
#         self.root = root
#         self.split = split
#         self.transform = transform
#         self.target_transform = target_transform
#         self.device = device
        
#         # Cache file path: e.g., celeba_train_cached.pt
#         self.cache_path = os.path.join(root, f"celeba_{split}_cached.pt")

#         if isinstance(target_type, list):
#             self.target_type = target_type
#         else:
#             self.target_type = [target_type]

#         if not self.target_type and self.target_transform is not None:
#             raise RuntimeError("target_transform is specified but target_type is empty")

#         # --- 1. Try Loading from Cache ---
#         if os.path.exists(self.cache_path):
#             print(f"[{split}] Loading data from cache: {self.cache_path}...")
#             # Load entire dataset
#             try:
#                 cached_data = torch.load(self.cache_path, map_location=device)
#                 self.data = cached_data['data']
#                 self.attr = cached_data['attr']
#                 print(f"[{split}] Successfully loaded {self.data.shape[0]} images to {self.device}.")
#             except Exception as e:
#                 print(f"[{split}] Failed to load cache: {e}. Re-generating...")
#                 self._generate_cache(split)
#         else:
#             # --- 2. If no cache, Read from Disk and Save ---
#             self._generate_cache(split)

#     def _generate_cache(self, split):
#         print(f"[{split}] Cache not found. Reading images from disk...")
        
#         # Load CSV metadata
#         split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
#         split_ = split_map[split]
#         splits = self._load_csv("list_eval_partition.txt")
#         attr = self._load_csv("list_attr_celeba.txt", header=1)

#         mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
#         if mask == slice(None):
#             self.filename = splits.index
#         else:
#             self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        
#         self.attr = attr.data[mask]
#         # map from {-1, 1} to {0, 1}
#         self.attr = torch.floor_divide(self.attr + 1, 2)

#         # Read Images Loop
#         images_list = []
#         # tqdm for progress bar
#         for img_name in tqdm(self.filename, desc=f"Processing {split} images"):
#             path = os.path.join(self.root, "", img_name)
#             img = Image.open(path)
            
#             # Apply deterministic transform NOW
#             if self.transform is not None:
#                 img = self.transform(img)
            
#             images_list.append(img)
        
#         # Stack into one tensor (N, C, H, W)
#         print(f"[{split}] Stacking tensors...")
#         self.data = torch.stack(images_list)
        
#         # Move to target device
#         # If OOM occurs here, change device to 'cpu' in the CelebA() function call
#         self.data = self.data.to(self.device)
#         self.attr = self.attr.to(self.device)

#         # Save to .pt
#         print(f"[{split}] Saving cache to: {self.cache_path}...")
#         torch.save({'data': self.data.cpu(), 'attr': self.attr.cpu()}, self.cache_path)

#     def _load_csv(self, filename: str, header: Optional[int] = None) -> CSV:
#         with open(os.path.join(self.root, filename)) as csv_file:
#             data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

#         if header is not None:
#             headers = data[header]
#             data = data[header + 1:]
#         else:
#             headers = []
#         indices = [row[0] for row in data]
#         data = [row[1:] for row in data]
#         data_int = [list(map(int, i)) for i in data]

#         return CSV(headers, indices, torch.tensor(data_int))

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         # Fast retrieval from memory
#         X = self.data[index]
#         target = self.attr[index, self.target_label_idx]
#         sensitive = self.attr[index, self.sensitive_label_idx]
#         return X, target, sensitive

#     def __len__(self) -> int:
#         return len(self.attr)


# class CelebA_test(Dataset):
#     """
#     Optimized CelebA Test Dataset (with Balancing Logic)
#     """
#     def __init__(
#             self,
#             target_label_idx: int,
#             sensitive_label_idx: int,
#             root: str,
#             split: str = "test",
#             target_type: Union[List[str], str] = "attr",
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             download: bool = False,
#             device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
#     ) -> None:
#         super(CelebA_test, self).__init__()
#         self.target_label_idx = target_label_idx
#         self.sensitive_label_idx = sensitive_label_idx
#         self.root = root
#         self.split = split
#         self.transform = transform
#         self.target_transform = target_transform
#         self.device = device
        
#         # Cache file path for balanced test set
#         # Note: If you change the balancing logic, you must delete this .pt file!
#         self.cache_path = os.path.join(root, f"celeba_{split}_balanced_cached.pt")

#         if isinstance(target_type, list):
#             self.target_type = target_type
#         else:
#             self.target_type = [target_type]

#         if not self.target_type and self.target_transform is not None:
#             raise RuntimeError("target_transform is specified but target_type is empty")

#         if os.path.exists(self.cache_path):
#             print(f"[{split}] Loading data from cache: {self.cache_path}...")
#             try:
#                 cached_data = torch.load(self.cache_path, map_location=device)
#                 self.data = cached_data['data']
#                 self.attr = cached_data['attr']
#                 print(f"[{split}] Successfully loaded {self.data.shape[0]} images to {self.device}.")
#             except Exception as e:
#                 print(f"[{split}] Failed to load cache: {e}. Re-generating...")
#                 self._generate_cache(split)
#         else:
#             self._generate_cache(split)

#     def _generate_cache(self, split):
#         print(f"[{split}] Cache not found. Calculating balance and reading images...")
        
#         split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
#         split_ = split_map[split]
#         splits = self._load_csv("list_eval_partition.txt")
#         attr = self._load_csv("list_attr_celeba.txt", header=1)

#         mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
#         if mask == slice(None):
#             self.filename = splits.index
#         else:
#             self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

#         self.attr = attr.data[mask]
#         self.attr = torch.floor_divide(self.attr + 1, 2)

#         # --- Balancing Logic ---
#         lab_idx_dict = {}
#         col_idx_dict = {}
#         lab = self.attr[:, self.target_label_idx]
#         col = self.attr[:, self.sensitive_label_idx]

#         for lab_id in np.unique(lab):
#             lab_idx_dict[lab_id] = [idx for idx, c in enumerate(lab) if lab_id == c]
#         for col_id in np.unique(col):
#             col_idx_dict[col_id] = [idx for idx, c in enumerate(col) if col_id == c]

#         test_idx = []
#         min_intersection = 1e10
#         for i in range(len(lab_idx_dict)):
#             for j in range(len(col_idx_dict)):
#                 intersection = list(set(lab_idx_dict[i]) & set(col_idx_dict[j]))
#                 min_intersection = min(min_intersection, len(intersection))

#         for i in range(len(lab_idx_dict)):
#             for j in range(len(col_idx_dict)):
#                 intersection = list(set(lab_idx_dict[i]) & set(col_idx_dict[j]))
#                 select_idx = random.sample(intersection, min(len(intersection), min_intersection))
#                 test_idx.extend(select_idx)
#                 lab_idx_dict[i] = list(set(lab_idx_dict[i]) - set(select_idx))
#                 col_idx_dict[j] = list(set(col_idx_dict[j]) - set(select_idx))

#         self.attr = self.attr[test_idx]
#         temp = []
#         for i in test_idx:
#             temp.append(self.filename[i])
#         self.filename = temp

#         # --- Read Images ---
#         images_list = []
#         for img_name in tqdm(self.filename, desc=f"Processing {split} images"):
#             path = os.path.join(self.root, "", img_name)
#             img = Image.open(path)
#             if self.transform is not None:
#                 img = self.transform(img)
#             images_list.append(img)
        
#         print(f"[{split}] Stacking tensors...")
#         self.data = torch.stack(images_list)
        
#         self.data = self.data.to(self.device)
#         self.attr = self.attr.to(self.device)

#         print(f"[{split}] Saving cache to: {self.cache_path}...")
#         torch.save({'data': self.data.cpu(), 'attr': self.attr.cpu()}, self.cache_path)

#     def _load_csv(self, filename: str, header: Optional[int] = None) -> CSV:
#         with open(os.path.join(self.root, filename)) as csv_file:
#             data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
#         if header is not None:
#             headers = data[header]
#             data = data[header + 1:]
#         else:
#             headers = []
#         indices = [row[0] for row in data]
#         data = [row[1:] for row in data]
#         data_int = [list(map(int, i)) for i in data]
#         return CSV(headers, indices, torch.tensor(data_int))

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         X = self.data[index]
#         target = self.attr[index, self.target_label_idx]
#         sensitive = self.attr[index, self.sensitive_label_idx]
#         return X, target, sensitive

#     def __len__(self) -> int:
#         return len(self.attr)


# def iterable_to_str(iterable: Iterable) -> str:
#     return "'" + "', '".join([str(item) for item in iterable]) + "'"


# def CelebA(target_label_idx, sensitive_label_idx, data_dir="/mnt/DatasetCondensation-master/data/celeba"):
#     mean = (0.5063, 0.4258, 0.3832)
#     std = (0.2676, 0.2453, 0.2410)
#     from torchvision import transforms
    
#     # 1. Deterministic Transform (Resize/Normalize)
#     # Important: Random Augmentations should NOT be here if caching processed tensors.
#     transform = transforms.Compose([
#         transforms.CenterCrop(178),
#         transforms.Resize(64),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std), 
#     ])

#     # 2. Determine Device (GPU vs CPU)
#     # If you run out of GPU memory, change this to 'cpu'
#     # 'cpu' will still be much faster than reading files from disk.
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     train_dataset = CelebA_train(
#         target_label_idx, sensitive_label_idx, root=data_dir, split='train',
#         transform=transform, device=device
#     )
#     test_dataset = CelebA_test(
#         target_label_idx, sensitive_label_idx, root=data_dir, split='test', 
#         transform=transform, device=device
#     )
    
#     return train_dataset, test_dataset, mean, std


# if __name__ == "__main__":
#     # target_label_idx = 33  # Wavy Hair
#     sensitive_label_idx = 20  # Gender

#     # Using a subset loop as in your example
#     for i in range(0, 40):
#         target_label_idx = i
#         print(f"\n--- Processing Target Attribute Index: {target_label_idx} ---")
        
#         # This will create the .pt file on the first iteration, and load it instantly on subsequent ones
#         train_dataset, test_dataset, mean, std = CelebA(
#             target_label_idx, sensitive_label_idx,
#             data_dir="/mnt/DatasetCondensation-master/data/celeba"
#         )

#         train_target = train_dataset.attr[:, target_label_idx]
#         train_sensitive = train_dataset.attr[:, sensitive_label_idx]

#         test_target = test_dataset.attr[:, target_label_idx]
#         test_sensitive = test_dataset.attr[:, sensitive_label_idx]

#         # --- Your Counting Logic ---
#         # Note: Vectorized logic is faster than loops for counting
#         def count_groups(targets, sensitives):
#             aa = ((targets == 0) & (sensitives == 0)).sum().item()
#             ab = ((targets == 0) & (sensitives == 1)).sum().item()
#             ba = ((targets == 1) & (sensitives == 0)).sum().item()
#             bb = ((targets == 1) & (sensitives == 1)).sum().item()
#             return aa, ab, ba, bb

#         print("Train Counts (aa, ab, ba, bb):", count_groups(train_target, train_sensitive))
#         print("Test Counts (aa, ab, ba, bb):", count_groups(test_target, test_sensitive))
#         print("--------------------------")


import csv
import os
import torch
import random
import numpy as np
import gc  # Garbage collection
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union, TypeVar, Iterable
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import math

T = TypeVar("T", str, bytes)
CSV = namedtuple("CSV", ["header", "index", "data"])

class CelebA_train(Dataset):
    """
    Partitioned CelebA Train Dataset
    - Splits processing into 20 chunks to avoid "Killed" (OOM) errors.
    - Loads chunks into a list of tensors on GPU/RAM.
    """
    def __init__(
            self,
            target_label_idx: int,
            sensitive_label_idx: int,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            num_partitions: int = 20  # Split into 20 parts
    ) -> None:
        super(CelebA_train, self).__init__()
        self.target_label_idx = target_label_idx
        self.sensitive_label_idx = sensitive_label_idx
        self.root = root
        self.split = split
        self.transform = transform
        self.device = device
        self.num_partitions = num_partitions
        
        # Base cache name
        self.cache_base = os.path.join(root, f"celeba_{split}_chunk")

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        # Check if ALL parts exist
        all_parts_exist = all([os.path.exists(f"{self.cache_base}_{i}.pt") for i in range(num_partitions)])

        if all_parts_exist:
            print(f"[{split}] Found all {num_partitions} cache chunks. Loading into {device}...")
            self._load_cache()
        else:
            print(f"[{split}] Cache incomplete. Generating {num_partitions} chunks from disk...")
            self._generate_cache()
            print(f"[{split}] Generation complete. Loading into {device}...")
            self._load_cache()

    def _generate_cache(self):
        # 1. Load Metadata
        split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
        split_val = split_map[self.split]
        splits = self._load_csv("list_eval_partition.txt")
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_val is None else (splits.data == split_val).squeeze()
        if mask == slice(None):
            all_filenames = splits.index
        else:
            all_filenames = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        
        # Save Attributes (Small enough to save in one go)
        full_attr = attr.data[mask]
        full_attr = torch.floor_divide(full_attr + 1, 2)
        torch.save(full_attr, os.path.join(self.root, f"celeba_{self.split}_attr.pt"))

        # 2. Loop through partitions
        total_len = len(all_filenames)
        chunk_size = math.ceil(total_len / self.num_partitions)

        for i in range(self.num_partitions):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_len)
            
            # If we went past the end, stop
            if start_idx >= total_len:
                break

            chunk_filenames = all_filenames[start_idx:end_idx]
            chunk_path = f"{self.cache_base}_{i}.pt"
            
            if os.path.exists(chunk_path):
                print(f"  - Chunk {i+1}/{self.num_partitions} exists, skipping.")
                continue

            images_list = []
            desc = f"  - Chunk {i+1}/{self.num_partitions}"
            
            for img_name in tqdm(chunk_filenames, desc=desc, leave=False):
                path = os.path.join(self.root, "", img_name)
                img = Image.open(path)
                if self.transform is not None:
                    img = self.transform(img)
                images_list.append(img)
            
            # Stack and Save
            # We keep it on CPU for saving to avoid OOM during the 'stack' operation
            data_chunk = torch.stack(images_list)
            torch.save(data_chunk, chunk_path)
            
            # CRITICAL: Clean up memory immediately
            del images_list
            del data_chunk
            gc.collect()

    def _load_cache(self):
        self.data_chunks = []
        self.chunk_starts = [0]
        
        # Load Attributes
        attr_path = os.path.join(self.root, f"celeba_{self.split}_attr.pt")
        self.attr = torch.load(attr_path, map_location=self.device)

        # Load Image Chunks
        current_idx = 0
        for i in range(self.num_partitions):
            chunk_path = f"{self.cache_base}_{i}.pt"
            if not os.path.exists(chunk_path): 
                break # Should not happen if logic is correct
                
            # Load directly to target device (GPU/RAM)
            chunk = torch.load(chunk_path, map_location=self.device)
            self.data_chunks.append(chunk)
            
            current_idx += chunk.shape[0]
            self.chunk_starts.append(current_idx)
            
        # Helper for fast lookups
        self.chunk_size_approx = self.data_chunks[0].shape[0]

    def _load_csv(self, filename: str, header: Optional[int] = None) -> CSV:
        with open(os.path.join(self.root, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []
        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]
        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Logic to find which chunk the index belongs to
        # Since chunks are equal size (except last), we can guess safely
        chunk_idx = index // self.chunk_size_approx
        
        # Edge case correction for the last chunk or irregular sizes
        if chunk_idx >= len(self.data_chunks):
            chunk_idx = len(self.data_chunks) - 1
        
        # Adjust index relative to the chunk
        # We need the start index of this specific chunk
        # (Using predefined list is faster than calculating every time)
        local_idx = index - self.chunk_starts[chunk_idx]
        
        # Safety check (rare case if simple division is off by one near boundaries)
        if local_idx < 0:
            chunk_idx -= 1
            local_idx = index - self.chunk_starts[chunk_idx]
        elif local_idx >= self.data_chunks[chunk_idx].shape[0]:
            chunk_idx += 1
            local_idx = index - self.chunk_starts[chunk_idx]

        X = self.data_chunks[chunk_idx][local_idx]
        target = self.attr[index, self.target_label_idx]
        sensitive = self.attr[index, self.sensitive_label_idx]

        return X, target, sensitive

    def __len__(self) -> int:
        return len(self.attr)

# CelebA_test is smaller (20k images), so it usually doesn't need partitioning.
# However, for consistency and safety, here is the updated Test class using the standard caching
# (Single file is usually fine for test set, but we handle memory carefully).
class CelebA_test(Dataset):
    def __init__(
            self,
            target_label_idx: int,
            sensitive_label_idx: int,
            root: str,
            split: str = "test",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        super(CelebA_test, self).__init__()
        self.target_label_idx = target_label_idx
        self.sensitive_label_idx = sensitive_label_idx
        self.root = root
        self.split = split
        self.transform = transform
        self.device = device
        
        self.cache_path = os.path.join(root, f"celeba_{split}_balanced_cached.pt")

        if os.path.exists(self.cache_path):
            print(f"[{split}] Loading data from cache: {self.cache_path}...")
            cached = torch.load(self.cache_path, map_location=device)
            self.data = cached['data']
            self.attr = cached['attr']
        else:
            self._generate_cache()

    def _generate_cache(self):
        print(f"[{self.split}] Cache not found. Reading images...")
        split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
        split_val = split_map[self.split]
        splits = self._load_csv("list_eval_partition.txt")
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_val is None else (splits.data == split_val).squeeze()
        if mask == slice(None):
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

        self.attr = attr.data[mask]
        self.attr = torch.floor_divide(self.attr + 1, 2)

        # Balancing Logic
        lab_idx_dict = {}
        col_idx_dict = {}
        lab = self.attr[:, self.target_label_idx]
        col = self.attr[:, self.sensitive_label_idx]

        for lab_id in np.unique(lab):
            lab_idx_dict[lab_id] = [idx for idx, c in enumerate(lab) if lab_id == c]
        for col_id in np.unique(col):
            col_idx_dict[col_id] = [idx for idx, c in enumerate(col) if col_id == c]

        test_idx = []
        min_intersection = 1e10
        for i in range(len(lab_idx_dict)):
            for j in range(len(col_idx_dict)):
                intersection = list(set(lab_idx_dict[i]) & set(col_idx_dict[j]))
                min_intersection = min(min_intersection, len(intersection))

        for i in range(len(lab_idx_dict)):
            for j in range(len(col_idx_dict)):
                intersection = list(set(lab_idx_dict[i]) & set(col_idx_dict[j]))
                select_idx = random.sample(intersection, min(len(intersection), min_intersection))
                test_idx.extend(select_idx)
                lab_idx_dict[i] = list(set(lab_idx_dict[i]) - set(select_idx))
                col_idx_dict[j] = list(set(col_idx_dict[j]) - set(select_idx))

        self.attr = self.attr[test_idx]
        temp_files = [self.filename[i] for i in test_idx]
        self.filename = temp_files

        images_list = []
        for img_name in tqdm(self.filename, desc=f"Processing {self.split}"):
            path = os.path.join(self.root, "", img_name)
            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
            images_list.append(img)
        
        self.data = torch.stack(images_list).to(self.device)
        self.attr = self.attr.to(self.device)

        torch.save({'data': self.data.cpu(), 'attr': self.attr.cpu()}, self.cache_path)

    def _load_csv(self, filename: str, header: Optional[int] = None) -> CSV:
        with open(os.path.join(self.root, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []
        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]
        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.data[index], self.attr[index, self.target_label_idx], self.attr[index, self.sensitive_label_idx]

    def __len__(self) -> int:
        return len(self.attr)

def CelebA(target_label_idx, sensitive_label_idx, data_dir="/mnt/DatasetCondensation-master/data/celeba"):
    mean = (0.5063, 0.4258, 0.3832)
    std = (0.2676, 0.2453, 0.2410)
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std), 
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train uses Partitioned class
    train_dataset = CelebA_train(
        target_label_idx, sensitive_label_idx, root=data_dir, split='train',
        transform=transform, device=device, num_partitions=20
    )
    # Test uses standard optimized class
    test_dataset = CelebA_test(
        target_label_idx, sensitive_label_idx, root=data_dir, split='test', 
        transform=transform, device=device
    )
    
    return train_dataset, test_dataset, mean, std

if __name__ == "__main__":
    target_label_idx = 33 
    sensitive_label_idx = 20  

    for i in range(0, 40):
        target_label_idx = i
        print(f"\n--- Processing Target Index: {target_label_idx} ---")
        
        train_dataset, test_dataset, mean, std = CelebA(
            target_label_idx, sensitive_label_idx,
            data_dir="/mnt/DatasetCondensation-master/data/celeba"
        )
        
        # Simple count verification
        train_target = train_dataset.attr[:, target_label_idx]
        train_sensitive = train_dataset.attr[:, sensitive_label_idx]
        
        print(f"Train Dataset Size: {len(train_dataset)}")
        print("--------------------------")