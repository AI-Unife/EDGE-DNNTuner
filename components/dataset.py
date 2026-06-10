from __future__ import annotations

import os
import numpy as np

def get_balanced_subset(x, y, n_per_class=500):
    """
    Obtain a balanced subset of the dataset.

    """
    x_list = []
    y_list = []

    classes = np.unique(y)

    for c in classes:
        mask = (y == c)
        mask = mask.reshape(-1)

        x_c = x[mask]

        y_c = y[mask]

        x_c = x_c[:n_per_class]
        y_c = y_c[:n_per_class]

        x_list.append(x_c)
        y_list.append(y_c)

    new_x = np.concatenate(x_list, axis=0)
    new_y = np.concatenate(y_list, axis=0)

    perm = np.random.permutation(len(new_x))
    return new_x[perm], new_y[perm]

class TunerDataset:

    X_train: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_test: np.ndarray
    n_classes: int
    
    def data_as_float32(self):
        self.X_train = self.X_train.astype(np.float32)
        self.X_test  = self.X_test.astype(np.float32)

    def normalize_data(self):
        self.X_train = self.X_train.astype(np.float32) / 255.0
        self.X_test  = self.X_test.astype(np.float32) / 255.0

    def load_light_cifar(self):
        self.n_classes = 10
        self.load_hf_dataset("cifar10", image_key="img", label_key="label")

        self.X_train, self.Y_train = get_balanced_subset(self.X_train, self.Y_train, n_per_class=100)
        self.X_test, self.Y_test = get_balanced_subset(self.X_test, self.Y_test, n_per_class=100)


    def load_custom_dataset(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

    def _load_hf_dataset_offline(self, name: str):
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        from datasets import config as datasets_config
        from datasets import load_dataset

        datasets_config.HF_DATASETS_OFFLINE = True

        try:
            from huggingface_hub import constants as hub_constants

            hub_constants.HF_HUB_OFFLINE = True
        except Exception:
            pass

        cache_dir = os.getenv("HF_DATASETS_CACHE")
        # NOTE: local_files_only must NOT be passed as a kwarg: with some versions of
        # datasets it gets appended to the cache config name (e.g. 'default-local_files_only=True')
        # causing a cache miss. Offline mode is already enforced via HF_DATASETS_OFFLINE=1.
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        try:
            return load_dataset(name, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face dataset '{name}' not found in {cache_dir}. "
                "Make sure it is available in the local cache (HF_DATASETS_CACHE) "
                "or pre-downloaded on a node with Internet access."
            ) from exc

    def load_hf_dataset(self, name: str, image_key: str, label_key: str):
        dataset = self._load_hf_dataset_offline(name)
        train = dataset["train"].with_format("numpy")
        test = dataset["test"].with_format("numpy")

        self.X_train = np.asarray(train[image_key])
        self.Y_train = np.asarray(train[label_key])
        self.X_test = np.asarray(test[image_key])
        self.Y_test = np.asarray(test[label_key])

        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

    def load_cifar_10(self):
        self.n_classes = 10
        # from tensorflow.keras.datasets import cifar10
        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # self.X_train = x_train
        # self.Y_train = y_train.reshape(-1)
        # self.X_test = x_test
        # self.Y_test = y_test.reshape(-1)
        
        # print(self.X_train.shape[0], 'train samples')
        # print(self.X_test.shape[0], 'test samples')
        self.load_hf_dataset("cifar10", image_key="img", label_key="label")
        self.normalize_data()

    def load_cifar_100(self):
        self.n_classes = 100
        # from tensorflow.keras.datasets import cifar100
        # (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        # self.X_train = x_train
        # self.Y_train = y_train.reshape(-1)
        # self.X_test = x_test
        # self.Y_test = y_test.reshape(-1)
        
        # print(self.X_train.shape[0], 'train samples')
        # print(self.X_test.shape[0], 'test samples')

        self.load_hf_dataset("cifar100", image_key="img", label_key="fine_label")
        self.normalize_data()

    def load_mnist(self):
        self.n_classes = 10
        self.load_hf_dataset(
            "mnist",
            image_key="image",
            label_key="label",
        )
        # self.normalize_data()
        
    def load_tiny_imagenet(self):
        from PIL import Image
        from io import BytesIO

        dataset = self._load_hf_dataset_offline("zh-plus/tiny-imagenet")
        train = dataset["train"]
        test_split = "valid" if "valid" in dataset else "test"
        test = dataset[test_split]

        # Helper to extract the image regardless of the column format
        def open_image_from_row(img_field):
            # img_field can be bytes/bytearray or a dict with key "bytes"
            if isinstance(img_field, (bytes, bytearray)):
                data = img_field
            elif isinstance(img_field, dict) and "bytes" in img_field:
                data = img_field["bytes"]
            else:
                # In some datasets the image is already a PIL object (rare with parquet)
                # or a path. Handle these cases as well.
                if isinstance(img_field, Image.Image):
                    return img_field.convert("RGB")
                raise TypeError(f"Unrecognized image format: {type(img_field)}")
            img = Image.open(BytesIO(data)).convert("RGB")
            # Tiny-ImageNet is 64x64; ensure it in case resizing is needed
            if img.size != (64, 64):
                img = img.resize((64, 64))
            return img

        # Load into lists (faster than np.concatenate in a loop)
        x_train_list, y_train_list = [], []
        for img_field, label in zip(train["image"], train["label"]):
            img = open_image_from_row(img_field)
            x_train_list.append(np.array(img))  # (64, 64, 3), dtype uint8
            y_train_list.append(int(label))

        x_test_list, y_test_list = [], []
        for img_field, label in zip(test["image"], test["label"]):
            img = open_image_from_row(img_field)
            x_test_list.append(np.array(img))
            y_test_list.append(int(label))

        # Convert to array; optional: normalization in [0,1]
        x_train = np.stack(x_train_list)
        x_test = np.stack(x_test_list)
        y_train = np.array(y_train_list, dtype=np.int64)
        y_test = np.array(y_test_list, dtype=np.int64)
        
        self.n_classes = len(np.unique(y_train))
        self.X_train = x_train
        self.X_test = x_test
        self.Y_train = y_train
        self.Y_test = y_test
        self.normalize_data()

    def load_gesture(self):
        """Load DVSGesture dataset using the specialized gesture_dataset module."""
        from components.gesture_dataset import gesture_data
        
        self.n_classes = 11
        self.X_train, self.Y_train, self.X_test, self.Y_test = gesture_data(num_classes=11, ROI=False)
        
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')
    
    def load_roi_gesture(self):
        """Load DVSGesture dataset using the specialized gesture_dataset module, with ROI."""
        from components.gesture_dataset import gesture_data
        
        self.n_classes = 11
        X_train_raw, self.Y_train, X_test_raw, self.Y_test = gesture_data(num_classes=11, ROI=True)
        if isinstance(X_test_raw[0], dict):
            self.X_train = np.array([item["data"] for item in X_train_raw]).astype("float32")
            self.pos_train = np.array([item["pos"] for item in X_train_raw])
            self.X_test = np.array([item["data"] for item in X_test_raw]).astype("float32")
            self.pos_test = np.array([item["pos"] for item in X_test_raw])
            print(self.pos_train.shape, 'pos train samples')
            print(self.pos_test.shape, 'pos test pos samples')
        else: 
            self.X_train = np.array(X_train_raw).astype("float32")
            self.X_test = np.array(X_test_raw).astype("float32")
        
        print(self.X_train.shape, 'X train samples')
        print(self.Y_train.shape, 'Y train label samples')
        print(self.X_test.shape, 'X test samples')
        print(self.Y_test.shape, 'Y test label samples')

    def _load_cimads_dir(self, root_dir: str, resize=None, label_to_int=None):
        """
        Load images from a CIMADS-structured directory.
        Layout: root_dir/{denom}_*/{orient}/{serial}_{Front|Rear}.png
        The label is "{denom}_{orient}" (e.g. "1_A", "100_D").
        """
        from PIL import Image

        images = []
        labels_str = []

        for denom_dir in sorted(os.listdir(root_dir)):
            denom_path = os.path.join(root_dir, denom_dir)
            if not os.path.isdir(denom_path):
                continue
            denomination = denom_dir.split("_")[0]  # "1_CCA_A" -> "1"

            for orient in sorted(os.listdir(denom_path)):
                orient_path = os.path.join(denom_path, orient)
                if not os.path.isdir(orient_path):
                    continue
                label = f"{denomination}_{orient}"  # e.g. "1_A", "100_D"

                for fname in sorted(os.listdir(orient_path)):
                    if not fname.lower().endswith(".png"):
                        continue
                    img = Image.open(os.path.join(orient_path, fname)).convert("RGB")
                    if resize is not None:
                        img = img.resize(resize)
                    images.append(np.array(img))
                    labels_str.append(label)

        if label_to_int is None:
            unique_labels = sorted(
                set(labels_str),
                key=lambda s: (int(s.split("_")[0]), s.split("_")[1])
            )
            label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}

        X = np.stack(images)
        Y = np.array([label_to_int[l] for l in labels_str], dtype=np.int64)
        return X, Y, label_to_int

    def load_cca(self, cca_dir: str = "/hpc/home/bzzlca/CIMADS/DATABASE_CCA",
                 test_split: float = 0.2, resize=None):
        """Load DATABASE_CCA banknote dataset with a stratified train/test split."""
        X, Y, label_to_int = self._load_cimads_dir(cca_dir, resize=resize)
        self.n_classes = len(label_to_int)

        rng = np.random.default_rng(42)
        train_idx, test_idx = [], []
        for cls in np.unique(Y):
            idx = rng.permutation(np.where(Y == cls)[0])
            n_test = max(1, int(len(idx) * test_split))
            test_idx.extend(idx[:n_test].tolist())
            train_idx.extend(idx[n_test:].tolist())

        self.X_train = X[train_idx]
        self.Y_train = Y[train_idx]
        self.X_test  = X[test_idx]
        self.Y_test  = Y[test_idx]

        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

    def load_cim(self, cim_dir: str = "/hpc/home/bzzlca/CIMADS/DATABASE_CIM",
                 resize=None):
        """Load DATABASE_CIM banknote dataset (DataSet=train, ValidationSet=test)."""
        train_dir = os.path.join(cim_dir, "DataSet")
        test_dir  = os.path.join(cim_dir, "ValidationSet")

        self.X_train, self.Y_train, label_to_int = self._load_cimads_dir(train_dir, resize=resize)
        self.X_test,  self.Y_test,  _            = self._load_cimads_dir(test_dir,  resize=resize,
                                                                          label_to_int=label_to_int)
        self.n_classes = len(label_to_int)

        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')