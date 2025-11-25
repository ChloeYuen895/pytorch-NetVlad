import os
import numpy as np
from collections import namedtuple
from os.path import join, exists
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import h5py

# Dataset root (expects `zoo5/train/<class>/*` and `zoo5/val/<class>/*`)
root_dir = os.path.abspath('zoo5')

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def _build_struct():
    # Expects `root_dir/train/<class>/*.jpg` and `root_dir/val/<class>/*.jpg`
    if not exists(root_dir):
        raise FileNotFoundError('zoo5 dataset folder not found at: ' + root_dir)

    train_dir = join(root_dir, 'train')
    val_dir = join(root_dir, 'val')

    dbImage = []
    qImage = []
    class_names = []
    # collect images per-class. Support two layouts:
    # 1) zoo5/train/<class> and zoo5/val/<class>
    # 2) zoo5/<class> (no train/val) -- we will split 80/20 per-class
    if exists(train_dir):
        for cls in sorted(os.listdir(train_dir)):
            cls_dir = join(train_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            if cls not in class_names:
                class_names.append(cls)
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    rel = join('train', cls, f)
                    dbImage.append(rel)

    if exists(val_dir):
        for cls in sorted(os.listdir(val_dir)):
            cls_dir = join(val_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            if cls not in class_names:
                class_names.append(cls)
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    rel = join('val', cls, f)
                    qImage.append(rel)

    # If neither train/val directories exist, look for class subfolders directly under root_dir
    if (not exists(train_dir)) and (not exists(val_dir)):
        for cls in sorted(os.listdir(root_dir)):
            if cls in ('train', 'val', 'labels'):
                continue
            cls_dir = join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            if cls not in class_names:
                class_names.append(cls)
            imgs = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
            if not imgs:
                continue
            # split 80% db / 20% queries
            split = max(1, int(len(imgs) * 0.2))
            q_imgs = imgs[:split]
            db_imgs = imgs[split:]
            for f in db_imgs:
                dbImage.append(join(cls, f))
            for f in q_imgs:
                qImage.append(join(cls, f))

    # If we still didn't find any images, try a recursive scan of subfolders
    if len(dbImage) == 0 and len(qImage) == 0:
        files_by_class = {}
        for root, dirs, files in os.walk(root_dir):
            rel_root = os.path.relpath(root, root_dir)
            if rel_root in ('.', 'train', 'val', 'labels'):
                # skip top-level control folders here
                continue
            imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
            if not imgs:
                continue
            cls = os.path.basename(rel_root)
            files_by_class.setdefault(cls, [])
            for f in imgs:
                relpath = os.path.join(rel_root, f).replace('\\', '/')
                files_by_class[cls].append(relpath)

        # If we found classes via recursive scan, split per-class
        if files_by_class:
            for cls, imgs in files_by_class.items():
                imgs = sorted(imgs)
                split = max(1, int(len(imgs) * 0.2))
                q_imgs = imgs[:split]
                db_imgs = imgs[split:]
                for f in db_imgs:
                    dbImage.append(f)
                for f in q_imgs:
                    qImage.append(f)

    # Build utm coordinates: assign same coordinate for images of same class so they are positives
    # utmDb is an array shape (numDb, 2)
    def _class_index_from_path(p):
        # expected formats: 'train/class/file', 'val/class/file', 'class/file', or deeper 'sub/.../class/file'
        parts = p.replace('\\', '/').split('/')
        if len(parts) >= 2:
            if parts[0] in ('train', 'val'):
                return parts[1]
            # otherwise take the parent directory name as class
            return parts[-2]
        return '0'

    all_classes = sorted(list({_class for p in (dbImage+qImage) for _class in ([_class_index_from_path(p)])}))
    class_to_idx = {c:i for i,c in enumerate(all_classes)}

    utmDb = np.zeros((len(dbImage), 2), dtype=np.float32)
    for i,p in enumerate(dbImage):
        cls = _class_index_from_path(p)
        utmDb[i, 0] = class_to_idx.get(cls, 0)
        utmDb[i, 1] = 0.0

    utmQ = np.zeros((len(qImage), 2), dtype=np.float32)
    for i,p in enumerate(qImage):
        cls = _class_index_from_path(p)
        utmQ[i, 0] = class_to_idx.get(cls, 0)
        utmQ[i, 1] = 0.0

    # thresholds: make posDistThr small so only same-class coords are within radius
    posDistThr = 0.1
    posDistSqThr = posDistThr**2
    nonTrivPosDistSqThr = 1.0

    whichSet = 'zoo5'
    dataset = 'zoo5'

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, len(dbImage), len(qImage), posDistThr, posDistSqThr, nonTrivPosDistSqThr)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, struct, input_transform=None, onlyDB=False):
        super().__init__()
        self.input_transform = input_transform
        self.dbStruct = struct
        self.images = [join(root_dir, p) for p in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(root_dir, p) for p in self.dbStruct.qImage]
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)
        return self.positives

def collate_fn(batch):
    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None
    query, positive, negatives, indices = zip(*batch)
    import torch
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))
    return query, positive, negatives, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, struct, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()
        self.input_transform = input_transform
        self.margin = margin
        self.dbStruct = struct
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample
        self.nNeg = nNeg

        # Build a mapping from relative paths in dbStruct to actual filesystem paths.
        # If a file isn't found at the expected location, try to locate it by basename.
        self.full_paths = {}
        all_rel = list(self.dbStruct.dbImage) + list(self.dbStruct.qImage)
        for rel in all_rel:
            candidate = join(root_dir, rel)
            if os.path.exists(candidate):
                self.full_paths[rel] = candidate
                continue
            # try normalizing separators
            candidate2 = join(root_dir, rel.replace('/', os.sep).replace('\\', os.sep))
            if os.path.exists(candidate2):
                self.full_paths[rel] = candidate2
                continue
            # fallback: search by basename under root_dir (first match)
            base = os.path.basename(rel)
            found = None
            for r, d, files in os.walk(root_dir):
                if base in files:
                    found = os.path.join(r, base)
                    break
            if found:
                self.full_paths[rel] = found
            else:
                # leave missing; will handle at access time
                self.full_paths[rel] = None

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
                return_distance=False))
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True))

        self.cache = None
        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index]
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")
            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index+qOffset]
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[list(map(int, negSample))]
            knn.fit(negFeat)

            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), self.nNeg*10)
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            violatingNeg = dNeg < dPos + self.margin**0.5
            if np.sum(violatingNeg) < 1:
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        # resolve filesystem paths for query and positive
        q_rel = self.dbStruct.qImage[index]
        p_rel = self.dbStruct.dbImage[posIndex]
        q_path = self.full_paths.get(q_rel)
        p_path = self.full_paths.get(p_rel)
        if not q_path or not os.path.exists(q_path):
            print('Warning: missing query image, skipping sample:', q_rel)
            return None
        if not p_path or not os.path.exists(p_path):
            print('Warning: missing positive image, skipping sample:', p_rel)
            return None
        query = Image.open(q_path).convert('RGB')
        positive = Image.open(p_path).convert('RGB')

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            neg_rel = self.dbStruct.dbImage[negIndex]
            neg_path = self.full_paths.get(neg_rel)
            if not neg_path or not os.path.exists(neg_path):
                # skip missing negative
                continue
            negative = Image.open(neg_path).convert('RGB')
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)
        if len(negatives) == 0:
            # no valid negatives found for this query, skip sample
            return None
        negatives = __import__('torch').stack(negatives, 0)
        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)

def get_whole_training_set(onlyDB=False):
    struct = _build_struct()
    return WholeDatasetFromStruct(struct, input_transform=input_transform(), onlyDB=onlyDB)

def get_whole_val_set():
    struct = _build_struct()
    return WholeDatasetFromStruct(struct, input_transform=input_transform())

def get_whole_test_set():
    struct = _build_struct()
    return WholeDatasetFromStruct(struct, input_transform=input_transform())

def get_training_query_set(margin=0.1):
    struct = _build_struct()
    return QueryDatasetFromStruct(struct, input_transform=input_transform(), margin=margin)
