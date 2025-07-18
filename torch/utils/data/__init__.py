# TODO(VitalyFedyunin): Rearranging this imports leads to crash,
# need to cleanup dependencies and fix it
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
    QuiverBatchSampler
)
from torch.utils.data.dataset import (
    ChainDataset,
    ConcatDataset,
    DFIterDataPipe,
    DataChunk,
    Dataset,
    IterDataPipe,
    IterableDataset,
    MapDataPipe,
    Subset,
    TensorDataset,
    random_split,
)
from torch.utils.data.dataloader import (
    DataLoader,
    _DatasetKind,
    get_worker_info,
    default_collate,
    default_convert,
)
from torch.utils.data.distributed import DistributedSampler, QuiverDistributedSampler
from torch.utils.data._decorator import (
    argument_validation,
    functional_datapipe,
    guaranteed_datapipes_determinism,
    non_deterministic,
    runtime_validation,
    runtime_validation_disabled,
)

from .shadesampler import ShadeSampler
from .shadedataset import ShadeDataset, ShadeValDataset

from torch.utils.data.dataloader_experimental import DataLoader2
from torch.utils.data import communication

__all__ = ['BatchSampler',
           'ChainDataset',
           'ConcatDataset',
           'DFIterDataPipe',
           'DataChunk',
           'DataLoader',
           'DataLoader2',
           'Dataset',
           'DistributedSampler',
           'IterDataPipe',
           'IterableDataset',
           'MapDataPipe',
           'RandomSampler',
           'Sampler',
           'SequentialSampler',
           'Subset',
           'SubsetRandomSampler',
           'TensorDataset',
           'WeightedRandomSampler',
           '_DatasetKind',
           'argument_validation',
           'communication',
           'default_collate',
           'default_convert',
           'functional_datapipe',
           'get_worker_info',
           'guaranteed_datapipes_determinism',
           'non_deterministic',
           'random_split',
           'runtime_validation',
           'runtime_validation_disabled',
           'QuiverBatchSampler',
           'QuiverDistributedSampler',
           'ShadeSampler',
           'ShadeDataset',
           'ShadeValDataset']

# Please keep this list sorted
#assert __all__ == sorted(__all__)
