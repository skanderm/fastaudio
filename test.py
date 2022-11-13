from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastaudio.ci import skip_if_ci

def CrossValidationSplitter(col='fold', fold=1):
    "Split `items` (supposed to be a dataframe) by fold in `col`"
    def _inner(o):
        assert isinstance(o, pd.DataFrame), "ColSplitter only works when your items are a pandas DataFrame"
        col_values = o.iloc[:,col] if isinstance(col, int) else o[col]
        valid_idx = (col_values == fold).values.astype('bool')
        return IndexSplitter(mask2idxs(valid_idx))(o)
    return _inner


cfg = AudioConfig.Voice()
print(cfg)
a2s = AudioToSpec.from_cfg(cfg)
# crop = ResizeSignal(1000)


db = DataBlock(
    blocks=(AudioBlock, CategoryBlock), 
    get_items=get_audio_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    batch_tfms=[a2s],
    get_y=parent_label,
)

dls = db.dataloaders("audio_subset", verbose=True)