# Model architecture and execution

The checkpoint-compatible model accepts `N×1×H×W` normalized float32 tensors
and produces exact `N×1×2H×2W` restoration tensors. A width-48 convolutional
stem feeds three NAF encoder stages with `[2,2,4]` blocks, six middle blocks,
and three decoder stages with `[2,2,2]` blocks. The PixelShuffle head produces
2× output and adds a bicubic residual.

Mean, population standard deviation, minimum, and maximum are computed per
input image and passed through the frozen statistics conditioner. Direct
inference computes these internally. Tiled inference preprocesses the complete
image once, computes the same global statistics once, and supplies them to
every overlapping tile through a parameter-free override. Tiles are blended as
raw floats on CPU before one global clipping and PNG encoding step.

`SemiRestorePipeline` owns one persistent `ModelManager` and one restoration
service. Preprocessing and input diagnostics occur once per request. The
service retains its execution lock, so concurrent callers share one model
allocation and do not overlap model execution.
