# PyTorch Memory Profiler

GPU memory is one of the primary bottlenecks for modern machine learning tasks. This problem is especially apparent for the training phase where intermediate tensors must be stored during the forward pass, before they can ultimately be used for the backward pass. This tool allows researchers to profile their PyTorch machine learning models to better understand what parts of their model are consuming the most memory. Because it is written in Python, it can be easily integrated into existing models using only a few line of code.

## Getting Started

These instructions will teach you how to use this tool and show you examples of it in action.

### Prerequisites

```
Python 3.6 or later
PyTorch (https://pytorch.org/)
CUDA (any version, as long as your training code uses it)
```

### Installation
There is nothing to install to use this memory profiler, but you must ensure that you import it correctly. After cloning this directory, add it to your path in your training code as follows:
```Python
import sys
sys.path.append('<path to pytorch_mem_profiler/ directory>')
from pytorch_mem_profiler import *
```

For example, the second line above might be:
```Python
sys.path.append('./pytorch_mem_profiler/')
```

And that's it! You are now ready to start profiling your models.

## Usage

There only three steps (and only four lines of code) needed to use the profiler:

* Create a profiler instance
* Call ```.record_stats()``` after every iteration
* Call ```.epoch_end()``` after every epoch

To illustrate how to use the profiler, we will see an example of it being used to profile a PyTorch NeuMF Recommendation model. First, we create our model as usual and put it on the GPU with ```.cuda()``` :
```Python
model = NeuMF(users, items).cuda()
```

Next, we initialize an instance of the ```memory_profiler``` class before we start the training loop. You **must** provide your model (```torch.nn.Module```) as an input. Optionally, you can specify the following as well:

>```print_period``` (postive integer) determines the number of iterations between memory reporting. Default is 1.

>```csv``` (boolean) allows profiling data to also be exported into a .csv file located in ```./memory_csv_data/``` . Default is False.

Below, we initialize the profiler, which will report memory statistics every 5 iterations to the terminal and to a .csv file. The global keyword ensures that the profiler is accessible anywhere within the main training program. Ensure that the profiler is initialized right before the training loop.

```Python
global profiler
profiler = memory_profiler(model, print_period=5, csv=True)
```

Finally, at the end of each iteration we call ```.record_stats()```, and at the end of each epoch we call ```.epoch_end()``` :

```Python
for epoch in range(num_epochs):
    for batch_index, batch_data in enumerate(loader):
        ...
        <training loop logic>
        ...
        profiler.record_stats()
    
    <end of epoch logic>
    profiler.epoch_end()
```

The results will be printed as the training progresses. Since PyTorch uses a [memory caching](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) strategy, tensors dynamically take and release from the GPU memory cache. The profiler will give you insight into the cache size, as well as a detailed layer-by-layer breakdown of what the memory is being used for:

```
*******************************************
Memory Usage for Iteration 85 of Epoch 1
*******************************************
Peak cached..........................857 MB
Current cached.......................857 MB
Total feature map usage..............301 MB

Total layer weight usage.............127 MB
  mf_user_embed.weight................35 MB
  mf_item_embed.weight.................7 MB
  mlp_user_embed.weight...............71 MB
  mlp_item_embed.weight...............14 MB
  mlp.0.weight.........................0 MB
  mlp.0.bias...........................0 MB
  mlp.1.weight.........................0 MB
  mlp.1.bias...........................0 MB
  mlp.2.weight.........................0 MB
  mlp.2.bias...........................0 MB
  final.weight.........................0 MB
  final.bias...........................0 MB

Total layer weight gradient usage....127 MB
  mf_user_embed.weight grad...........35 MB
  mf_item_embed.weight grad............7 MB
  mlp_user_embed.weight grad..........71 MB
  mlp_item_embed.weight grad..........14 MB
  mlp.0.weight grad....................0 MB
  mlp.0.bias grad......................0 MB
  mlp.1.weight grad....................0 MB
  mlp.1.bias grad......................0 MB
  mlp.2.weight grad....................0 MB
  mlp.2.bias grad......................0 MB
  final.weight grad....................0 MB
  final.bias grad......................0 MB
Intermediate gradients................34 MB
```
