# DNN-Training-Suite

## Introduction

This repository contains the benchmark suite for DNN training.

- [Training Instructions](#training-instructions)
- [Profiling with CUDA](#profiling-with-cuda)
  - [nvprof](#nvprof)
  - [Visual Profiler](#visual-profiler)
- [Known Issues](#known-issues)


The benchmarks were categorized according to their application and model (e.g. MachineTranslation-Seq2Seq, 
MachineTranslation-Transformer, ...). 
Each category contains a `README.md` that points to related resources (e.g. papers, tutorials, etc.).

Each benchmark contains the following:

* `source`: Source files of the DNN benchmarks, with the suffix indicating its version (e.g. git branch or tag name).
Each source folder includes `requirements.gpu-cu80.txt` that contains all the packages needed on a `CUDA 8.0` machine.
Please note that benchmark source files might be changed in order for them to run properly or make fair comparisons. All the changes will be highlighted using the following format:

```python
# <EcoSys> A brief description of what the changes are.
Changes go here ...
# </EcoSys>
```

* `dataset`: Scripts that download the dataset, and follows the naming convention `download-dataset_name.sh`.

* `scripts`: Scripts that run the benchmark on certain dataset, and follows the naming convention `benchmark-dataset.sh`.
If no arguments are given, the script will help you run the benchmark. 
If command argument `--profile` is given, the script will do profiling on CUDA kernels.
If command argument `--profile-fp32` is given, the script will do profiling on utilization on floating point units.
(Please note that you will need to input those `.nvvp` files to NVidia Visual Profiler to see the profiling results.)

* `README.md`: An introduction on what the benchmark is, 
and contains changelog that records changes that were made to the benchmark. 
If the benchmark was published as a paper, then proper bibtex will also be included.

## Training Instructions

* Setup Virtual Environment: To avoid conflicts between different benchmarks, 
we strongly recommend that you setup a Python virtual environment for each benchmark separately. 
Please note that many benchmarks have a restriction on the Python version used.

```
virtualenv --system-site-packages -p python2 <virtual-env-name> # python 2.x
virtualenv --system-site-packages -p python3 <virtual-env-name> # python 3.x

source <virtual-env-name>/bin/activate # Activate the virtual environment.
```

* Install the benchmark prerequisite: 

```
cd <benchmark-name>/source/; pip install -r requirements.gpu-cu80.txt
```

* Download the benchmark dataset:

```
cd <benchmark-name>/dataset/
chmod 700 ./download-<dataset_name>.sh
./download-<dataset_name>.sh
```

* Use the script files to run or profile the benchmark:

```
cd <benchmark-name>/scripts/
chmod 700 ./<benchmark_name>-<dataset-name>.sh
./<benchmark_name>-<dataset_name>.sh
```

## Profiling with CUDA
The [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) provides ways to profile all the CUDA-related
 activities of your application. There are two tools of interest, which provide a powerful way to understand what your 
 machine is actually doing while you train your neural network. The first, called 
 [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview), is a tool which enables you 
 to collect profiling data straight from the command-line. This data can then be viewed in the 
 [Visual Profiler](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual), which displays a timeline of 
 the recorded activity of your application.

To start profiling, ensure that you have installed 
[the latest toolkit version](https://developer.nvidia.com/cuda-downloads) (9.2 as of September 2018).

### nvprof
First, we will verify the version of nvprof we are using:
```bash
$ which nvprof
/usr/local/cuda-9.2/bin/nvprof
```
By default, your system may come with an older version pre-installed. If this is the case, then ensure you 
uninstall that version to avoid any conflicts.

Whenever you use nvprof, either from a script or directly from the command line itself, you are required to use the 
following syntax:

```bash
$ nvprof [options] <application>
```

For this example we will profile a simple linear regression task, found on the [Pytorch Examples repository](https://github.com/pytorch/examples/tree/master/regression). 
This model is a single fully-connected layer which fits a 4th degree polynomial.

First, let's clone the repository:

```bash
$ git clone https://github.com/pytorch/examples
$ cd examples/regression
$ git reset --hard 645c7c386e62d2fb1d50f4621c1a52645a13869f
```

Now, since this model isn't currently using CUDA, we will make a few changes to allow us to profile it. Open up `main.py` 
using your favourite text editor and add `from numba import cuda` as one of your import statements. From the 
[documentation](http://numba.pydata.org/numba-doc/0.13/CUDAJit.html), this allows us to "translate Python functions into PTX code 
which execute on the CUDA hardware". In other words, we want to be able to train on our GPU.

<pre>
import torch
import torch.autograd
import torch.nn.functional as F
<b>from numba import cuda</b>
</pre>

Next, we need to ensure that our objects are actually being manipulated on our GPU. This is done using the `cuda()` function call. 
First, let's convert our `W_target` and `b_target` tensors, defined a few lines below our new import statement:

<pre>
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5
<b>
W_target=W_target.cuda()
b_target=b_target.cuda()</b>
</pre>
Then, change the `get_batch()` function similarly:

<pre>
def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)<b>.cuda()</b>
    y = f(x)<b>.cuda()</b>
    return x, y
</pre>

Do the same for the neural network:
<pre>
fc = torch.nn.Linear(W_target.size(0), 1)<b>.cuda()</b>
</pre>

And finally for the output:
<pre>
output = F.smooth_l1_loss(fc(batch_x), batch_y)<b>.cuda()</b>
</pre>

For this example, we will choose to only profile part of the model, specifically the training of the polynomial. To do 
this, we need to specify the start and stop locations of the profiling. Wrap the training loop with the functions
 `cuda.profile_start()` and `cuda.profile_stop()`. For this small application there will be no noticeable difference, 
 but for larger projects you might want to avoid profiling the pre-processing of data, or other tasks unrelated to training.
 
<pre>
<b>cuda.profile_start()</b>
for batch_idx in count(1):
...
...
...
    if loss < 1e-3:
        break
<b>cuda.profile_stop()</b>
</pre>
Now that we have finished modifying the code, we are ready to start profiling. From the terminal, run the following 
command:
```bash
$ nvprof --profile-from-start off --export-profile compute_data%p.nvvp python ./main.py
```
As previously mentioned, the `--profile-from-start off` option specifies that we do not want to profile the entire application.
The `--export-profile compute_data%p.nvvp` option will output a results file, following the naming convention we specified. 

[There are many more options available](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-command-line-options-cuda),
 allowing you to choose exactly what data you want to gather. Two specific values of interest to our research group are
 **Compute Utilization** and **Single-Precision Function Unit Utilization**. To profile the former, use the command we just used. To gather data
 on the Single-Precision Function Unit Utilization metric, instead use the command:

```bash
$ nvprof --profile-from-start off --export-profile fp32_data%p.nvvp --metrics single_precision_fu_utilization python ./main.py
```

When collecting data on events or metrics (in this case the `single_precision_fu_utilization` metric), nvprof will replay each kernel as many times as is
necessary to collect all the requested data. Because of this, you will notice that the second command will take 
significantly longer to finish than the first one.

If everything goes smoothly, you should notice two new files in this directory; one from each command.


### Visual Profiler
Now that the data we need has been collected, we want to see what the results look like. First, ensure that you are using
the correct version of the Visual Profiler, and then run it:
```bash
$ which nvvp
/usr/local/cuda-9.2/bin/nvvp
$ nvvp &
```
Then, navigate to the file browser (File > Open) and select the first output file we generated. For example, it may be
named `compute_data6257.nvvp`.
 
First, click on the "Compute" tab on the left. It will highlight in green, as seen in the image below. Next, you may notice that
a significant portion of the timeline is not being used for actual computation. This overhead region, highlighed in pink below,
unfortunately is not excluded from the compute utilization calculations by the profiler. In other words, the value highlighed
in purple is significantly lower than it should be. As a workaround, to obtain an approximate value for the actual program without profiling overhead,
we must follow these steps:

1. Determine the true runtime of the program, equal to the session time (green) minus the overhead time (pink). Here, that works
out to around 1.609 - 1.3 = **0.309** seconds.

2. Multiply the raw Compute Utilization (purple) by the session time (green), then divide by 100 to acquire the true compute time. Here,
the compute time is 1.609 * 0.5 / 100 = **0.008045** seconds.

3. Divide the compute time by the true program runtime to acquire the actual compute utilization ratio. For our example,
that is 0.008045 / 0.309 = ~0.026 = **2.6%**.

![Compute Utilization](https://i.imgur.com/EvlQfBh.png)

Next, let's open up the results from the second command in the same way as before. You should see something similar to
the timeline below. To analyze the utilization for this metric, we must export the details to csv files. This is done 
using the button highlighted in red. By default, the "GPU Details" tab lists only a summary of your data. To view
a full list, press the button highlighted in pink. Again, you can export these details to another csv file. Once you
have obtained both files, you can analyze them with 
[two of our parsing utilities available from the TBD suite](https://github.com/tbd-ai/tbd-tools/tree/master/Metric_Analysis).


![Single-Precision Function Unit Utilization](https://i.imgur.com/CcQANbE.png)

Both nvprof and the visual profiler provide many other features and capabilities beyond the scope of this tutorial. Feel
free to explore them to get the most from your memory profiling.

## Known Issues

* The hyperparameters have been tuned specifically to fit into the GPU that has `8 GB` memory. If you receive an error message that informs you of Out-Of-Memory error, please carefully check `nvidia-smi` and make sure that no one else is using the machine.
