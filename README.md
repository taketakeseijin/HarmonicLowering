# Harmonic Lowering

The official implementation of harmonic convolution by Harmonic Lowering proposed in "[Harmonic Lowering for Accelerating Harmonic Convolution for Audio Signals](https://google.com)".

<img src="https://github.com/taketakeseijin/HarmonicLowering/blob/master/figs/HarmonicLowering.png" width="480">

Note that this implementation is not the official one of the [original paper](http://dap.csail.mit.edu/).

## Benchmark

Harmonic Lowering is faster computational method of harmonic convolution.
Here are benchmarks.

<img src="https://github.com/taketakeseijin/HarmonicLowering/blob/master/figs/runtime1.png" width="480">

<img src="https://github.com/taketakeseijin/HarmonicLowering/blob/master/figs/runtime2.png" width="480">

These are measured in Nvidia GeForce GTX 1080Ti.


## Reference

If you use the code, please cite:

```bibtex
    @InProceedings{Hirotoshi_2020_Interspeech,
        author = {Hirotoshi, Takeuchi and Kunio, Kashio and Yasunori,Ohishi and Hiroshi Saruwatari},
        title = {Harmonic Lowering for Accelerating Harmonic Convolution for Audio Signals},
        booktitle = {Interspeech},
        month = {},
        year = {2020}
    }
```
