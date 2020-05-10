# Harmonic Lowering

The official implementation of harmonic convolution by Harmonic Lowering proposed in "[Harmonic Lowering for Accelerating Harmonic Convolution for Audio Signals](https://google.com)".

![Illustration of HL]("https://github.com/taketakeseijin/HarmonicLowering/blob/master/figs/HarmonicLowering.png" width="200")

Note that this implementation is not the official one of the [original paper](http://dap.csail.mit.edu/).

## Benchmark

Harmonic Lowering is faster computational method of harmonic convolution.
Here are benchmarks.
![Benchmark1]("https://github.com/taketakeseijin/HarmonicLowering/blob/master/figs/runtime1.png" width="200")
![Benchmark2]("https://github.com/taketakeseijin/HarmonicLowering/blob/master/figs/runtime2.png" width="200")

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
