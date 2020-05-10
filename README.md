# Harmonic Lowering

The official implementation of harmonic convolution by Harmonic Lowering proposed in "[Harmonic Lowering for Accelerating Harmonic Convolution for Audio Signals](https://google.com)".

<img src="https://github.com/taketakeseijin/HarmonicLowering/blob/master/figs/HarmonicLowering.png" width="480">

Note that this implementation is not the official one of the [original paper](http://dap.csail.mit.edu/).

## Benchmark

Harmonic Lowering is faster computational method of harmonic convolution.
Here are benchmarks and the tables of settings.

<img src="https://github.com/taketakeseijin/HarmonicLowering/blob/master/figs/runtime1.png" width="480">

| | n | C<sub>in</sub> | C<sub>out</sub> | S | K | P | 
|:--|:-:|:-:|:-:|:-:|:-:|:-:| 
|Setting1|1|16|32|(256,256)|(7,7)|(3,3)|
|Setting2 |1|16|32|(256,256)|(5,5)|(2,2)|
|Setting3 |1|16|32|(256,256)|(3,3)|(1,1)|
|Setting1a |7|16|32|(256,256)|(7,7)|(3,3)|
|Setting2a |5|16|32|(256,256)|(5,5)|(2,2)|
|Setting3a |3|16|32|(256,256)|(3,3)|(1,1)|

<img src="https://github.com/taketakeseijin/HarmonicLowering/blob/master/figs/runtime2.png" width="480">

| | n | C<sub>in</sub> | C<sub>out</sub> | S | K | P | 
|:--|:-:|:-:|:-:|:-:|:-:|:-:| 
|Setting4 |1|16|32|(512,512)|(3,3)|(1,1)|
|Setting5 |1|16|32|(256,256)|(3,3)|(1,1)|
|Setting6 |1|16|32|(128,128)|(3,3)|(1,1)|
|Setting7 |1|16|32|(64,64)|(3,3)|(1,1)|
|Setting8 |1|16|32|(32,32)|(3,3)|(1,1)|
|Setting9 |1|16|32|(16,16)|(3,3)|(1,1)|

These are measured in Nvidia GeForce GTX 1080Ti.
Batch Size is 16, dilation=stride=groups=1.
The parameters n, C<sub>in</sub>, C<sub>out</sub>, S, K, P in the above tables means anchor, input channel size, output channel size, input spectrogram (image) size, kernel size, padding size respectively.

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
