# file description
harmonic_conv.py : main file. import this file.  
csrc : cpp and cuda source file for interpolation  
interp_function.py : python interface of csrc interpolation  
Lowering : Lowering input  
setup.py : build csrc files.
# dependence between files
harmonic_conv  
↓  
Lowering  
↓  
interp_function  
↓  
interp_same (build from csrc)  
↓  
interp_same_size.cpp  
↓  
interp_same_size.cu