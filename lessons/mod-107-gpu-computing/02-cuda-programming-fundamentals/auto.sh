#nvcc vector_scale.cu -o vector_scale   -gencode arch=compute_120,code=sm_120
#./vector_scale


nvcc matrix_transpose.cu -o matrix_transpose   -gencode arch=compute_120,code=sm_120
./matrix_transpose