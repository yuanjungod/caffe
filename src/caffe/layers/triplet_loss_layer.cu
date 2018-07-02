// add by Binbin Xu
// declanxu@gmail.com or declanxu@126.com
// Zhejiang University, State Key Lab of CAD&CG.

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void subAndDot(const int N, const int len, const Dtype* a, const Dtype* b, Dtype* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        // int index = i%len;
        Dtype tmp = a[i] - b[i%len];
        out[i] = tmp*tmp;
    }
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //LOG(INFO) << "enter triplet Forward_gpu";
    //LOG(INFO) << "inner_num_: " << inner_num_ << ", label_separator_: " << label_separator_;
    Dtype loss = Dtype(0);
    int batch_size = bottom[0]->num(); // get the batch_size
    //CHECK_EQ(batch_size, bottom[1]->count());
    //LOG(INFO) << batch_size << ":" << bottom[1]->num();

    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* diff_mutable = diff_.mutable_gpu_data();
    Dtype* sub_mutable = sub_.mutable_gpu_data();
    Dtype* diff_diff = diff_.mutable_gpu_diff(); // store the diff
    caffe_gpu_set(diff_.count(), Dtype(0), diff_mutable);
    caffe_gpu_set(diff_.count(), Dtype(0), diff_diff);
    vector<int> labels(batch_size, 0);
    for (int i = 0; i < batch_size; i++){
        labels[i] = static_cast<int>(bottom_label[i]);
        //std::cout << labels[i] << " ";
    }
    //std::cout << "\n";

    int count = diff_.count();
    //Dtype** mat = new Dtype*[batch_size];
    Dtype* val = new Dtype[batch_size];

    Dtype* device_scalar;
    Dtype* device_tmp;
    Dtype* middle;
    //Dtype* middle_tmp = new Dtype[inner_num_];
    CUDA_CHECK(cudaMalloc((void**)&middle, inner_num_*sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc((void**)&device_scalar, inner_num_*sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc((void**)&device_tmp, batch_size*sizeof(Dtype)));

    caffe_gpu_set(inner_num_, Dtype(1.0), device_scalar);
    int N = batch_size*inner_num_;
    for (int i = 0; i < batch_size; i++){
        int label = labels[i];
        // mat[i] = new Dtype[batch_size];
        if (label < label_separator_) {
            sub_mutable = sub_.mutable_gpu_data();
            subAndDot<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(N, inner_num_, bottom_data, bottom_data+i*inner_num_, sub_mutable);
            caffe_gpu_gemv(CblasNoTrans, inner_num_, batch_size, Dtype(1.0), sub_mutable, device_scalar, Dtype(0.0), device_tmp);     
            // cublasDgemv(handle, batch_size, inner_num_, &Dtype(1.0), sub_mutable, batch_size, device_scalar, 1,&Dtype(0), mat[i], 1);
            cudaMemcpy(val, device_tmp, batch_size*sizeof(Dtype), cudaMemcpyDeviceToHost);

            // Dtype* val = mat[i];
            // bool flag = true;
            Dtype margin = Dtype(10000.0);
            int tmp_k = -1;
            int tmp_j = -1;
            for (int j = 0; j < batch_size; j++){    
                if (j != i && labels[j] == label){ // j is the positive
                    for (int k = 0; k < batch_size; k++){
                        if (labels[k] != label) { // k is the negative
                            if (val[j] >= val[k]) {
								loss += val[j] + alpha_ - val[k];
                                caffe_gpu_sub(inner_num_, bottom_data+k*inner_num_, bottom_data+j*inner_num_, middle);
                                caffe_gpu_axpy(inner_num_, Dtype(1.0), middle, diff_diff+i*inner_num_);
                                //caffe_gpu_sub(inner_num_, bottom_data+j*inner_num_, bottom_data+i*inner_num_, middle);
                                // caffe_gpu_axpy(inner_num_, Dtype(1.0), middle, diff_diff+j*inner_num_);
                                // caffe_gpu_sub(inner_num_, bottom_data+i*inner_num_, bottom_data+k*inner_num_, middle);
                                // caffe_gpu_axpy(inner_num_, Dtype(1.0), middle, diff_diff+k*inner_num_);
                                break;
                            }
                            else {
                                if (val[k] - val[j] <= 0.2) {
                                    loss += val[j] + alpha_ - val[k];
                                    caffe_gpu_sub(inner_num_, bottom_data+k*inner_num_, bottom_data+j*inner_num_, middle);
                                    caffe_gpu_axpy(inner_num_, Dtype(1.0), middle, diff_diff+i*inner_num_);
                                    // caffe_gpu_sub(inner_num_, bottom_data+j*inner_num_, bottom_data+i*inner_num_, middle);
                                    // caffe_gpu_axpy(inner_num_, Dtype(1.0), middle, diff_diff+j*inner_num_);
                                    // caffe_gpu_sub(inner_num_, bottom_data+i*inner_num_, bottom_data+k*inner_num_, middle);
                                    // caffe_gpu_axpy(inner_num_, Dtype(1.0), middle, diff_diff+k*inner_num_);
                                    break;
                                }

                                if (val[k] - val[j] < margin) {
                                    tmp_k = k;
                                    tmp_j = j;
                                    margin = val[k] - val[j];
                                }

                            }
                        }
                    }

                    if (margin < alpha_ && tmp_k != -1) {
                        loss += val[tmp_j] + alpha_ - val[tmp_k];
                        caffe_gpu_sub(inner_num_, bottom_data+tmp_k*inner_num_, bottom_data+tmp_j*inner_num_, middle);
                        caffe_gpu_axpy(inner_num_, Dtype(1.0), middle, diff_diff+i*inner_num_);
                        // caffe_gpu_sub(inner_num_, bottom_data+tmp_j*inner_num_, bottom_data+i*inner_num_, middle);
                        // caffe_gpu_axpy(inner_num_, Dtype(1.0), middle, diff_diff+tmp_j*inner_num_);
                        // caffe_gpu_sub(inner_num_, bottom_data+i*inner_num_, bottom_data+tmp_k*inner_num_, middle);
                        // caffe_gpu_axpy(inner_num_, Dtype(1.0), middle, diff_diff+tmp_k*inner_num_);
                    }
                }
            }
        }

    }


    top[0]->mutable_cpu_data()[0] = loss/(Dtype(2)*bottom[0]->num());
    delete[] val;
    // for (int i = 0; i < batch_size; i++) {
    //     delete[] mat[i];
    // }
    // delete[] mat;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]){
        if (top[0]->cpu_diff()[0] != Dtype(1.0)){
            LOG(INFO) << "Triplet.cu top cpu_diff is not 1.0 is " << top[0]->cpu_diff()[0];
        }

        Dtype scale = Dtype(2.0)*top[0]->cpu_diff()[0]/bottom[0]->num();
        caffe_gpu_scale(
            bottom[0]->count(),    // count
            scale,                 // scale
            diff_.gpu_diff(),      // input
            bottom[0]->mutable_gpu_diff() // output
        );
	/*
	const Dtype* ptr = bottom[0]->cpu_diff();
	for (int i = 0; i < bottom[0]->num(); i++) {
	    int tmp = i*128;
	    std::cout << i << ": ";
	    for (int j = 0; j < 128; j++) {
	    	std::cout << ptr[tmp++] << " ";
	    }
	    std::cout << "\n";
	}
	
	bottom[0]->gpu_diff();
	CHECK_EQ(1,2);
	*/
    }
    else {
        LOG(ERROR) << "should be back propagate to prev-layer AT TripletLossLayer::Backward_cpu" << std::endl;
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);
} // namespace caffe
