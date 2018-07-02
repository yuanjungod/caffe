// add by Binbin Xu
// declanxu@gmail.com or declanxu@126.com
// Zhejiang University, State Key Lab of CAD&CG.

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"
namespace caffe {

/// find triplet(X_i^a, X_i^p, X_i^n) satisfies some constraint
/// X_i^p is the positive, means has the same label as X_i^a
/// X_i^n is the negative, means has the different labels as X_i^a

// in layer.hpp Reshape will be called after LayerSetUp
template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    LossLayer<Dtype>::Reshape(bottom, top);
    diff_.ReshapeLike(*bottom[0]); // bottom[0] is batch_size*channels(128)*1*1
    sub_.ReshapeLike(*bottom[0]);
    inner_num_ = bottom[0]->count(1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // top is just a scalar
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // store (X_i^n - X_i^p)/N which can backpropagate to prev-layer directly
  // diff_.ReshapeLike(*bottom[0]); // bottom[0] is batch_size*channels(128)*1*1
  // sub_.ReshapeLike(*bottom[0]);
  // inner_num_ = bottom[0]->count(1);
  // get some parameters
  label_separator_ = this->layer_param_.triplet_loss_param().separate();
  identities_per_batch_ = this->layer_param_.triplet_loss_param().ids_per_batch();
  num_per_identity_ = this->layer_param_.triplet_loss_param().num_per_id();
  alpha_ = (Dtype)(this->layer_param_.triplet_loss_param().alpha());
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    // it is the hardest part which is trying to find the triplet
    // algorithm should be fast enough and we just choose semi-hard triplets
    // refer to the paper: FaceNet for more details
    Dtype loss = Dtype(0);
    int batch_size = bottom[0]->num(); // get the batch_size
    //CHECK_EQ(batch_size, bottom[1]->count());

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* diff_mutable = diff_.mutable_cpu_data();
    Dtype* sub_mutable = sub_.mutable_cpu_data();
    Dtype* diff_diff = diff_.mutable_cpu_diff(); // store the diff
    caffe_set(diff_.count(), Dtype(0), diff_mutable);
    caffe_set(diff_.count(), Dtype(0), diff_diff);
    // #program
    vector<int> labels(batch_size, 0);
    for (int i = 0; i < batch_size; i++){
        labels[i] = static_cast<int>(bottom_label[i]);
    }
    // #pragma omp parallel for
    for (int i = 0; i < batch_size; i++){
        Dtype* diff_mutable_ptr = diff_mutable + i*inner_num_;
        int label = labels[i];
        if (label < label_separator_) {
            vector<Dtype> val(batch_size, 0);
            // int ptr = 0;
            Dtype* sub_mutable_ptr = sub_mutable;
            Dtype* diff_inner_ptr = diff_mutable;
            // #pragma omp parallel for
            // change to gemv
            for (int j = 0; j < batch_size; j++){
                caffe_sub(inner_num_, diff_mutable_ptr, diff_inner_ptr, sub_mutable_ptr);
                val[j] = caffe_cpu_dot(inner_num_, sub_mutable_ptr, sub_mutable_ptr);
                diff_inner_ptr += inner_num_;
                sub_mutable_ptr += inner_num_;
            }

            for (int j = 0; j < batch_size; j++){
                if (j != i && labels[j] == label){
                    for (int k = 0; k < batch_size; k++){
                        if (labels[k] != label) {
                            if (val[j]+alpha_ >= val[k]) {
                                loss += val[j] + alpha_ - val[k];
                                // store half of the gradients
                                caffe_sub(inner_num_, bottom_data+k*inner_num_, bottom_data+j*inner_num_, diff_diff+i*inner_num_);
                            }
                        }
                    }
                }
            }
        }
        /* has already been done in the above caffe_set()
        else{
            caffe_set(inner_num_, 0, diff_mutable[i*inner_num_]);
        }
        */


    }

    top[0]->mutable_cpu_data()[0] = loss;

  // int count = bottom[0]->count();
  // caffe_sub(
  //     count,
  //     bottom[0]->cpu_data(),
  //     bottom[1]->cpu_data(),
  //     diff_.mutable_cpu_data());
  // Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  // Dtype loss = dot / bottom[0]->num() / Dtype(2);
  // top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]){
        //Dtype scale = Dtype(2.0);
        Dtype scale = Dtype(2.0)*top[0]->cpu_diff()[0];
        caffe_cpu_scale(
                bottom[0]->count(),    // count
                scale,                 // scale
                diff_.cpu_diff(),      // input
                bottom[0]->mutable_cpu_diff() // output
        );
    }
    else {
        LOG(ERROR) << "should be back propagate to prev-layer AT TripletLossLayer::Backward_cpu" << std::endl;
    }

  // for (int i = 0; i < 2; ++i) {
  //   if (propagate_down[i]) {
  //     const Dtype sign = (i == 0) ? 1 : -1;
  //     const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
  //     caffe_cpu_axpby(
  //         bottom[i]->count(),              // count
  //         alpha,                              // alpha
  //         diff_.cpu_data(),                   // a
  //         Dtype(0),                           // beta
  //         bottom[i]->mutable_cpu_diff());  // b
  //   }
  // }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);
} // namespace caffe
