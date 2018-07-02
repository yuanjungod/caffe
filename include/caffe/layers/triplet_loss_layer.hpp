#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class TripletLossLayer : public LossLayer<Dtype> {
public:
    explicit TripletLossLayer(const LayerParameter& param)
     : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "TripletLoss"; }

    // the return value should change, it is wrong for now
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    // virtual inline int MinTopBlobs() const { return 1; }
    // virtual inline int MaxTopBlobs() const { return 2; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    /// The internal TripletLossLayer used to map predictions to a distribution.
    // shared_ptr<Layer<Dtype> > tripletloss_layer_;
    /// prob stores the output probability predictions from the SoftmaxLayer.
    // Blob<Dtype> prob_;
    /// bottom vector holder used in call to the underlying TripletLossLayer::Forward
    // vector<Blob<Dtype>*> tripletloss_bottom_vec_;
    /// top vector holder used in call to the underlying TripletLossLayer::Forward
    // vector<Blob<Dtype>*> tripletloss_top_vec_;
    /// Whether to ignore instances with a certain label.
    // bool has_ignore_label_;
    /// The label indicating that an instance should be ignored.
    // int ignore_label_;

    /// Whether to normalize the loss by the total number of values present
    /// (otherwise just by the batch size).
    // bool normalize_;

    // int tripletloss_axis_, outer_num_,

    Blob<Dtype> diff_;
    Blob<Dtype> sub_;
    int inner_num_;
    int label_separator_;// = this->layer_param_.triplet_loss_param().separate();
    int identities_per_batch_;// = this->layer_param_.triplet_loss_param().ids_per_batch();
    int num_per_identity_;// = this->layer_param_.triplet_loss_param().num_per_id();
    Dtype alpha_;// = (Dtype)(this->layer_param_.triplet_loss_param().alpha());
    // Dtype* sub_tmp_;
};

}  // namespace caffe

#endif  // CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
