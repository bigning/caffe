#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sdd_detection_loss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::LayerSetUp(bottom, top);

    // Set Up conf_loss_layer_
    LayerParameter conf_loss_param(this->layer_param_);
    conf_loss_param.set_type("SoftmaxWothLoss");
    conf_loss_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(conf_loss_param);

    std::vector<int> conf_pred_shape;
    conf_pred_shape.push_back(1);
    conf_pred_shape.push_back(this->layer_param_.detection_param().label_num());
    conf_pred_.Reshape(conf_pred_shape);
    
    std::vector<int> conf_gt_shape;
    conf_gt_shape.push_back(1);
    conf_gt_.Reshape(conf_gt_shape);

    std::vector<int> conf_loss_top_shape;
    conf_loss_top_shape.push_back(1);
    conf_loss_top_.Reshape(conf_loss_top_shape);

    conf_bottom_vec_.push_back(&conf_pred_);
    conf_bottom_vec_.push_back(&conf_gt_);
    conf_top_vec_.push_back(&conf_loss_top_);

    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);

    // Set Up loc_loss_layer_
    LayerParameter loc_loss_param(this->layer_param_);
    loc_loss_param.set_type("SmoothL1Loss");
    loc_loss_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(loc_loss_param);

    std::vector<int> loc_pred_shape;
    loc_pred_shape.push_back(1);
    loc_pred_shape.push_back(4);
    loc_pred_.Reshape(loc_pred_shape);
    loc_gt_.Reshape(loc_pred_shape);

    std::vector<int> loc_loss_top_shape;
    loc_loss_top_shape.push_back(1);
    loc_loss_top_.Reshape(loc_loss_top_shape);

    loc_bottom_vec_.push_back(&loc_pred_);
    loc_bottom_vec_.push_back(&loc_gt_);

    loc_top_vec_.push_back(loc_loss_top_);

    /*
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  */
}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
Dtype SddDetectionLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    //NOTE! NOTE!
    //NOTE! NOTE! NOTE!
    //NOTE! NOTE!
    //input of loc_layer_loss_ must have the size of (1, n*4), n is the number of matched window


  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_CLASS(SddDetectionLossLayer);
REGISTER_LAYER_CLASS(SddDetectionLoss);

}  // namespace caffe
