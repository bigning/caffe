#include <algorithm>
#include <cfloat>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "caffe/layers/sdd_detection_loss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::LayerSetUp(bottom, top);

    // Set Up conf_loss_layer_
    LayerParameter conf_loss_param(this->layer_param_);
    conf_loss_param.set_type("SoftmaxWithLoss");
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

    loc_top_vec_.push_back(&loc_loss_top_);

    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);


    detect_param_ = this->layer_param_.detection_param();
    generate_default_windows();

    //set up softmax_layer_
    LayerParameter softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);

    std::vector<int> softmax_shape;
    softmax_shape.push_back(default_windows_.size());
    softmax_shape.push_back(detect_param_.label_num());
    original_score_.Reshape(softmax_shape);
    prob_.Reshape(softmax_shape);
    softmax_bottom_vec_.push_back(&original_score_);
    softmax_top_vec_.push_back(&prob_);
    softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
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

    forward_init();

    get_match_and_negatives(bottom);
    
    if (detect_param_.check_match_res()) {
        check_match_result(bottom);
        return;
    }

    prepare_for_conf_loss(bottom);
    prepare_for_loc_loss(bottom);


    conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
    conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);

    loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
    loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);

    float normalizer = match_num_;
    top[0]->mutable_cpu_data()[0] = 
        (conf_loss_top_.mutable_cpu_data()[0] + 
         detect_param_.loc_loss_weight() * loc_loss_top_.mutable_cpu_data()[0]) / normalizer;

}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::forward_init() {
    gt2default_windows_.clear();
    gt_data_.clear();
    match_num_= 0;
    is_matched_.clear();
    neg_windows_.clear();
    img_ind_for_label_data_.clear();
}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::prepare_for_conf_loss(
        const std::vector<Blob<Dtype>*>& bottom) {
    Blob<Dtype>* p_label_blob = bottom[bottom.size() - 1];

    int total = match_num_ + neg_windows_.size();
    std::vector<int> pred_conf_shape;
    pred_conf_shape.push_back(total);
    pred_conf_shape.push_back(detect_param_.label_num());
    conf_pred_.Reshape(pred_conf_shape);
    Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();

    std::vector<int> gt_conf_shape;
    gt_conf_shape.push_back(total);
    conf_gt_.Reshape(gt_conf_shape);
    Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();

    int match_ind = 0;
    int data_ind = 0;

    // for matched windows
    std::map<int, std::vector<int> >::iterator iter = gt2default_windows_.begin();
    for (; iter != gt2default_windows_.end(); iter++) {
        int gt_ind = iter->first;
        std::vector<int>& matched_wins = iter->second;
        int label = p_label_blob->data_at(0, 0, gt_ind, 1);
        int img_ind = p_label_blob->data_at(0, 0, gt_ind, 0);
        for (int i = 0; i < matched_wins.size(); i++) {
            int from_ind = default_windows_[matched_wins[i]].from_index;
            int ratio_scale_ind = 
                default_windows_[matched_wins[i]].ratio_scale_index;
            int row = default_windows_[matched_wins[i]].row;
            int col = default_windows_[matched_wins[i]].col;

            for (int label_ind = 0; label_ind < detect_param_.label_num();
                    label_ind++) {
                int channel_ind = ratio_scale_ind * (detect_param_.label_num() + 4)
                    + label_ind;
                conf_pred_data[data_ind++] = bottom[from_ind]->data_at(img_ind, channel_ind, row, col);
            } 
            conf_gt_data[match_ind] = label;
            match_ind++;
        }
    }
    CHECK_EQ(match_ind, match_num_);


    // for negative windows
    for (int i = 0; i < neg_windows_.size(); i++) {
        NegWin neg_win = neg_windows_[i];
        int from_ind = default_windows_[neg_win.window_index].from_index;
        int ratio_scale_ind = 
            default_windows_[neg_win.window_index].ratio_scale_index;
        int row = default_windows_[neg_win.window_index].row;
        int col = default_windows_[neg_win.window_index].col;
        for (int label_ind = 0; label_ind < detect_param_.label_num();
                label_ind++) {
            int channel_ind = ratio_scale_ind * (detect_param_.label_num() + 4)
                + label_ind;
            conf_pred_data[data_ind++] = bottom[from_ind]->
                data_at(neg_win.img_index, channel_ind, row, col);
        }
        conf_gt_data[match_ind++] = 0; // background label is 0
    }
    CHECK_EQ(match_ind - match_num_, neg_windows_.size());
}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::prepare_for_loc_loss(
        const std::vector<Blob<Dtype>*>& bottom) {
    std::vector<int> loc_pred_shape;
    loc_pred_shape.push_back(1);
    loc_pred_shape.push_back(4 * match_num_);
    loc_pred_.Reshape(loc_pred_shape);
    loc_gt_.Reshape(loc_pred_shape);
    Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
    Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();

    Blob<Dtype>* p_label_blob = bottom[bottom.size() - 1];
    
    int pred_data_ind = 0;
    int gt_data_ind = 0;
    std::map<int, std::vector<int> >::iterator iter = gt2default_windows_.begin();
    for (; iter != gt2default_windows_.end(); iter++) {
        int gt_ind = iter->first;
        std::vector<int> matched_wins = iter->second;
        float gt_xmin = p_label_blob->data_at(0, 0, gt_ind, 2);
        float gt_ymin = p_label_blob->data_at(0, 0, gt_ind, 3);
        float gt_xmax = p_label_blob->data_at(0, 0, gt_ind, 4);
        float gt_ymax = p_label_blob->data_at(0, 0, gt_ind, 5);
        int img_ind = p_label_blob->data_at(0, 0, gt_ind, 0);

        for (int i = 0; i < matched_wins.size(); i++) {
            DefaultWindowIndexStruct& win = default_windows_[matched_wins[i]];
            int from_ind = win.from_index;
            int ratio_scale_ind = win.ratio_scale_index;
            int row = win.row;
            int col = win.col;
            int channel_ind = ratio_scale_ind*(detect_param_.label_num() + 4)
                + detect_param_.label_num();

            // actually here, xmin is the center_x, ymin is the center_y, xmax and ymax are width and height
            float pred_xmin = bottom[from_ind]->data_at(img_ind, 
                    channel_ind + 0, row, col);
            float pred_ymin = bottom[from_ind]->data_at(img_ind, 
                    channel_ind + 1, row, col);
            float pred_xmax = bottom[from_ind]->data_at(img_ind, 
                    channel_ind + 2, row, col);
            float pred_ymax = bottom[from_ind]->data_at(img_ind, 
                    channel_ind + 3, row, col);

            /*
            float small_num = 0.00001;
            if (pred_xmax < small_num) {
                pred_xmax = small_num;
            }
            if (pred_ymax < small_num) {
                pred_ymax = small_num;
            }
            */

            // reference: faster-rcnn
            loc_pred_data[pred_data_ind++] = 
                (pred_xmin - win.center_col)/win.width;
            loc_pred_data[pred_data_ind++] = 
                (pred_ymin - win.center_row)/win.height;
            loc_pred_data[pred_data_ind++] = 
                pred_xmax / win.width;
                //log(pred_xmax/win.width);
            loc_pred_data[pred_data_ind++] = 
                pred_ymax / win.height;
                //log(pred_ymax/win.height);

            //LOG(INFO) <<"pred_x log: "  << log(pred_xmax / win.width);
            //LOG(INFO) <<"pred_y log: "  << log(pred_ymax / win.height);

            loc_gt_data[gt_data_ind++] = 
                (gt_xmin - win.center_col)/win.width;
            loc_gt_data[gt_data_ind++] = 
                (gt_ymin - win.center_row)/win.height;
            loc_gt_data[gt_data_ind++] = 
                gt_xmax/win.width;
                //log(gt_xmax/win.width);
            loc_gt_data[gt_data_ind++] = 
                gt_ymax/win.height;
                //log(gt_ymax/win.height);

            //LOG(INFO) <<"gtx log: "  << log(gt_xmax / win.width);
            //LOG(INFO) <<"gt_y log: "  << log(gt_ymax/ win.height);

            /*
            loc_pred_data[pred_data_ind++] = pred_xmin;
            loc_pred_data[pred_data_ind++] = pred_ymin;
            loc_pred_data[pred_data_ind++] = pred_xmax;
            loc_pred_data[pred_data_ind++] = pred_ymax;

            loc_gt_data[gt_data_ind++] = gt_xmin;
            loc_gt_data[gt_data_ind++] = gt_ymin;
            loc_gt_data[gt_data_ind++] = gt_xmax;
            loc_gt_data[gt_data_ind++] = gt_ymax;
            */
        }
    }
    CHECK_EQ(pred_data_ind/4, match_num_);
}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::initialize_bottom_diff(
        const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < bottom.size() - 1; i++) {
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
        caffe_set(bottom[i]->count(), Dtype(0), bottom_diff);
    }
}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    initialize_bottom_diff(bottom);

    //back prop loc_loss
    if (match_num_ > 0) {
        std::vector<bool> loc_propagate_down;
        loc_propagate_down.push_back(true);
        loc_propagate_down.push_back(false);

        loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down, 
                loc_bottom_vec_);
        Dtype normalizer = match_num_;
        Dtype loss_weight = detect_param_.loc_loss_weight() * top[0]->cpu_diff()[0] / normalizer;
        caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());

        const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
        
        std::map<int, std::vector<int> >::iterator iter = gt2default_windows_.begin();
        int ind = 0;
        for (; iter != gt2default_windows_.end(); iter++) {
            int gt_ind = iter->first;
            int img_idx = img_ind_for_label_data_[gt_ind];
            std::vector<int>& match_wins = iter->second;
            for (int i = 0; i < match_wins.size(); i++) {
                int from_ind = default_windows_[match_wins[i]].from_index;
                int ratio_scale_ind = default_windows_[match_wins[i]].ratio_scale_index;
                int row = default_windows_[match_wins[i]].row;
                int col = default_windows_[match_wins[i]].col;

                // center_x
                int channel_idx = ratio_scale_ind *
                    (detect_param_.label_num() + 4) +
                    detect_param_.label_num();

                int diff_ind = bottom[from_ind]->offset(img_idx, channel_idx, 
                        row, col);
                bottom[from_ind]->mutable_cpu_diff()[diff_ind] = 
                    loc_pred_diff[ind++] / default_windows_[match_wins[i]].width;

                // center_y
                channel_idx++;
                diff_ind = bottom[from_ind]->offset(img_idx, channel_idx, 
                        row, col);
                bottom[from_ind]->mutable_cpu_diff()[diff_ind] = 
                    loc_pred_diff[ind++] / default_windows_[match_wins[i]].height;

                // width 
                channel_idx++;
                diff_ind = bottom[from_ind]->offset(img_idx, channel_idx, 
                        row, col);
                int t = ind;
                float w = bottom[from_ind]->data_at(img_idx, channel_idx, row, col);
                bottom[from_ind]->mutable_cpu_diff()[diff_ind] = 
                    loc_pred_diff[t] / default_windows_[match_wins[i]].width;
                    //loc_pred_diff[t] / w;
                ind++;

                //height 
                channel_idx++;
                diff_ind = bottom[from_ind]->offset(img_idx, channel_idx, 
                        row, col);
                t = ind;
                //float h = 
                bottom[from_ind]->mutable_cpu_diff()[diff_ind] = 
                    loc_pred_diff[t] / default_windows_[match_wins[i]].height;
                    //loc_pred_diff[t] / h;
                ind++;

                /*
                for (int j = 0; j < 4; j++) {
                    int channel_idx = ratio_scale_ind *
                        (detect_param_.label_num() + 4) +
                        detect_param_.label_num() + j;
                    int diff_ind = bottom[from_ind]->offset(img_idx, channel_idx,
                            row, col);
                    bottom[from_ind]->mutable_cpu_diff()[diff_ind] = 
                        loc_pred_diff[ind++];
                }
                */
            }
        }
        CHECK_EQ(ind / 4, match_num_);
    }

    //back prop conf_loss
    if (match_num_ > 0) {
        std::vector<bool> conf_propagate_down;
        conf_propagate_down.push_back(true);
        conf_propagate_down.push_back(false);
        conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down, conf_bottom_vec_);
        Dtype normalizer = match_num_;
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
        caffe_scal(conf_pred_.count(), loss_weight, conf_pred_.mutable_cpu_diff());

        const Dtype* conf_pred_diff = conf_pred_.cpu_diff();

        int ind = 0;
        // for matched windows
        std::map<int, std::vector<int> >::iterator iter = gt2default_windows_.begin();
        for (; iter != gt2default_windows_.end(); iter++) {

            int gt_ind = iter->first;
            int img_idx = img_ind_for_label_data_[gt_ind];
            std::vector<int> match_wins = iter->second;
            for (int i = 0; i < match_wins.size(); i++) {
                int from_ind = default_windows_[match_wins[i]].from_index;
                int ratio_scale_ind = default_windows_[match_wins[i]]
                    .ratio_scale_index;
                int row = default_windows_[match_wins[i]].row;
                int col = default_windows_[match_wins[i]].col;

                for (int label_ind = 0; label_ind < detect_param_.label_num();
                        label_ind++) {
                    int channel_ind = ratio_scale_ind * 
                        (detect_param_.label_num() + 4) + label_ind;
                    int diff_ind = bottom[from_ind]->offset(img_idx, channel_ind, 
                            row, col);
                    bottom[from_ind]->mutable_cpu_diff()[diff_ind] = 
                        conf_pred_diff[ind++];
                }
            }
        }

        CHECK_EQ(ind / detect_param_.label_num(), match_num_);


        // for negative windows;
        for (int i = 0; i < neg_windows_.size(); i++) {
            NegWin neg_win = neg_windows_[i];
            int from_ind = default_windows_[neg_win.window_index].from_index;
            int ratio_scale_ind =
                default_windows_[neg_win.window_index].ratio_scale_index;
            int row = default_windows_[neg_win.window_index].row;
            int col = default_windows_[neg_win.window_index].col;
            int img_idx = neg_win.img_index;
            
            for (int label_ind = 0; label_ind < detect_param_.label_num();
                    label_ind++) {
                int channel_ind = ratio_scale_ind * 
                    (detect_param_.label_num() + 4) + label_ind;
                int diff_ind = bottom[from_ind]->offset(img_idx, channel_ind,
                        row, col);
                bottom[from_ind]->mutable_cpu_diff()[diff_ind] = 
                    conf_pred_diff[ind++];
            }
        }

        CHECK_EQ(ind / detect_param_.label_num() - match_num_, 
                neg_windows_.size());
    }

    /*
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
  */

}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::generate_default_windows() {
    // construct the default windows
    int window_index = 0;
    DetectionParam detection_param = this->layer_param_.detection_param();

    DefaultWindowIndexStruct default_win;

    for (int from_index = 0;
            from_index < detection_param.default_box_param().size();
            from_index++) {
        DefaultBoxParam default_box_param = 
            detection_param.default_box_param(from_index);
        for (int ratio_scale_index = 0;
                ratio_scale_index < default_box_param.ratio_scale().size();
                ratio_scale_index++) {
            for (int row = 0; row < default_box_param.img_height(); row++) {
                for (int col = 0; col < default_box_param.img_width(); col++) {
                    float center_row = ((float)row + 0.5) 
                        / (float)default_box_param.img_height();
                    float center_col = ((float)col + 0.5) / 
                        (float)default_box_param.img_width();

                    default_win.from_index = from_index;
                    default_win.ratio_scale_index = ratio_scale_index;
                    default_win.center_row = center_row;
                    default_win.center_col = center_col;
                    default_win.window_index = window_index;
                    default_win.row = row;
                    default_win.col = col;

                    float ratio = detect_param_.default_box_param(from_index).ratio_scale(ratio_scale_index).ratio();
                    float scale = detect_param_.default_box_param(from_index).ratio_scale(ratio_scale_index).scale();
                    
                    default_win.width = scale * sqrt(ratio);
                    default_win.height = scale / sqrt(ratio);

                    float xmin = center_col - 0.5 * default_win.width;
                    float xmax = center_col + 0.5 * default_win.width;
                    float ymin = center_row - 0.5 * default_win.height;
                    float ymax = center_row + 0.5 * default_win.height;

                    xmin = (xmin > 0 ? xmin : 0);
                    ymin = (ymin > 0 ? ymin : 0);
                    xmax = (xmax < 1 ? xmax : 1);
                    ymax = (ymax < 1 ? ymax : 1);

                    default_win.center_col = (xmin + xmax) * 0.5;
                    default_win.center_row = (ymin + ymax) * 0.5;
                    default_win.width = xmax - xmin;
                    default_win.height = ymax - ymin;

                    default_windows_.push_back(default_win);

                    window_index++;
                }
            }
        }
    } 
}

template <typename Dtype> 
void SddDetectionLossLayer<Dtype>::get_match_and_negatives(
        const vector<Blob<Dtype>*>& bottom) {
    int mini_batch_size = bottom[0]->num();

    gt_data_.clear();
    gt_data_ = std::vector<std::vector<GtData> >(mini_batch_size);

    // convert ground_truth data to gt_data_
    Blob<Dtype>* p_label_blob = bottom[bottom.size() - 1];
    if (detect_param_.check_match_res()) {
        p_label_blob = bottom[bottom.size() - 2];
    }

    //LOG(INFO) << "label blob height: " << p_label_blob->height();
    for (int i = 0; i <  p_label_blob->height(); i++) {
        GtData d;
        d.img_idx = p_label_blob->data_at(0, 0, i, 0);
        d.label = p_label_blob->data_at(0, 0, i, 1);
        d.gt_index = i;
        d.xmin = p_label_blob->data_at(0, 0, i, 2);
        d.ymin = p_label_blob->data_at(0, 0, i, 3);
        d.xmax = p_label_blob->data_at(0, 0, i, 4);
        d.ymax = p_label_blob->data_at(0, 0, i, 5);
        gt_data_[d.img_idx].push_back(d);

        img_ind_for_label_data_.push_back(p_label_blob->data_at(0, 0, i, 0));
    }
    //LOG(INFO) << "gt_data_ size: " << gt_data_.size();
    //LOG(INFO) << "gt_data_[0] size" << gt_data_[0].size();

    for (int img_idx = 0; img_idx < mini_batch_size; img_idx++) {
        //LOG(INFO) << gt_data_[img_idx].size() << " in " << img_idx;
        std::vector<std::vector<float> > match_scores(default_windows_.size());
        for (int i = 0; i < default_windows_.size(); i++) {
            for (int j = 0; j < gt_data_[img_idx].size(); j++) {
                match_scores[i].push_back(0);
            }
        }
        get_match_score(match_scores, img_idx);
        int match_num = get_match_result(match_scores, img_idx);
        get_top_score_negatives(bottom, match_num, img_idx);
    }
}

// this struct is used only in function get_top_score_negatives()
struct WinConf {
    float conf;
    int win_ind;
    bool operator<(const WinConf& b) const { conf < b.conf; }
};
template <typename Dtype>
void SddDetectionLossLayer<Dtype>::get_top_score_negatives(
        const std::vector<Blob<Dtype>*>& bottom, int match_num, int img_idx) {

    std::vector<WinConf> win_scores;

    if (match_num == 0) {
        return;
    }

    // run softmax firstly to get score, then select windows with maximum score
    Dtype* original_data = original_score_.mutable_cpu_data();
    int original_data_ind = 0;
    for (int win_ind = 0; win_ind < default_windows_.size(); win_ind++) {
        int from_ind = default_windows_[win_ind].from_index;
        int ratio_scale_ind = default_windows_[win_ind].ratio_scale_index;
        int row = default_windows_[win_ind].row;
        int col = default_windows_[win_ind].col;

        for (int label = 0; label < detect_param_.label_num(); label++) {
            int channel_idx = ratio_scale_ind * (detect_param_.label_num() + 4) 
                + label;
            original_data[original_data_ind++] = bottom[from_ind]->
                data_at(img_idx, channel_idx, row, col);
        }
    }
    CHECK_EQ(original_data_ind,
            default_windows_.size()*detect_param_.label_num());
    softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    

    const Dtype* prob_data = prob_.cpu_data();
    for (int win_ind = 0; win_ind < default_windows_.size(); win_ind++) {
        if (is_matched_[win_ind] > 0) {
            continue;
        }

        int data_ind = win_ind * detect_param_.label_num(); 
        float max_conf = prob_data[data_ind];
        int max_label = 0;
        for (int label_id = 1; label_id < detect_param_.label_num(); label_id++) {
            float tmp_conf = prob_data[data_ind + label_id];
            if (tmp_conf > max_conf) {
                max_conf = tmp_conf;
                max_label = label_id;
            }
        }
        
        if (max_label == 0) {
            continue;
        }

        WinConf win_conf;
        win_conf.conf = max_conf;
        win_conf.win_ind = win_ind;
        win_scores.push_back(win_conf);
    }
    std::sort(win_scores.begin(), win_scores.end()); 
    int neg_num = match_num * detect_param_.neg_pos_ratio();
    neg_num = (neg_num < win_scores.size() ? neg_num : win_scores.size());
    
    for (int i = 0; i < neg_num; i++) {
        WinConf win_conf = win_scores[win_scores.size() - 1 - i];
        NegWin neg_win;
        neg_win.img_index = img_idx;
        neg_win.window_index = win_conf.win_ind;
        neg_windows_.push_back(neg_win);
    }
}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::check_match_result(const vector<Blob<Dtype>*>& bottom) {
    int num = bottom[0]->num();
    Blob<Dtype>* p_label_blob = bottom[bottom.size() - 2];
    Blob<Dtype>* p_img_data = bottom[bottom.size() - 1];
    for (int i = 0; i < num; i++) {
        cv::Mat img(300, 300, CV_8UC3, cv::Scalar(255, 255, 255));

        //
        unsigned char* img_data = (unsigned char*)(img.data);
        for (int row_ind = 0; row_ind < 300; row_ind++) {
            for (int col_ind = 0; col_ind < 300; col_ind++) {
                img_data[img.step * row_ind + col_ind * 3] = p_img_data->data_at(i, 0, row_ind, col_ind) + 104;
                img_data[img.step * row_ind + col_ind * 3 + 1] = p_img_data->data_at(i, 1, row_ind, col_ind) + 117;
                img_data[img.step * row_ind + col_ind * 3 + 2] = p_img_data->data_at(i, 2, row_ind, col_ind) + 123;
            }
        }

        int rand1 = rand();

        std::map<int, std::vector<int> >::iterator it;
        for (it = gt2default_windows_.begin(); it != gt2default_windows_.end(); it++) {
            int label_index = it->first;
            /*
            if (label_index != 1) {
                continue;
            }
            */
            if (p_label_blob->data_at(0, 0, label_index, 0) != i) {
                continue;
            }
            std::vector<int>& matched_wins = it->second;
            //LOG(INFO) << "matched windows num " << matched_wins.size();
            for (int j = 0; j < matched_wins.size(); j++) {
                DefaultWindowIndexStruct win = default_windows_[matched_wins[j]];
                float ratio = detect_param_.default_box_param(win.from_index).ratio_scale(win.ratio_scale_index).ratio();
                float scale = detect_param_.default_box_param(win.from_index).ratio_scale(win.ratio_scale_index).scale();
                float w = win.width;
                //float h = scale / sqrt(ratio);
                float h = win.height;
                float xmin = win.center_col - 0.5 * w;
                float ymin = win.center_row - 0.5 * h;

                /*
                xmin = (xmin > 0 ? xmin : 0);
                ymin = (ymin > 0 ? ymin : 0);
                
                if (xmin + w > 1) {
                    w = w - (xmin + w - 1);
                }
                if (ymin + h > 1) {
                    h = h - (ymin + h - 1);
                }
                */

                xmin *= 300;
                ymin *= 300;
                w *= 300;
                h *= 300;

                CHECK_GE(xmin, 0);
                CHECK_GE(300, ymin + h);

                float gt_center_x = p_label_blob->data_at(0, 0, label_index, 2);
                float gt_center_y = p_label_blob->data_at(0, 0, label_index, 3);
                float gt_w = p_label_blob->data_at(0, 0, label_index, 4);
                float gt_h = p_label_blob->data_at(0, 0, label_index, 5);

                int gt_x = (gt_center_x - 0.5 * gt_w) * 300;
                int gt_y = (gt_center_y - 0.5 * gt_h) * 300;
                int gt_w_int = (gt_w)*300;
                int gt_h_int = (gt_h) * 300;


                cv::Mat tmp_img = img.clone();

                cv::rectangle(tmp_img, cv::Rect(gt_x, gt_y, gt_w_int, gt_h_int), cv::Scalar(0, 0, 255));

                if (j == 0) {
                    cv::rectangle(tmp_img, cv::Rect(xmin, ymin, w, h), cv::Scalar(0, 255, 0));
                }
                else {
                    cv::rectangle(tmp_img, cv::Rect(xmin, ymin, w, h), cv::Scalar(255, 0, 0));
                }
                char imgname[500];
                std::string save_name = detect_param_.match_res_save_path() + "/%d_%d_%d_%d.jpg";
                snprintf(imgname, 500, save_name.c_str(), i, rand1, label_index, j);
                cv::imwrite(imgname, tmp_img);
            }

        }
    }

}

template <typename Dtype>
int SddDetectionLossLayer<Dtype>::get_match_result(
        std::vector<std::vector<float> >& scores, int img_idx) {
    int res = 0;

    //std::vector<int> is_matched(default_windows_.size(), 0);
    is_matched_ = vector<int>(default_windows_.size(), 0);


    // firstly match gt to default box with highest overlapping area
    for (int i = 0; i < gt_data_[img_idx].size(); i++) {
        float max_score = 0;
        int max_index = -1;
        for (int j = 0; j < scores.size(); j++) {
            if (scores[j][i] >= max_score) {
                max_score = scores[j][i];
                max_index = j;
            }
        }
        int gt_global_index = gt_data_[img_idx][i].gt_index;
        if (gt2default_windows_.find(gt_global_index) == gt2default_windows_.end()) {
            std::vector<int> matched_wins;
            matched_wins.push_back(max_index);
            gt2default_windows_.insert(std::pair<int, std::vector<int> >
                    (gt_global_index, matched_wins));
        }
        else {
            gt2default_windows_[gt_global_index].push_back(max_index);
        }
        is_matched_[max_index] = 1;
        match_num_++;
        res++;
        //LOG(INFO) << "max match score: " << max_score;
    }

    // secondly, reversely matching, i.e. match each default window to any gt windows with a overlap rate being larger than 0.5
    for (int i = 0; i < scores.size(); i++) {
        if (is_matched_[i] == 1) continue;
        for (int j = 0; j < gt_data_[img_idx].size(); j++) {
            if (scores[i][j] > 0.5) {
                int gt_global_index = gt_data_[img_idx][j].gt_index;
                CHECK(gt2default_windows_.find(gt_global_index) 
                        != gt2default_windows_.end());
                gt2default_windows_[gt_global_index].push_back(i);
                match_num_++;
                res++;
                is_matched_[i] = 1;
                break;
            }
        }
    }

    //LOG(INFO) << "matched num:" << match_num_;
    return res;
}

template <typename Dtype>
void SddDetectionLossLayer<Dtype>::get_match_score(
        std::vector<std::vector<float> >& scores, int img_idx) {
    std::vector<GtData> gt_data = gt_data_[img_idx];
    

    for (int i = 0; i < default_windows_.size(); i++) {
        int from_index = default_windows_[i].from_index;
        int ratio_scale_index = default_windows_[i].ratio_scale_index;
        float center_row = default_windows_[i].center_row;
        float center_col = default_windows_[i].center_col;

        float pred_w = default_windows_[i].width;
        float pred_h = default_windows_[i].height;

        float pred_xmin = center_col - 0.5 * pred_w;
        float pred_ymin = center_row - 0.5 * pred_h;
        float pred_xmax = center_col + 0.5 * pred_w;
        float pred_ymax = center_row + 0.5 * pred_h;
        
        CHECK_GE(pred_xmin, 0);
        CHECK_GE(1, pred_ymax);

        for (int j = 0; j < gt_data.size(); j++) {
            float gt_xmin = gt_data[j].xmin - 0.5 * gt_data[j].xmax;
            float gt_ymin = gt_data[j].ymin - 0.5 * gt_data[j].ymax;
            float gt_xmax = gt_data[j].xmin + 0.5 * gt_data[j].xmax;
            float gt_ymax = gt_data[j].ymin + 0.5 * gt_data[j].ymax;

            CHECK_GE(gt_xmax , gt_xmin) << "center_x" << gt_data[j].xmin << " w=" << gt_data[j].xmax;
            CHECK_GE(gt_ymax , gt_ymin) << "center_y" << gt_data[j].ymin << " h=" << gt_data[j].ymax;

            float overlap_area = 0;
            if (pred_xmin > gt_xmax || gt_xmin > pred_xmax) {
                scores[i][j] = 0;
                continue;
            }
            if (pred_ymin > gt_ymax || gt_ymin > pred_ymax) {
                scores[i][j] = 0;
                continue;
            }
            float overlap_xmin = (pred_xmin > gt_xmin ? pred_xmin : gt_xmin);
            float overlap_ymin = (pred_ymin > gt_ymin ? pred_ymin : gt_ymin);
            float overlap_xmax = (pred_xmax < gt_xmax ? pred_xmax : gt_xmax);
            float overlap_ymax = (pred_ymax < gt_ymax ? pred_ymax : gt_ymax);

            overlap_area = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin);
            float total = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin) +
                (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) - overlap_area;
            float tmp_score = overlap_area / total;
            CHECK_GE(tmp_score, 0);
            scores[i][j] = tmp_score;
        }
    }
}


INSTANTIATE_CLASS(SddDetectionLossLayer);
REGISTER_LAYER_CLASS(SddDetectionLoss);

}  // namespace caffe
