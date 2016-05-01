#include <algorithm>
#include <cfloat>
#include <vector>
#include <set>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <stdio.h>
#include <algorithm>

#include "caffe/layers/ssd_detect_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SsdDetectLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    detect_param_ = this->layer_param_.detection_param();

    Layer<Dtype>::LayerSetUp(bottom, top);

    // Set Up softmax_layer_ 
    LayerParameter softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);

    std::vector<int> original_score_shape;
    original_score_shape.push_back(1);
    original_score_shape.push_back(this->layer_param_.
            detection_param().label_num());
    original_score_.Reshape(original_score_shape);
    softmax_bottom_vec_.push_back(&original_score_);

    std::vector<int> prob_shape;
    prob_.Reshape(original_score_shape);
    softmax_top_vec_.push_back(&prob_);

    softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

    vector<int> output_shape;
    output_shape.push_back(1);
    output_shape.push_back(1);
    output_shape.push_back(1); // positive windows number

    //img_ind,label, x1, y1, x2, y2, conf[0], conf[1]...
    output_shape.push_back(6 + detect_param_.label_num());
    top[0]->Reshape(output_shape);

    //top.push_back(&output_);

    generate_default_windows();

}

template <typename Dtype>
void SsdDetectLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::Reshape(bottom, top);
}


template <typename Dtype>
void SsdDetectLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    mini_batch_ = bottom[0]->num();

    forward_init();
    get_positive_windows(bottom);

    if (results_.size() == 0) {
        std::vector<int> top_shape(1,1);
        top[0]->Reshape(top_shape);
        LOG(INFO) << "no positive windows";
        return;
    }

    prepare_softmax_data(bottom);
    softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

    get_softmax_result();

    LOG(INFO) << "before refine, there are " << results_.size() << " res";

    refine_res();

    fill_top(top);
    LOG(INFO) << "after refine, there are " << results_.size() << " res";
}

template <typename Dtype>
void SsdDetectLayer<Dtype>::get_softmax_result() {
    const Dtype* prob_data = prob_.cpu_data();
    int prob_data_ind = 0;

    for (int i = 0; i < results_.size(); i++) {
        int label = results_[i].label;
        for (int label_ind = 0; 
                label_ind < detect_param_.label_num(); label_ind++) {
            results_[i].confs[label_ind] = prob_data[prob_data_ind++];
        }
        //LOG(INFO) << "max conf: " << results_[i].confs[results_[i].label];
    }
}

template <typename Dtype>
void SsdDetectLayer<Dtype>::fill_top(const vector<Blob<Dtype>*>& top) {
    std::vector<int> top_shape;
    top_shape.push_back(1);
    top_shape.push_back(1);
    top_shape.push_back(results_.size());
    top_shape.push_back(6 + detect_param_.label_num());
    top[0]->Reshape(top_shape);

    Dtype* output_data = top[0]->mutable_cpu_data();
    int output_ind = 0;

    for (int i = 0; i < results_.size(); i++) {
        output_data[output_ind++] = results_[i].img_ind;
        output_data[output_ind++] = results_[i].label;
        output_data[output_ind++] = results_[i].xmin;
        output_data[output_ind++] = results_[i].ymin;
        output_data[output_ind++] = results_[i].xmax;
        output_data[output_ind++] = results_[i].ymax;
        for (int j = 0; j < detect_param_.label_num(); j++) {
            output_data[output_ind++] = results_[i].confs[j];
        }
    }
}

template <typename Dtype>
void SsdDetectLayer<Dtype>::refine_res() {

    //step1. remove windows with a max-confidence less than a threshold
    std::vector<Result> tmp_res = results_;

    results_.clear();
    for (int i = 0; i < tmp_res.size(); i++) {
        if (tmp_res[i].confs[tmp_res[i].label] < detect_param_.min_conf()) {
            continue;
        }
        results_.push_back(tmp_res[i]);
    }

    if (detect_param_.do_nms()) {
        nms(mini_batch_);
    }
}

template <typename Dtype>
void SsdDetectLayer<Dtype>::nms(int img_num) {
    std::vector<std::vector<std::vector<Result> > >
        img_label_res(img_num);

    for (int i = 0; i < img_num; i++) {
        std::vector<vector<Result> > tmp(detect_param_.label_num());
        img_label_res[i] = tmp;
    }

    for (int i = 0; i < results_.size(); i++) {
        img_label_res[results_[i].img_ind][results_[i].label].push_back(
                results_[i]);
    }

    results_.clear();
    int t = 0;

    for (int i = 0; i < img_num; i++) {
        for (int label_ind = 0; label_ind < detect_param_.label_num(); label_ind++)
        {
            std::set<int> remove_ind;
            remove_ind.clear();
            if (img_label_res[i][label_ind].size() <= 1) {
                for (int m = 0; m < img_label_res[i][label_ind].size(); m++) {
                    results_.push_back(img_label_res[i][label_ind][m]);
                }
                continue;
            }
            LOG(INFO) << "label: " << label_ind << " " << img_label_res[i][label_ind].size();
            for (int m = 0; m < img_label_res[i][label_ind].size(); m++) {
                for (int n = m + 1; n < img_label_res[i][label_ind].size(); n++) {

                    float rate = get_overlap_score(img_label_res[i][label_ind][m],
                            img_label_res[i][label_ind][n]);
                    if (rate < 0.45) {

                        continue;
                    }
                    if (img_label_res[i][label_ind][m].confs[label_ind] > 
                            img_label_res[i][label_ind][n].confs[label_ind]) {
                        remove_ind.insert(n);
                    } 
                    else {
                        remove_ind.insert(m);
                    }
                }
            }


            for (int m = 0; m < img_label_res[i][label_ind].size(); m++) {
                if (remove_ind.find(m) == remove_ind.end()) {
                    results_.push_back(img_label_res[i][label_ind][m]);
                }
            }
        }
    }
}

template <typename Dtype>
float SsdDetectLayer<Dtype>::get_overlap_score(Result a, Result b) {
    CHECK_EQ(a.img_ind, b.img_ind);
    CHECK_EQ(a.label, b.label);

    if (a.xmin > b.xmax || b.xmin > a.xmax) {
        return 0;
    }
    if (a.ymin > b.ymax || b.ymin > a.ymax) {
        return 0;
    }

    float overlap_xmin = (a.xmin > b.xmin ? a.xmin : b.xmin);
    float overlap_ymin = (a.ymin > b.ymin ? a.ymin : b.ymin);
    float overlap_xmax = (a.xmax < b.xmax ? a.xmax : b.xmax);
    float overlap_ymax = (a.ymax < b.ymax ? a.ymax : b.ymax);

    float overlap_area = (overlap_xmax - overlap_xmin) *
        (overlap_ymax - overlap_ymin);
    float total = (a.xmax - a.xmin) * (a.ymax - a.ymin) +
                  (b.xmax - b.xmin) * (b.ymax - b.ymin) - overlap_area;
    return overlap_area / total;
}


template <typename Dtype>
void SsdDetectLayer<Dtype>::prepare_softmax_data(
        const std::vector<Blob<Dtype>*>& bottom) {
    std::vector<int> original_score_shape;
    original_score_shape.push_back(pos_wins_.size());
    original_score_shape.push_back(detect_param_.label_num());
    original_score_.Reshape(original_score_shape);

    Dtype* original_data = original_score_.mutable_cpu_data();

    prob_.Reshape(original_score_shape);


    int original_data_ind = 0;

    for (int i = 0; i < pos_wins_.size(); i++) {
        PositiveWindows pos_win = pos_wins_[i];
        int from_ind = default_windows_[pos_win.window_ind].from_index;
        int ratio_scale_ind = default_windows_[pos_win.window_ind].
            ratio_scale_index;
        int row = default_windows_[pos_win.window_ind].row;
        int col = default_windows_[pos_win.window_ind].col;

        for (int label_ind = 0; label_ind < detect_param_.label_num(); 
                label_ind++) {
            int channel_ind = ratio_scale_ind * (detect_param_.label_num() + 4) +
                label_ind;
            original_data[original_data_ind++] = bottom[from_ind]->data_at(
                    pos_win.img_ind, channel_ind, row, col); 
        }

    }

    CHECK_EQ(original_data_ind, pos_wins_.size() * detect_param_.label_num());
}

template <typename Dtype>
void SsdDetectLayer<Dtype>::get_positive_windows(
        const std::vector<Blob<Dtype>*>& bottom) {
    int mini_batch = bottom[0]->num();
    for (int img_ind = 0; img_ind < mini_batch; img_ind++) {
        for (int win_ind = 0; win_ind < default_windows_.size(); win_ind++) {
            int from_ind = default_windows_[win_ind].from_index;
            int ratio_scale_ind = default_windows_[win_ind].ratio_scale_index;
            int row = default_windows_[win_ind].row;
            int col = default_windows_[win_ind].col;

            int channel_ind = ratio_scale_ind * (detect_param_.label_num() + 4);
            float max_score = bottom[from_ind]->data_at(img_ind, channel_ind, 
                    row, col);
            int max_label = 0;

            for (int label_ind = 1; label_ind < detect_param_.label_num(); 
                    label_ind++) {
                channel_ind++;
                float tmp_score = bottom[from_ind]->data_at(img_ind, channel_ind, 
                        row, col);
                if (tmp_score > max_score) {
                    max_score = tmp_score;
                    max_label = label_ind;
                }
            }

            if (max_label == 0) {
                continue;
            }
            PositiveWindows pos_win;
            pos_win.img_ind = img_ind;
            pos_win.window_ind = win_ind;
            pos_wins_.push_back(pos_win);

            Result res;
            res.img_ind = img_ind;
            res.label = max_label;
            std::vector<float> tmp_vec(detect_param_.label_num(), -1);
            res.confs = tmp_vec;;
            channel_ind = ratio_scale_ind * (detect_param_.label_num() + 4)
                + detect_param_.label_num();

            float center_x = bottom[from_ind]->data_at(img_ind, channel_ind, row, col);
            float center_y = bottom[from_ind]->data_at(img_ind, channel_ind+1, row, col);
            float w = bottom[from_ind]->data_at(img_ind, channel_ind+2, row, col);
            float h = bottom[from_ind]->data_at(img_ind, channel_ind+3, row, col);

            res.xmin = center_x - 0.5 * w;
            res.ymin = center_y - 0.5 * h;
            res.xmax = center_x + 0.5 * w;
            res.ymax = center_y + 0.5 * h;

            float small_num = 0.000001;
            if (w < small_num || h < small_num) {
                continue;
            }

            results_.push_back(res);
        }
    }
}

template <typename Dtype>
void SsdDetectLayer<Dtype>::forward_init() {
    pos_wins_.clear();
    results_.clear();
    /*
    gt2default_windows_.clear();
    gt_data_.clear();
    match_num_= 0;
    is_matched_.clear();
    neg_windows_.clear();
    img_ind_for_label_data_.clear();
    */
}


template <typename Dtype>
void SsdDetectLayer<Dtype>::generate_default_windows() {
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

                    default_windows_.push_back(default_win);

                    window_index++;
                }
            }
        }
    } 
}




INSTANTIATE_CLASS(SsdDetectLayer);
REGISTER_LAYER_CLASS(SsdDetect);

}  // namespace caffe
