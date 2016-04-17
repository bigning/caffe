#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include "caffe/util/io.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/visualize_bottom_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void VisualizeBottomLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    batch_index_= 0;
    inner_index_= 0;
}

template <typename Dtype>
void VisualizeBottomLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Blob<Dtype>* p_data = bottom[0];
    Blob<Dtype>* p_label = bottom[1];

    int rows = p_data->height();
    int columns = p_data->width();
    LOG(INFO) << p_label->height() << " " << p_label->width(); 

    for (int i = 0; i < p_data->num(); i++) {
        cv::Mat img(rows, columns, CV_8UC3, cv::Scalar(0));
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < rows; h++) {
                for (int w = 0; w < columns; w++) {
                    img.data[img.step*h+w*3 + c] = p_data->data_at(i, c, h, w);
                }
            }
        }

       for (int label_index = 0; label_index < p_label->height(); label_index++) {
            int img_index = p_label->data_at(0, 0, label_index, 0);
            if (img_index != i) {
                continue;
            }
            int label_name = p_label->data_at(0, 0, label_index, 1);
            int x = p_label->data_at(0,0, label_index, 2);
            int y = p_label->data_at(0,0, label_index, 3);
            int w = p_label->data_at(0,0, label_index, 4);
            int h = p_label->data_at(0,0, label_index, 5);
            cv::rectangle(img, cv::Rect(x,y,w,h), cv::Scalar(0,0,255));
       } 
       char filename[1000];
       snprintf(filename, 1000, "%s/%d_%d.jpg",this->layer_param().visualize_bottom_param().img_save_path().c_str(), i, rand()%1000);
       cv::imwrite(filename, img);
    }
  }


INSTANTIATE_CLASS(VisualizeBottomLayer);
REGISTER_LAYER_CLASS(VisualizeBottom);

}  // namespace caffe
