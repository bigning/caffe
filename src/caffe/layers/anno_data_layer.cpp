#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
#include <iostream>

#include <algorithm>
#include <vector>

#include "caffe/util/io.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/layers/anno_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
AnnoDataLayer<Dtype>::AnnoDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param), ready_to_load_(0)
    //reader_(param) {
{
}

template <typename Dtype>
AnnoDataLayer<Dtype>::~AnnoDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "anno_data_laeyr setup";
  const int batch_size = this->layer_param_.anno_data_param().batch_size();

  // load the list file
  std::ifstream list_file(this->layer_param_.anno_data_param().list_file().c_str()); 
  CHECK(list_file.is_open());
  std::string line;
  while (list_file >> line) {
      list_vec_.push_back(line);
  }
  list_file.close();
  CHECK(list_vec_.size() >= 1);
  LOG(INFO) << "There is totally " << list_vec_.size() << " images";

  // Read a data point, and use it to initialize the top blob.
  //Datum& datum = *(reader_.full().peek());
  Datum datum;
  const std::string img_name = this->layer_param_.anno_data_param().image_path() + "/" + list_vec_[0] + ".jpg";
  bool status = ReadImageToDatum(img_name, 0, this->layer_param_.anno_data_param().resize_h(), this->layer_param_.anno_data_param().resize_w(), true, "jpg", &datum);
  CHECK(status) << "Read first image failed";

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // read in all labels
  for (int i = 0; i < list_vec_.size(); i++) {
      std::string label_file_name = this->layer_param_.anno_data_param().gt_path() + "/" + list_vec_[i] + ".txt";
      // TODO: 04/14
      std::vector<std::vector<int> > l;
      int label = 0;
      int xmin = 0;
      int ymin = 0;
      int xmax = 0;
      int ymax = 0;
      /*
      std::ifstream gt_file(label_file_name.c_str());
      while (gt_file >> label >> xmin >> ymin >> xmax >> ymax) {
          std::vector<int> tmp_labels;
          tmp_labels.push_back(label);
          tmp_labels.push_back(xmin);
          tmp_labels.push_back(ymin);
          tmp_labels.push_back(xmax - xmin + 1);
          tmp_labels.push_back(ymax - ymin + 1);

          l.push_back(tmp_labels);
      }
      labels_.insert(std::pair<std::string, std::vector<std::vector<int> > >(list_vec_[i], l));
      gt_file.close();
      */
  }
  LOG(INFO) << "from anno_data_layer: all lables has been read";
  

  // initialize the index
  img_fetch_index_ = 0;
  // label
  if (this->output_labels_) {
    //[TODO]: top label blob size is (1,1,num_labels,6)
    //6 means (img_id, label_id, x, y, w, h),
    //here, when seting up this layer, we don't know how many
    //objectes there are in the mini-batch data, so, set num_labels
    //to be 1, reset it in load_batch
    //vector<int> label_shape(1, batch_size);
    vector<int> label_shape;
    label_shape.push_back(1);
    label_shape.push_back(1);
    label_shape.push_back(1);
    label_shape.push_back(6);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
  ready_to_load_ = 1;
}

template <typename Dtype> std::vector<std::vector<float> >
AnnoDataLayer<Dtype>::read_and_transform_img(std::string& img_id,
        cv::Mat& img) {
    std::vector<std::vector<float> > labels;
    
    std::string full_img_name = this->layer_param_.anno_data_param().image_path() +
        "/" + img_id + ".jpg";
    img = cv::imread(full_img_name);
    CHECK(img.data) << "load image failed: " << full_img_name;

    int dest_size_w = this->layer_param_.anno_data_param().resize_w();
    int dest_size_h = this->layer_param_.anno_data_param().resize_h();

    // read labels
    labels.clear();
    std::string label_name = this->layer_param_.anno_data_param().gt_path() + "/"
        + img_id + ".txt";

    std::ifstream gt_file(label_name.c_str());
    CHECK(gt_file.is_open()) << "Fail to open gt file of " << label_name;
    int label = 0;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
    while (gt_file >> label >> xmin >> ymin >> xmax >> ymax) {
        std::vector<float> tmp_label;
        
        float w_mmax = (float)(img.cols);
        float h_mmax = (float)(img.rows);
        xmin = (xmin)/w_mmax;
        ymin = (ymin)/h_mmax;
        xmax = (xmax)/w_mmax;
        ymax = (ymax)/h_mmax;
        tmp_label.push_back(label);

        float center_x = (xmin+xmax) / 2.0;
        float center_y = (ymin + ymax) / 2.0;
        tmp_label.push_back(center_x);
        tmp_label.push_back(center_y);
        tmp_label.push_back(xmax - xmin);
        tmp_label.push_back(ymax - ymin);

        labels.push_back(tmp_label); 
  }

    // img transform
    // use original image or randomly select a patch
    int rand1 = rand()%10;
    if (rand1 <= 4) {
        // orignal image
        cv::resize(img, img, cv::Size(dest_size_w, dest_size_h)); 
        /*
        float x_ratio = ((float)(dest_size_w)) / (float)img.cols;
        float y_ratio = ((float)(dest_size_h)) / (float)img.rows;
        for (int i = 0; i < labels.size(); i++) {
            labels[i][1] *= x_ratio;
            labels[i][2] *= y_ratio;
            labels[i][3] *= x_ratio;
            labels[i][4] *= y_ratio;
        }
        */
    }
    else {
        // randomly select a patch from original image
        float rnd_size = ((double)rand() / (RAND_MAX)) * 0.6 + 0.4;
        float rnd_ratio = ((double)rand() / (RAND_MAX)) * 1.5 + 0.5;
        float width = (rnd_size / sqrt(rnd_ratio)) * (float)(img.cols);
        float height = (rnd_size * sqrt(rnd_ratio)) * (float)(img.rows);

        if (width > img.cols - 0.1) {
            width = img.cols;
        }
        if (height > img.rows - 0.1) {
            height = img.rows;
        }

        float xmin_range = ((float)img.cols) - width;
        float ymin_range = ((float)img.rows) - height;

        float xstart_f = ((double)rand()/(RAND_MAX)) * xmin_range;
        float ystart_f = ((double)rand()/(RAND_MAX)) * ymin_range;
        int xstart = (xstart_f > 0 ? xstart_f:0); 
        int ystart = (ystart_f > 0 ? ystart_f:0); 
        int xend = (((int)(xstart + width - 1)) < img.cols ? ((int)(xstart + width - 1)) : img.cols - 1);
        int yend = (((int)(ystart + height - 1)) < img.rows? ((int)(ystart + height - 1)) : img.rows - 1);

        float delta_x = ((float)xstart) / (float)img.cols;
        float delta_y = ((float)ystart) / (float)img.rows;
        float xratio = ((float)(xend - xstart))/(float)(img.cols);
        float yratio = ((float)(yend - ystart))/(float)(img.rows);

        std::vector<std::vector<float> > new_label;
        std::vector<int> valid_index;
        for (int i = 0; i < labels.size(); i++) {
            labels[i][1] = (labels[i][1] - delta_x) / xratio;
            labels[i][2] = (labels[i][2] - delta_y) / yratio;
            labels[i][3] = (labels[i][3] ) / xratio;
            labels[i][4] = (labels[i][4] ) / yratio;
            float center_x = labels[i][1] ;
            float center_y = labels[i][2];

            if (center_x < 0 || center_x > 1 || center_y < 0 || center_y > 1) {

            }
            else {
                labels[i][1] = (labels[i][1] > 0 ? labels[i][1] : 0);
                labels[i][2] = (labels[i][2] > 0 ? labels[i][2] : 0);
                labels[i][3] = (labels[i][3] < 1 ? labels[i][3] : 1);
                labels[i][4] = (labels[i][4] < 1 ? labels[i][4] : 1);
                valid_index.push_back(i);
            }
        }
        for (int i = 0; i < valid_index.size(); i++) {
            new_label.push_back(labels[valid_index[i]]);
        }
        labels.clear();
        labels = new_label;

        cv::Mat new_img = img(cv::Rect(xstart, ystart, xend - xstart + 1, yend - ystart + 1));
        cv::resize(new_img, img, cv::Size(dest_size_w, dest_size_h));
    }
    //flip 
    int rand2 = rand()%10;
    if (rand2 <= 4) {
        //flip
        cv::flip(img, img, 1);
        for (int i = 0; i < labels.size(); i++) {
            float tmp = labels[i][1];
            labels[i][1] = 1 - labels[i][1];
            labels[i][3] = 1 - tmp;
        }
    }
    return labels;
}


// This function is called on prefetch thread
template<typename Dtype>
void AnnoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    std::vector<std::vector<std::vector<float> > > labels_vec;
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());


  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.anno_data_param().batch_size();
  //Datum& datum = *(reader_.full().peek());
  Datum datum;
  const std::string img_name = this->layer_param_.anno_data_param().image_path() + "/" + list_vec_[0] + ".jpg";
  bool status = ReadImageToDatum(img_name, 0, this->layer_param_.anno_data_param().resize_h(), this->layer_param_.anno_data_param().resize_w(), true, "jpg", &datum);
  CHECK(status) << "Read first image failed";

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  std::vector<std::string> mini_batch_img_names;

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  // record original image size and transform param if do transformation to input image. 
  std::vector<std::vector<int> > img_size_translate_param;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();


    if (img_fetch_index_ >= list_vec_.size()) {
        img_fetch_index_ = 0;
        if (this->layer_param_.anno_data_param().istest() == false) { 
            std::random_shuffle(list_vec_.begin(), list_vec_.end());
        }
    }
    const std::string img_name_2 = this->layer_param_.anno_data_param().image_path() + "/" + list_vec_[img_fetch_index_] + ".jpg";
    // get a datum
    //Datum& datum = *(reader_.full().pop("Waiting for data"));
    Datum datum;
    //cv::Mat img = cv::imread(img_name_2);
    cv::Mat img;

    std::vector<std::vector<float> > this_label;

    this_label = read_and_transform_img(list_vec_[img_fetch_index_], img);

    
    labels_vec.push_back(this_label);

    CHECK(img.data) << "Read image failed: " << img_name_2;
    std::vector<int> img_param;
    img_param.push_back(img.rows);
    img_param.push_back(img.cols);
    img_size_translate_param.push_back(img_param);
    //cv::resize(img, img, cv::Size(this->layer_param().anno_data_param().resize_w(), this->layer_param().anno_data_param().resize_h()));
    CVMatToDatum(img, &datum);
    datum.set_label(0); // set a fake label
    //status = ReadImageToDatum(img_name_2, 0, this->layer_param_.anno_data_param().resize_h(), this->layer_param_.anno_data_param().resize_w(), true, "jpg", &datum);
    
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    mini_batch_img_names.push_back(list_vec_[img_fetch_index_]);
    img_fetch_index_++;

    trans_time += timer.MicroSeconds();
    //reader_.free().push(const_cast<Datum*>(&datum));
  }
  // Copy label.
  // computing the number of all labels
  
  for (int n = 0; n < batch->data_.num(); n++) {
      for (int c = 0; c < batch->data_.channels(); c++) {
          for (int h = 0; h < batch->data_.height(); h++) {
              for (int w = 0; w < batch->data_.width(); w++) {
                  int data_ind = batch->data_.offset(n, c, h, w);
                  if (c == 0) {
                    batch->data_.mutable_cpu_data()[data_ind] -= 104;
                  }
                  if (c == 1) {
                    batch->data_.mutable_cpu_data()[data_ind] -= 117;
                  }
                  if (c == 2) {
                    batch->data_.mutable_cpu_data()[data_ind] -= 123;
                  }
              }
          }
      }
  }
  


  if (this->output_labels_) {
      int obj_nums = 0;


      for (int i = 0; i < labels_vec.size(); i++) {
          obj_nums += labels_vec[i].size();
      }


      std::vector<int> label_shape;
      label_shape.push_back(1);
      label_shape.push_back(1);
      label_shape.push_back(obj_nums);
      label_shape.push_back(6);
      batch->label_.Reshape(label_shape);
      top_label = batch->label_.mutable_cpu_data();

      int idx = 0;
      
      for (int i = 0; i < mini_batch_img_names.size(); i++) {
          std::vector<std::vector<float> >& tmp_label = labels_vec[i];

          for (int obj_index = 0; 
                  obj_index < tmp_label.size();
                  obj_index ++) {
              top_label[idx++] = i;
              top_label[idx++] = tmp_label[obj_index][0];
              top_label[idx++] = tmp_label[obj_index][1];
              top_label[idx++] = tmp_label[obj_index][2];
              top_label[idx++] = tmp_label[obj_index][3];
              top_label[idx++] = tmp_label[obj_index][4];
          }
      }

  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

}

INSTANTIATE_CLASS(AnnoDataLayer);
REGISTER_LAYER_CLASS(AnnoData);

}  // namespace caffe
