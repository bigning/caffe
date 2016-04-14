#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
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
void AnnoDataLayer<Dtype>::AnnoDataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
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
  }
  LOG(INFO) << "from anno_data_layer: all lables has been read";
  

  // initialize the index
  img_fetch_index_ = 0;
  // label
  /*
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
  */
  ready_to_load_ = 1;
}

// This function is called on prefetch thread
template<typename Dtype>
void AnnoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();

    if (img_fetch_index_ >= list_vec_.size()) {
        img_fetch_index_ = 0;
        std::random_shuffle(list_vec_.begin(), list_vec_.end());
    }
    const std::string img_name_2 = this->layer_param_.anno_data_param().image_path() + "/" + list_vec_[img_fetch_index_] + ".jpg";
    // get a datum
    //Datum& datum = *(reader_.full().pop("Waiting for data"));
    Datum datum;
    status = ReadImageToDatum(img_name_2, 0, this->layer_param_.anno_data_param().resize_h(), this->layer_param_.anno_data_param().resize_w(), true, "jpg", &datum);
    CHECK(status) << "Read image failed: " << img_name_2;
    
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
  if (this->output_labels_) {
      int obj_nums = 0;
      for (int i = 0; i < mini_batch_img_names.size(); i++) {
          if (labels_.find(mini_batch_img_names[i]) == labels_.end()) {
              // load label from gt file
              const std::string img_name = this->layer_param_.anno_data_param().gt_path() + "/" + mini_batch_img_names[i] + ".txt";
              std::ifstream gt_file(img_name.c_str());
              CHECK(gt_file.is_open()) << "Fail to open gt file of " << mini_batch_img_names[i];
              int label = 0, xmin = 0, ymin = 0, xmax = 0, ymax = 0;
              std::vector<std::vector<int> > label_for_img;
              while (gt_file >> label >> xmin >> ymin >> xmax >> ymax) {
                std::vector<int> tmp_label;
                tmp_label.push_back(label);
                tmp_label.push_back(xmin);
                tmp_label.push_back(ymin);
                tmp_label.push_back(xmax - xmin + 1);
                tmp_label.push_back(ymax - ymin + 1);
                label_for_img.push_back(tmp_label); 
              }
              labels_.insert(std::pair<std::string, std::vector<std::vector<int> > >(mini_batch_img_names[i], label_for_img));
          }
          obj_nums += labels_[mini_batch_img_names[i]].size();
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
          for (int obj_index = 0; obj_index < labels_[mini_batch_img_names[i]].size(); obj_index ++) {
              std::vector<std::vector<int> >& tmp_label = 
                  labels_[mini_batch_img_names[i]];
              top_label[idx++] = obj_index;
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
