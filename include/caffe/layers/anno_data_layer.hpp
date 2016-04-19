#ifndef ANNO_DATA_LAYER_HPP_ 
#define ANNO_DATA_LAYER_HPP_ 

#include <vector>
#include <string>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class AnnoDataLayer: public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit AnnoDataLayer(const LayerParameter& param);
  virtual ~AnnoDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "AnnoData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  //DataReader reader_;

  // newly added member variable
  std::vector<std::string> list_vec_;
  int img_fetch_index_;
  std::map<std::string, std::vector<std::vector<float> > > labels_;
  int ready_to_load_;
};

}  // namespace caffe

#endif  // CAFFE_AnnoDATA_LAYER_HPP_
