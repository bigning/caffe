#ifndef SDD_DETECTION_LOSS_HPP_ 
#define SDD_DETECTION_LOSS_HPP_ 

#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * @brief Computes the multinomial logistic loss for a one-of-many
 *        classification task, passing real-valued predictions through a
 *        softmax to get a probability distribution over classes.
 *
 * This layer should be preferred over separate
 * SoftmaxLayer + MultinomialLogisticLossLayer
 * as its gradient computation is more numerically stable.
 * At test time, this layer can be replaced simply by a SoftmaxLayer.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ x @f$, a Blob with values in
 *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
 *      the @f$ K = CHW @f$ classes. This layer maps these scores to a
 *      probability distribution over classes using the softmax function
 *      @f$ \hat{p}_{nk} = \exp(x_{nk}) /
 *      \left[\sum_{k'} \exp(x_{nk'})\right] @f$ (see SoftmaxLayer).
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      the labels @f$ l @f$, an integer-valued Blob with values
 *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
 *      indicating the correct class label among the @f$ K @f$ classes
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed cross-entropy classification loss: @f$ E =
 *        \frac{-1}{N} \sum\limits_{n=1}^N \log(\hat{p}_{n,l_n})
 *      @f$, for softmax output class probabilites @f$ \hat{p} @f$
 */
template <typename Dtype>
class SddDetectionLossLayer: public LossLayer<Dtype> {
 public:
   /**
    * @param param provides LossParameter loss_param, with options:
    *  - ignore_label (optional)
    *    Specify a label value that should be ignored when computing the loss.
    *  - normalize (optional, default true)
    *    If true, the loss is normalized by the number of (nonignored) labels
    *    present; otherwise the loss is simply summed over spatial locations.
    */
  explicit SddDetectionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SddDetectionLoss"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /**
   * @brief Computes the softmax loss error gradient w.r.t. the predictions.
   *
   * Gradients cannot be computed with respect to the label inputs (bottom[1]),
   * so this method ignores bottom[1] and requires !propagate_down[1], crashing
   * if propagate_down[1] is set.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   *      propagate_down[1] must be false as we can't compute gradients with
   *      respect to the labels.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$; Backward computes diff
   *      @f$ \frac{\partial E}{\partial x} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels -- ignored as we can't compute their error gradients
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<Layer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  Blob<Dtype> original_score_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// How to normalize the output loss.
  LossParameter_NormalizationMode normalization_;

  int softmax_axis_, outer_num_, inner_num_;


  // newly added for this layer
  void generate_default_windows();
  void get_match_and_negatives(const vector<Blob<Dtype>*>& bottom);
  void get_match_score(std::vector<std::vector<float> >& scores, int img_idx);
  // get match windows indices and return the number of matched default windows
  int get_match_result(std::vector<std::vector<float> >& scores, int img_idx);
  void check_match_result(const std::vector<Blob<Dtype>*>& bottom); // draw matchecd rectangle on image

  // match_num: the number of matched default windows, it's used to decide how many negative windows will be selected
  // img_idx: which image in the mini_batch_data is being processed.
  void get_top_score_negatives(const std::vector<Blob<Dtype>*>& bottom, int match_num, int img_idx);

  //generate bottom data for two loss layers
  void prepare_for_conf_loss(const std::vector<Blob<Dtype>*>& bottom);
  void prepare_for_loc_loss(const std::vector<Blob<Dtype>*>& bottom);

  // to clear some vector or map variables
  void forward_init();

  //
  void initialize_bottom_diff(const vector<Blob<Dtype>*>& bottom);

  shared_ptr<Layer<Dtype> > conf_loss_layer_;
  shared_ptr<Layer<Dtype> > loc_loss_layer_;

  //bottom and top vec for conf_loss_layer_
  std::vector<Blob<Dtype>*> conf_bottom_vec_;
  std::vector<Blob<Dtype>*> conf_top_vec_;
  Blob<Dtype> conf_pred_;
  Blob<Dtype> conf_gt_;
  Blob<Dtype> conf_loss_top_;

  // bottom and top vec for loc_loss_layer_
  std::vector<Blob<Dtype>*> loc_bottom_vec_;
  std::vector<Blob<Dtype>*> loc_top_vec_;
  Blob<Dtype> loc_pred_;
  Blob<Dtype> loc_gt_;
  Blob<Dtype> loc_loss_top_;

  //Detection parameter
  DetectionParam detect_param_;


  // define the struct to index a default window
  struct DefaultWindowIndexStruct {
      int from_index; // which bottom layer this window is from
      int ratio_scale_index; // 
      // the default window center normalized coordinate
      float center_row;
      float center_col;
      int window_index;

      // the default window center original coordinate (0,1,2,,,)
      int row;
      int col;

      //default window height and width;
      float width;
      float height;
  };
  std::vector<DefaultWindowIndexStruct> default_windows_;
  std::map<int, std::vector<int> > gt2default_windows_;

  std::vector<int> img_ind_for_label_data_;

  struct GtData {
    int img_idx;
    int label;
    int gt_index; // index in 3rd dimension of "label" blob
    float xmin;
    float ymin;
    float xmax;
    float ymax;
  };

  std::vector<std::vector<GtData> > gt_data_;
  int match_num_;
  std::vector<int> is_matched_; // record match result for each default box

  struct NegWin {
      // record which image in the mini_batch_data this negative window is from
      int img_index; 
      int window_index; 
  };
  std::vector<NegWin> neg_windows_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
