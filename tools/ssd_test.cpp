#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/layers/memory_data_layer.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(list, "",
     "list file path");
DEFINE_string(img_path, "",
     "img_path file path");
DEFINE_string(txt_save_path, "",
        "result save path");
DEFINE_string(img_save_path, "",
        "img save path");
DEFINE_string(class_name_path, "",
        "label name file");

DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (gpus.size() > 1) {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  } else {
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


struct Res {
    int label;
    std::vector<float> confs;
    float xmin; 
    float ymin;
    float xmax;
    float ymax;
};

// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  std::map<int, std::string> label2name;

  if (FLAGS_class_name_path.size() > 0) {
      std::ifstream class_file(FLAGS_class_name_path.c_str());
      if (!class_file.is_open()) {
          LOG(ERROR) << "can not open: " << FLAGS_class_name_path;
      }
      std::string label_name;
      int label;
      while (class_file >> label_name >> label) {
          label2name.insert(std::pair<int, std::string>(label, label_name));
      }
      class_file.close();
  }

  caffe::NetParameter net_param;
  caffe::ReadNetParamsFromTextFileOrDie(FLAGS_model, &net_param);  

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  std::string list_str = FLAGS_list;
  std::ifstream list_file(list_str.c_str());
  if (!list_file.is_open()) {
     LOG(ERROR) << "cannot open: " << list_str;
  }
  std::string tmp_str;
  std::vector<std::string> list_vec;
  while (list_file >> tmp_str) {
      list_vec.push_back(tmp_str);
  }
  list_file.close();
  int mini_batch= net_param.layer(0).memory_data_param().batch_size(); 

  int iterations = (list_vec.size()) / mini_batch + 1;
  

  LOG(INFO) << "Running for " << iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  int img_ind = 0;
  //float loss = 0;
  //
  int label_num = net_param.layer(net_param.layer().size() - 1).detection_param().label_num();
  img_ind = 0;
  for (int i = 0; i < iterations; ++i) {
  //for (int i = 5; i < 6; ++i) {
    //float iter_loss;
    
    // fill memory data layer
    std::vector<cv::Mat> img_vec;
    std::vector<cv::Mat> resize_img_vec;
    std::vector<int> fake_labels;
    std::vector<std::string> img_names;
    for (int j = 0; j < mini_batch; j++) {
        std::string img_name = FLAGS_img_path + "/" +
            list_vec[img_ind % list_vec.size()] + ".jpg";
        cv::Mat img = cv::imread(img_name);
        if (img.data == NULL) {
            LOG(ERROR) << "can't open image: " << img_name;
        }
        
        cv::Mat resize_img;
        cv::resize(img, resize_img, cv::Size(300, 300));
        resize_img_vec.push_back(resize_img);
        img_vec.push_back(img);
        fake_labels.push_back(0);
        img_names.push_back(list_vec[img_ind % list_vec.size()]);
        img_ind++;
    }

    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >
        (caffe_net.layers()[0])->AddMatVector(resize_img_vec, fake_labels);


    float loss = 0;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&loss);
    LOG(INFO) << "top size" << result.size();
    //loss += iter_loss;
    int idx = 0;
    LOG(INFO) << result[0]->count();
    Blob<float>* p_res = result[1];

    std::vector<std::vector<Res> > res_vec(mini_batch);

    if (p_res->count() > 1) {
        //LOG(INFO) << p_res->height() << " res num";

        const float* res_data = p_res->cpu_data();
        int res_data_ind = 0;

        for (int j = 0; j < p_res->height(); j++) {
            int img_id = res_data[res_data_ind++];
            int label = res_data[res_data_ind++];

            /*
            float center_x = res_data[res_data_ind++];
            float center_y = res_data[res_data_ind++];
            float w = res_data[res_data_ind++];
            float h = res_data[res_data_ind++];
            
            float xmin = center_x - 0.5 * w;
            float ymin = center_x - 0.5 * h;
            float xmax = center_x + 0.5 * w;
            float ymax = center_y + 0.5 * h;
            */
            float xmin = res_data[res_data_ind++];
            float ymin = res_data[res_data_ind++];
            float xmax = res_data[res_data_ind++];
            float ymax = res_data[res_data_ind++];
            

            std::vector<float> confs(label_num, 0.0);
            for (int k = 0; k < label_num; k++) {
                confs[k] = res_data[res_data_ind++];
            }
            Res res;
            res.label = label;
            res.xmin = xmin;
            res.xmax = xmax;
            res.ymin = ymin;
            res.ymax = ymax;
            res.confs = confs;

            res_vec[img_id].push_back(res);
        }
    }
    for (int j = 0; j < mini_batch; j++) {
        //img_vec[j] = resize_img_vec[j];
        if (FLAGS_img_save_path.size() > 0)
        {
            for (int k = 0; k < res_vec[j].size(); k++) {
                
                float xmin = res_vec[j][k].xmin * (float)(img_vec[j].cols);
                float ymin = res_vec[j][k].ymin * (float)(img_vec[j].rows);
                float xmax = res_vec[j][k].xmax * (float)(img_vec[j].cols);
                float ymax = res_vec[j][k].ymax * (float)(img_vec[j].rows);
                int label = res_vec[j][k].label;
                float max_conf = res_vec[j][k].confs[label];

                if (max_conf > 0.6) {
                    cv::rectangle(img_vec[j], 
                            cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1),
                            cv::Scalar(255, 0, 0), 2);
                }
                else {
                    cv::rectangle(img_vec[j], 
                            cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1),
                            cv::Scalar(0, 0, 255));
                }
                char write_str[255];
                if (label2name.size() > 0) {
                    snprintf(write_str, 255, "%s:%f", label2name[label].c_str(), max_conf);
                }
                else {
                    snprintf(write_str, 255, "%d:%f", label, max_conf);
                }
                std::string str(write_str);
                if (max_conf > 0.6) {
                    cv::putText(img_vec[j], str, cv::Point(xmin, ymin),CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0));
                }
                else {
                    cv::putText(img_vec[j], str, cv::Point(xmin, ymin),CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
                }
            }

            std::string save_name = FLAGS_img_save_path + "/" + img_names[j] + ".jpg";
            cv::imwrite(save_name, img_vec[j]);
                LOG(INFO) << "save ok" << save_name;
        }
    }
  }


  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}

