#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_SIMRUNNER_H_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_SIMRUNNER_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <map>
#include <vector>

#include "flamegpu/defines.h"
#include "flamegpu/simulation/LogFrame.h"
#include "flamegpu/simulation/detail/AbstractSimRunner.h"

namespace flamegpu {
struct ModelData;
class LoggingConfig;
class StepLoggingConfig;
class RunPlanVector;
class CUDAEnsemble;
namespace detail {

/**
 * A thread class which executes RunPlans on a single GPU
 *
 * This class is used by CUDAEnsemble, it creates one SimRunner instance per GPU, each executes in a separate thread.
 * There may be multiple instances per GPU, if running small models on large GPUs.
 */
class SimRunner : public AbstractSimRunner {
    /**
    * Flag for whether the ensemble should throw an exception if it errors out
    */
    const bool fail_fast;

 public:
    /**
     * Constructor, creates and initialise a new SimRunner
     * @param _model A copy of the ModelDescription hierarchy for the RunPlanVector, this is used to create the CUDASimulation instances.
     * @param _err_ct Reference to an atomic integer for tracking how many errors have occurred
     * @param _next_run Atomic counter for safely selecting the next run plan to execute across multiple threads
     * @param _plans The vector of run plans to be executed by the ensemble
     * @param _step_log_config The config of which data should be logged each step
     * @param _exit_log_config The config of which data should be logged at run exit
     * @param _device_id The GPU that all runs should execute on
     * @param _runner_id A unique index assigned to the runner
     * @param _verbosity Verbosity level (Verbosity::Quiet, Verbosity::Default, Verbosity::Verbose)
     * @param _fail_fast If true, the SimRunner will kill other runners and throw an exception on error
     * @param run_logs Reference to the vector to store generate run logs
     * @param log_export_queue The queue of logs to exported to disk
     * @param log_export_queue_mutex This mutex must be locked to access log_export_queue
     * @param log_export_queue_cdn The condition is notified every time a log has been added to the queue
     * @param err_detail Structure to store error details on fast failure for main thread rethrow
     * @param _total_runners Total number of runners executing
     * @param _isSWIG Flag denoting whether it's a Python build of FLAMEGPU
     */
    SimRunner(const std::shared_ptr<const ModelData> _model,
        std::atomic<unsigned int> &_err_ct,
        std::atomic<unsigned int> &_next_run,
        const RunPlanVector &_plans,
        std::shared_ptr<const StepLoggingConfig> _step_log_config,
        std::shared_ptr<const LoggingConfig> _exit_log_config,
        int _device_id,
        unsigned int _runner_id,
        flamegpu::Verbosity _verbosity,
        bool _fail_fast,
        std::map<unsigned int, RunLog> &run_logs,
        std::queue<unsigned int> &log_export_queue,
        std::mutex &log_export_queue_mutex,
        std::condition_variable &log_export_queue_cdn,
        std::vector<ErrorDetail> &err_detail,
        unsigned int _total_runners,
        bool _isSWIG);
    /**
     * SimRunner loop with shared next_run atomic
     */
    void main() override;
};

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_SIMRUNNER_H_
