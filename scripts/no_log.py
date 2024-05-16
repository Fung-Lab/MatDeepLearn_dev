import logging
import pprint
import os
import sys
import shutil
from datetime import datetime
from torch import distributed as dist
from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data
import time

import torch

# import submitit

# from matdeeplearn.common.utils import setup_logging


class Runner:  # submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        
        with new_trainer_context(args=args, config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer           

            self.task.setup(self.trainer)

            # Print settings for job
            logging.debug("Settings: ")
            logging.debug(pprint.pformat(self.config))

            tic = time.perf_counter() # time.perf_counter() / time.process_time()
            tic_ptime = time.process_time()
            
            torch.cuda.reset_peak_memory_stats(device=None)
            self.task.run()
            print(f"gpu used {torch.cuda.max_memory_allocated(device=None) * 1e-9} GB")
            
            toc = time.perf_counter()
            toc_ptime = time.process_time()
            logging.debug(f"Time taken (perf_counter): {toc - tic}")
            logging.debug(f"Time taken (process_time): {toc_ptime - tic_ptime}")
            
            # shutil.move('log_'+config["task"]["log_id"]+'.txt', os.path.join(self.trainer.save_dir, "results", self.trainer.timestamp_id, "log.txt"))

    def checkpoint(self, *args, **kwargs):
        # new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        # return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":


    # setup_logging()
    # local_rank = os.environ.get('LOCAL_RANK', None)
    # print("Local Rank: ", local_rank)
    # if local_rank == None or int(local_rank) == 0:
    #     root_logger = logging.getLogger()
    #     root_logger.setLevel(logging.DEBUG)
        
    #     timestamp = datetime.now().timestamp()
    #     timestamp_id = datetime.fromtimestamp(timestamp).strftime(
    #         "%Y-%m-%d-%H-%M-%S-%f"
    #     )[:-3]    
    #     fh = logging.FileHandler('log_'+timestamp_id+'.txt', 'w+')        
    #     fh.setLevel(logging.DEBUG)  
    #     root_logger.addHandler(fh)  
                
    #     sh = logging.StreamHandler(sys.stdout)
    #     sh.setLevel(logging.DEBUG)                               
    #     root_logger.addHandler(sh)
    # else:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)                               
    root_logger.addHandler(sh)

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    # if local_rank == None or int(local_rank) == 0:
    #     config["task"]["log_id"] = timestamp_id
    # else:
    
    if not config["dataset"]["processed"]:
        process_data(config["dataset"])

    if args.submit:  # Run on cluster
        # TODO: add setup to submit to cluster
        pass

    else:  # Run locally
        Runner()(config)
      
