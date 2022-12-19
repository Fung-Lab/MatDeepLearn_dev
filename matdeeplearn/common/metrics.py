import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

root_path = '/global/cfs/projectdirs/m3641/Sidharth/MatDeepLearn_dev/output/'

class MetricMonitor:
    '''
    Monitor and update various training metrics.
    '''
    def __init__(self, plotpath=os.path.join(root_path, 'plots'), datapath=os.path.join(root_path, 'data'), epoch_step=5, start_epoch=0) -> None:
        self.streams = {}
        self.epoch = {}
        self.start_epoch = start_epoch
        self.epoch_step = epoch_step
        self.plotpath = plotpath
        self.datapath = datapath
    
    def create_data_stream(self, stream_name):
        '''
        Create a metric data stream to be updated.
        '''
        self.streams[stream_name] = []
        self.epoch[stream_name] = [self.start_epoch]
    
    def update(self, stream_name, val):
        '''
        Update a data stream with a specified value.
        '''
        if stream_name in self.streams:
            self.streams[stream_name].append(val)             
            self.epoch[stream_name].append(self.epoch[stream_name][len(self.epoch[stream_name]) - 1] + self.epoch_step)
    
    def save_outputs(self):
        '''
        Save post-training metric outputs.
        '''
        timestamp = datetime.now()
            
        metric_df = pd.DataFrame(self.streams)
        print(metric_df)
        print(os.path.join(self.datapath, f'train_metric_{timestamp}.csv'))
        metric_df.to_csv(os.path.join(self.datapath, f'train_metric_{timestamp}.csv'))
        
        fig, axs = plt.subplots(len(self.streams.keys()))
        fig.suptitle(f'Training metrics {timestamp}')
        
        for ax, item in zip(axs, self.streams.items()):
            ax.plot(self.epoch[item[0]][:-1], item[1])
            ax.set_title(item[0])
            
        print(os.path.join(self.plotpath, f'plot_metrics_{timestamp}.png'))
        plt.savefig(os.path.join(self.plotpath, f'plot_metrics_{timestamp}.png'))

class DatasetMetrics:
    '''
    Analyze a graph dataset for basic properties
    and create basic visualization of overall statistics.
    '''
    def __init__(self) -> None:
        pass

class VisualizeGraph:
    '''
    Visualize input and latent space graphs with heatmap plots.
    TODO: Look at old MatDeepLearn to port over latent visualization code.
    ''' 
    def __init__(self) -> None:
        pass

# Testing code
if __name__ == '__main__':    
    print('Testing metrics')
    
    m = MetricMonitor()
    m.create_data_stream('test1')
    m.create_data_stream('test2')
    
    for i in range(100):
        m.update('test1', i)
        m.update('test2', i)
        
    m.save_outputs()
