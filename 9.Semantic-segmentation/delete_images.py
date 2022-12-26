import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

def delete_images(read_path, write_path):
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    
    e_filenames = []
    for (dirpath, _, filenames) in os.walk(read_path):
        e_filenames.extend(map(lambda x: os.path.join(dirpath, x), filenames))
        
    for n, e in enumerate(e_filenames):
        w = SummaryWriter(log_dir=os.path.join(write_path, str(n)))
        e_acc = EventAccumulator(e)
        e_acc.Reload()
        for s_name in e_acc.scalars.Keys():
            for s_event in e_acc.Scalars(s_name):
                w.add_scalar(s_name, s_event.value, s_event.step, s_event.wall_time)
        w.flush()
        w.close()