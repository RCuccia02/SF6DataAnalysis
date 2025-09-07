from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator

log_file = r'lightning_logs/version_51/events.out.tfevents.1757166022.DESKTOP-R3KV438.19120.0'

ea = event_accumulator.EventAccumulator(log_file)
ea.Reload()

# stampa tutte le metriche scalar disponibili
print(ea.Tags()['scalars'])
