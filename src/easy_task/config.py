from collections import namedtuple

mfcc_configs=namedtuple(typename="mfcc",field_names=['sample_rate','duration','n_mfcc'])(sample_rate=22050,duration=30.0,n_mfcc=20)

raw_data_path= "./data"


