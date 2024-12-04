import warnings
import pandas as pd

#Setting up pandas display settings
pd.options.mode.copy_on_write = True
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.3f}'.format
warnings.filterwarnings('ignore')