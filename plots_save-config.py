from PlotsFunctions import make_plot,make_stacked_plot
import pandas as pd
import seaborn as sns
import os

file = 'csv_files/testing for iou loss the best model.csv'
df = pd.read_csv(file)


    
for i in range(1,len(df.columns)):
    if not df.columns[i].startswith('val'):
        
        make_plot(df, df.columns[i], df.columns[i])
  

    




