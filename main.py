from Blacklitterman import black_litterman_optimization
from MaxSharpe import max_sharpe_optimization,get_historical_prices
from collections import OrderedDict
import pprint
import csv

INDUSTRY = 'CHANGEME'
ticker_list = ["PG", "KO", "PEP", "JNJ", "CL", "UL", "GIS", "KMB", "MDLZ", "COLM", "EL", "MCD",
                   "PM", "HSY", "CAG", "LRLCF", "KHC", "OR", "DGE", "IMB", "RB", "BN", "7203.T", 
                   "005930.KS", "066570.KS", "CX", "BABA", "AMZN", "TGT", "WMT", "COST", "TJX", "LOW", 
                   "NKE", "SBUX", "K"]

print("ORDERED LIST")
max_sharpe = max_sharpe_optimization(get_historical_prices(ticker_list),ticker_list)
sorted_dict = dict(max_sharpe.items())
sorted_by_values = dict(sorted(sorted_dict.items(), key=lambda item: item[1], reverse=True))
pprint.pprint(sorted_by_values,sort_dicts=False)

def write_dict_to_csv(data_dict, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(["Stock", "Weight"])
        
        # Write data
        for key, value in data_dict.items():
            writer.writerow([key, value])
    
    print(f"CSV file '{filename}' has been created.")

write_dict_to_csv(sorted_by_values, f'{INDUSTRY}.csv')
