from Blacklitterman import black_litterman_optimization
from MaxSharpe import max_sharpe_optimization,get_historical_prices
from collections import OrderedDict
import pprint
import csv


energy_list = ['XOM', 'CVX', 'COP', 'HES', 'OXY', 'EOG', 'MPC', 'PSX', 'VLO', 'ENB', 
 'SU', 'CNQ', 'CVE', 'IMO', 'BP', 'EQNR', 'IBE', 'WMB', 'OKE', 'ET', 
 'HOU', 'CQP', 'BKR', 'TRGP', 'TGTX', 'AR', 'VLO', 'KMI', 'PPL', 'DUK', 'XEL']

consumer_list = ["PG", "KO", "PEP", "JNJ", "CL", "UL", "GIS", "KMB", "MDLZ", "COLM", "EL", "MCD", 
 "PM", "HSY", "CAG", "LRLCF", "KHC", "OR", "DGE", "IMB", "RB", "BN", "7203.T", 
 "005930.KS", "066570.KS", "CX", "BABA", "AMZN", "TGT", "WMT", "COST", "TJX", 
 "LOW", "NKE", "SBUX", "K"]

industrial_materials = ["LIN", "NEM", "SCCO", "RIO", "VMC", "APD", "DD", "VALE", "XOM", "LMT", 
 "MMM", "CAT", "DE", "GE", "CSX", "ECL", "DOW", "FCX", "FLS", "JCI", "AME", 
 "IP", "PKG", "EMR", "X", "GWW", "CNI", "ITW", "HON", "IEX", "RS", "STLD", 
 "NUE", "VLO", "PHM", "TMO"]

INDUSTRY = 'IndustrialMaterials'
ticker_list = industrial_materials


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
