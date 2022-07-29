from tardis_dev import datasets, get_exchange_details
import os
from joblib import Parallel,delayed
from multiprocessing import cpu_count

# function used by default if not provided via options
def default_file_name(exchange, data_type, date, symbol, format):
    return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# customized get filename function - saves data in nested directory structure
def file_name_nested(exchange, data_type, date, symbol, format):
    return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"

def get_options(symbol,start,path):
    datasets.download(
        # one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
        exchange="deribit",
        # accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit,
        # or get those values from 'deribit_details["datasets"]["symbols][]["dataTypes"] dict above
        data_types=["incremental_book_L2", "trades", "quotes", "derivative_ticker", "book_snapshot_25",
                    "book_snapshot_5", "liquidations"],
        # change date ranges as needed to fetch full month or year for example
        from_date=start,
        # to date is non inclusive
        # to_date=end,
        # accepted values: 'datasets.symbols[].id' field in https://api.tardis.dev/v1/exchanges/deribit
        symbols=symbol,
        # (optional) your API key to get access to non sample data as well
        api_key="YOUR API KEY",
        # (optional) path where data will be downloaded into, default dir is './datasets'
        download_dir=path,
        # (optional) - one can customize downloaded file name/path (flat dir strucure, or nested etc) - by default function 'default_file_name' is used
        # get_filename=default_file_name,
        # (optional) file_name_nested will download data to nested directory structure (split by exchange and data type)
        # get_filename=file_name_nested,
    )

# 下载数据
deribit_details = get_exchange_details("deribit")

all_symbols = deribit_details['availableSymbols']

options_list = []
for symbol in all_symbols:
    if (symbol['type'] == 'option'):
        options_list.append(symbol)
    else:
        pass

path = ''
if not os.path.exists(path):
    os.mkdir(path)

n = cpu_count()
Parallel(n_jobs = n-1)(delayed(get_options)(i['id'],i['availableSince'],path)for i in options_list)