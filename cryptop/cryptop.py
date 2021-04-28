import curses
import os
import sys
import re
import time
import hmac
import hashlib
try:
    from urllib import urlencode
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urlencode
    from urllib.parse import urljoin
import requests
import shutil
import ntpath
import configparser
import json
import pkg_resources
import locale
import datetime
import requests
import requests_cache
import http.client
import json
import _thread
from datetime import date, timedelta

import signal
import faulthandler
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)

# GLOBALS!
BASEDIR = os.path.join(os.path.expanduser('~'), '.cryptop')
WALLETFILE = os.path.join(BASEDIR, 'wallet.json')
CONFFILE = os.path.join(BASEDIR, 'config.ini')
LOGFILE = os.path.join(BASEDIR, 'log' if len(sys.argv) == 1 else sys.argv[1] + '.log')
LOGTIME = 0
CONFIG = configparser.ConfigParser()
COIN_FORMAT = re.compile('[A-Z]{2,5},\d{0,}\.?\d{0,}')

SORT_FNS = { 'coin' : lambda item: item[0],
             'price': lambda item: float(item[1][0]),
             'held' : lambda item: float(item[2]),
             'val'  : lambda item: float(item[1][0]) * float(item[2]) }
SORTS = list(SORT_FNS.keys())
COLUMN = SORTS.index('val')
ORDER = True

ethplorer_conn = http.client.HTTPSConnection("api.ethplorer.io")

isfiat = lambda c: c.upper() in ['EUR', 'USD']
BLACKLIST = []
CURRENCYLIST = [ 'USD', 'ETH', 'BTC' ]
CURRENCY = 'USD'
SYMBOL = '$'
SYMBOLMAP = {
'USD' : '$',
'EUR' : '€',
'ETH' : 'Ξ',
'BTC' : 'Ƀ'
}
CURRENCYCOUNTER = 0
NROFDECIMALS = 2
FIELD = 0
FIELD_OFFSET = 0
BALANCE_TIME = 0
SHOW_BALANCES = 1

KEY_ESCAPE = 27
KEY_ENTER = 13
KEY_BACKSPACE = 8
KEY_SPACE = 32
KEY_ZERO = 48
KEY_A = 65
KEY_F = 70
KEY_H = 72
KEY_Q = 81
KEY_R = 82
KEY_S = 83
KEY_C = 67
KEY_T = 84
KEY_V = 86
KEY_a = 97
KEY_f = 102
KEY_h = 104
KEY_q = 113
KEY_r = 114
KEY_s = 115
KEY_c = 99
KEY_t = 116
KEY_v = 118

def log(*args, **kwargs):
  global LOGFILE
  date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
  print(date, *args, **kwargs, file=LOGFILE)
  LOGFILE.flush()

def read_configuration(confpath):
  # copy our default config file
  if not os.path.isfile(confpath):
    defaultconf = pkg_resources.resource_filename(__name__, 'config.ini')
    shutil.copyfile(defaultconf, CONFFILE)
  CONFIG.read(confpath)
  return CONFIG

LOGFILE = open(LOGFILE, 'w')
log('cryptop starting up')
CONFIG = read_configuration(CONFFILE)
rget = lambda url:requests.get(url,timeout=7).json()
coinstats = {}

clist = []
try:
  CCLIST = rget('https://min-api.cryptocompare.com/data/blockchain/list?api_key='+CONFIG['keys'].get('cryptocompare', ''))['Data']
  #CCLIST = rget('https://min-api.cryptocompare.com/data/blockchain/list?api_key=')['Data']
except:
  log("error: clist | cryptocompare")
  CCLIST = {}
try:
  CGMAP = {x['symbol'].upper() : x['id'] for x in rget('https://api.coingecko.com/api/v3/coins/list')[::-1]}
  CGMAP['UNI'] = 'uniswap' # HACK
except:
  log("error: cgmap | coingecko")
  raise
CCLIST = set(CCLIST.keys())
CCSET = set([])

def update_coins():
  global CURRENCYLIST
  wallet = read_wallet()
  #CCSET = set(wallet.keys()) - CCLIST
  stats = {}
  cmclist = set([])
  for coin in [ c for c in CURRENCYLIST if not isfiat(c) ]:
    ret = {}
    try:
      ret = rget('https://min-api.cryptocompare.com/data/histohour?fsym=%s&tsym=USD&toTs=%d&limit=175&api_key=%s' %
        (coin.upper(),int(time.time()),CONFIG['keys'].get('cryptocompare', '')))
    except:
      log("error: update_coins | cryptocompare: " + str(ret))
      continue
    if ret['Data']:
      if not coin in stats.keys():
        stats[coin] = {}
      stats[coin]['price_usd'] = ret['Data'][-1]['close']
      stats[coin]['percent_change_1h'] = 100. - 100. * (ret['Data'][-2]['close'] / ret['Data'][-1]['close'])
      stats[coin]['percent_change_24h'] = 100. - 100. * (ret['Data'][-25]['close'] / ret['Data'][-1]['close'])
      stats[coin]['percent_change_7d'] = 100. - 100. * (ret['Data'][-169]['close'] / ret['Data'][-1]['close'])
    else:
      log("error: update_coins | cryptocompare: " + str(ret))

  #cmc = http.client.HTTPSConnection("api.coinmarketcap.com")
  #clist = []
  #try:
  #  cmc.request("GET", '/v1/ticker/?convert=EUR&limit=2000', {}, {})
  #  data = cmc.getresponse()
  #  data = json.loads(data.read().decode())
  #except:
  #  log("error: update_coins | coinmarketcap: " + str(data))
  #  return
  #for item in data[::-1]:
  #  if item['symbol'] in coinmap.keys() and coinmap[item['symbol']] != item['id']:
  #    continue
  #  if item['symbol'] in CCLIST:
  #    clist.append(item['symbol'])
  #  if item['symbol'] in CURRENCYLIST and not isfiat(item['symbol']) and not item['symbol'] in cmclist:
  #    stats[item['symbol']]['24h_volume_usd'] = item['24h_volume_usd']
  #  else:
  #    stats[item['symbol']] = {}
  #    for key in ['price_usd', '24h_volume_usd', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d']:
  #      stats[item['symbol']][key] = float(item[key] or 0)

  for fiat in [ f for f in CURRENCYLIST if isfiat(f) and f != 'USD']:
    if not fiat in stats.keys():
      stats[fiat] = {'.' : 0}
    try:
      stats[fiat]['price_usd'] = 1. / rget('https://api.exchangeratesapi.io/latest?base=USD')['rates']['EUR']
      stats[fiat]['percent_change_24h'] = 0
      stats[fiat]['percent_change_1h'] = 0
      stats[fiat]['percent_change_7d'] = 0
    except:
      try:
        #if time.time() - stats[fiat]['.'] > 30 * 60:
        rates = rget('https://www.quandl.com/api/v3/datasets/ECB/EURUSD?api_key='+CONFIG['keys'].get('quandl', ''))['dataset']['data']
        #stats[fiat]['.'] = time.time()
      except:
        continue
      stats[fiat]['price_usd'] = rates[0][1]
      stats[fiat]['percent_change_24h'] = 100. - 100. * rates[1][1] / rates[0][1]
      stats[fiat]['percent_change_1h'] = stats[fiat]['percent_change_24h'] / 24.
      stats[fiat]['percent_change_7d'] = 100. - 100. * rates[7][1] / rates[0][1]

  for tok in CCSET:
    if tok.upper() in CGMAP:
      try:
        ret = rget('https://api.coingecko.com/api/v3/coins/' + CGMAP[tok.upper()])
        if 'usd' not in ret['market_data']['total_volume'] or ret['market_data']['total_volume']['usd'] == 0:
          continue
        if not tok in stats:
          stats[tok] = {}
          for d in ['1h','24h','7d']:
            stats[tok]['percent_change_' + d] = 0
        stats[tok]['price_usd'] = ret['market_data']['current_price']['usd']
        stats[tok]['24h_volume_usd'] = ret['market_data']['total_volume']['usd']
        if stats[tok]['24h_volume_usd'] == 0:
          for alt in ['eur','btc','eth']:
            if alt in ret['market_data']['total_volume']:
              stats[tok]['24h_volume_usd'] = ret['market_data']['total_volume'][alt] * stats[alt.upper()]['price_usd']
              break
        for d in ['1h','24h','7d']:
          if 'usd' in ret['market_data']['price_change_percentage_%s_in_currency'%d]:
            stats[tok]['percent_change_' + d] = ret['market_data']['price_change_percentage_%s_in_currency'%d]['usd']
          else:
            for alt in ['eur','btc','eth']:
              if alt in ret['market_data']['price_change_percentage_%s_in_currency'%d]:
                stats[tok]['percent_change_' + d] = ret['market_data']['price_change_percentage_%s_in_currency'%d][alt] * stats[alt.upper()]['price_usd']
                break
      except Exception as e:
        log("error: update_coins | coingecko:", str(e))
        continue

  if CCSET:
    try:
      ret = rget('https://min-api.cryptocompare.com/data/pricemultifull?fsyms=%s&tsyms=USD&api_key=%s' %
        (','.join(list(CCSET)), CONFIG['keys'].get('cryptocompare', '')))
    except:
      pass
    if 'RAW' in ret:
      for tok in ret['RAW']:
        if tok in stats.keys() and not isfiat(tok):
          rates = [ stats[tok]['price_usd'],
          stats[tok]['price_usd'] * (1. - stats[tok]['percent_change_1h'] / 100.),
          stats[tok]['price_usd'] * (1. - stats[tok]['percent_change_24h'] / 100.),
          stats[tok]['price_usd'] * (1. - stats[tok]['percent_change_7d'] / 100.) ]
          prev = stats[tok]['price_usd']
          ratio = float(ret['RAW'][tok]['USD']['PRICE']) / prev
          if ratio > 0.75 and ratio < 1.5:
            stats[tok]['price_usd'] = 0.25 * float(ret['RAW'][tok]['USD']['PRICE']) + 0.75 * prev
            rates = [ r + (stats[tok]['price_usd'] - prev) * r / prev for r in rates ]
            stats[tok]['percent_change_1h'] = 100. - 100. * rates[1] / rates[0]
            stats[tok]['percent_change_24h'] = 100. - 100. * rates[2] / rates[0]
            stats[tok]['percent_change_7d'] = 100. - 100. * rates[3] / rates[0]
        else:
          stats[tok] = {}
          stats[tok]['price_usd'] = ret['RAW'][tok]['USD']['PRICE']
          stats[tok]['percent_change_1h'] = ret['RAW'][tok]['USD']['CHANGEPCTHOUR']
          stats[tok]['percent_change_24h'] = ret['RAW'][tok]['USD']['CHANGEPCT24HOUR']
          stats[tok]['percent_change_7d'] = 0
          stats[tok]['24h_volume_usd'] = ret['RAW'][tok]['USD']['TOTALVOLUME24HTO']

  if not 'ETH' in stats.keys():
    return

  for tok in stats:
    for k in ['24h_volume_usd']:
      if not k in stats[tok]:
        stats[tok][k] = 0

  global coinstats
  coinstats = stats
  return

  try:
    ret = rget('https://bittrex.com/api/v1.1/public/getmarketsummaries')
  except:
    pass
  else:
    for pair in ret['result'] or []:
      if pair['MarketName'].split('-')[0] == 'ETH' and pair['Last'] is not None:
        tok = pair['MarketName'].split('-')[1].replace('BCC','BCH')
        if tok in stats.keys() and not isfiat(tok):
          rates = [ stats[tok]['price_usd'],
          stats[tok]['price_usd'] * (1. - stats[tok]['percent_change_1h'] / 100.),
          stats[tok]['price_usd'] * (1. - stats[tok]['percent_change_24h'] / 100.),
          stats[tok]['price_usd'] * (1. - stats[tok]['percent_change_7d'] / 100.) ]
          price = stats[tok]['price_usd'] / stats['ETH']['price_usd']
          prev = stats[tok]['price_usd']
          if prev == 0:
            prev = price if price > 0 else 1
          stats[tok]['price_usd'] = (0.75 * float(pair['Last']) + 0.25 * price) * stats['ETH']['price_usd']
          rates = [ r + (stats[tok]['price_usd'] - prev) * r / prev for r in rates ]
          stats[tok]['percent_change_1h'] = 100. - 100. * rates[1] / rates[0]
          stats[tok]['percent_change_24h'] = 100. - 100. * rates[2] / rates[0]
          stats[tok]['percent_change_7d'] = 100. - 100. * rates[3] / rates[0]

  try:
    ret = rget('https://www.binance.com/api/v1/ticker/allPrices')
  except:
    pass
  else:
    if isinstance(ret,list):
      for pair in ret:
        if pair['symbol'][-3:] == 'ETH':
          tok = pair['symbol'][:-3].replace('BCC','BCH')
          if isfiat(tok): continue
          if tok in list(stats.keys()):
            rates = [ stats[tok]['price_usd'],
            stats[tok]['price_usd'] * (1. - stats[tok]['percent_change_1h'] / 100.),
            stats[tok]['price_usd'] * (1. - stats[tok]['percent_change_24h'] / 100.),
            stats[tok]['price_usd'] * (1. - stats[tok]['percent_change_7d'] / 100.) ]
            price = stats[tok]['price_usd'] / stats['ETH']['price_usd']
            prev = stats[tok]['price_usd']
            if prev == 0:
              prev = price if price > 0 else 1
            stats[tok]['price_usd'] = (0.75 * float(pair['price']) + 0.25 * price) * stats['ETH']['price_usd']
            rates = [ r + (stats[tok]['price_usd'] - prev) * r / prev for r in rates ]
            if rates[0]:
              stats[tok]['percent_change_1h'] = 100. - 100. * rates[1] / rates[0]
              stats[tok]['percent_change_24h'] = 100. - 100. * rates[2] / rates[0]
              stats[tok]['percent_change_7d'] = 100. - 100. * rates[3] / rates[0]
          elif 'ETH' in stats.keys():
            stats[tok] = {}
            stats[tok]['price_usd'] = float(pair['price']) * stats['ETH']['price_usd']
            stats[tok]['percent_change_1h'] = 0
            stats[tok]['percent_change_24h'] = 0
            stats[tok]['percent_change_7d'] = 0
            stats[tok]['24h_volume_usd'] = 0
        elif pair['symbol'][-4:] == 'USDT' and False:
          tok = pair['symbol'][:-4].replace('BCC','BCH')
          if isfiat(tok): continue
          if tok in list(stats.keys()):
            stats[tok]['price_usd'] = float(pair['price'])

  for tok in stats:
    for k in ['24h_volume_usd']:
      if not k in stats[tok]:
        stats[tok][k] = 0

  #global coinstats
  coinstats = stats

def ticker():
  import time
  ts = 0
  while True:
    #time.sleep(max(min(int(time.time()) - ts, 20), 5))
    #time.sleep(30)
    time.sleep(2 * 25 * (len(CURRENCYLIST)-1))
    #ts = int(time.time())
    update_coins()

bittrex_tokens = {}
def update_bittrex(key, secret):
  try:
    url = "https://bittrex.com/api/v1.1/account/getbalances?apikey=%s&nonce=%d&" % (key, int(time.time() * 1000))
    ret = requests.get(url,
        headers={"apisign": hmac.new(secret.encode(), url.encode(), hashlib.sha512).hexdigest() },
        timeout=5).json()
    if 'result' in ret and ret['result'] is not None:
      bittrex_tokens[key] = ret
  except:
    if not key in bittrex_tokens.keys():
      bittrex_tokens[key] = { 'result' : [] }
  return bittrex_tokens[key]

bittrex_time = 0
def bittrex(key, secret):
  global bittrex_time
  if not key in bittrex_tokens.keys():
    return update_bittrex(key, secret)
  if time.time() - bittrex_time > 20:
    bittrex_time = time.time()
    _thread.start_new_thread(update_bittrex, (key,secret))
  return bittrex_tokens[key]

binance_tokens = {}
def update_binance(key, secret):
  try:
    url = "https://api.binance.com/api/v3/account?timestamp=%d" % int(time.time() * 1000)
    url += '&signature=' + hmac.new(secret.encode('utf-8'), url.split('?')[1].encode('utf-8'), hashlib.sha256).hexdigest()
    ret = requests.get(url,
        headers={'X-MBX-APIKEY': key, 'Accept': 'application/json', 'User-Agent': 'binance/python'},
        timeout=5).json()
    if 'balances' in ret and ret['balances'] is not None:
      binance_tokens[key] = ret
    else:
      assert False, str(ret)
  except:
    if not key in binance_tokens.keys():
      binance_tokens[key] = { 'balances' : [] }
  return binance_tokens[key]

binance_time = 0
def binance(key, secret):
  global binance_time
  if not key in binance_tokens.keys():
    return update_binance(key, secret)
  if time.time() - binance_time > 20:
    binance_time = time.time()
    _thread.start_new_thread(update_binance, (key,secret))
  return binance_tokens[key]

erc20_block = {}
erc20_contracts = set([])
erc20 = {}

def er(method, default):
  global etherscan_conn
  try:
    etherscan_conn.request("GET", method, {}, {})
    data = etherscan_conn.getresponse()
    data = json.loads(data.read().decode())
    if 'result' in data.keys():
      return data['result']
    return default
  except:
    return default

etherscan_conn = None
erc20_balance = {}
def get_erc20_balance(token, address):
  import json

  global etherscan_conn
  if etherscan_conn is None:
    etherscan_conn = http.client.HTTPSConnection("etherscan.io")

  tokens = {}
  if not address in tokens:
    tokens[address] = {}
  if not token in tokens[address]:
    tokens[address][token] = { 'balance' : 0, 'eth_balance' : 0, 'time' : 0 }

  if time.time() - tokens[address][token]['time'] > 180: #60:
    tokens[address][token]['time'] = time.time()
    try:
      if token.lower() ==  'eth':
        data = rget("https://etherscan.io/api?module=account&action=balance&address=%s&tag=latest&apikey=%s" % (address, CONFIG['keys'].get('etherscan', '')))
        if 'result' in data and data['result']:
          tokens[address][token]['balance'] = float(data['result']) / 1e18
          tokens[address]['token']['eth_balance'] = float(data['result']) / 1e18
      elif False:
        etherscan_conn.request("GET", "/tokens?q="+token.lower(), {}, {})
        res = etherscan_conn.getresponse()
        data = res.read()
        data = str(data)

        end = 0
        nth = 1
        for i in range(nth):
          start = data.find('/token/0x', end) + len('/token/')
          end = data.find('>',start) - 2
        contract = data[start:end]
        r = rget("https://api.tokenbalance.com/token/%s/%s" % (contract,address))
        tokens[address][token].update(r)
    except:
      pass

  return float(tokens[address][token]['balance']),float(tokens[address][token]['eth_balance'])

tokens = {}
image = {}
def get_ethereum(address):
  import json

  global BLACKLIST, tokens, ethplorer_conn, etherscan_conn
  tinfo = {'.' : 0}
  data = {}
  apidown = False
  try:
    if not address in image:
      image[address] = {'data': None,'time':0}
    if time.time() - image[address]['time'] > 60 * 5:
      image[address]['time'] = time.time()
      image[address]['data'] = rget('https://api.ethplorer.io/getAddressInfo/%s?apiKey=%s' % (address,CONFIG['keys'].get('ethplorer', 'freekey')))
    data = image[address]['data']
    if 'error' in data and address in tokens:
      assert False
    else:
      tinfo = {'.' : time.time(), 'ETH' : data['ETH']['balance']}
      apidown = not tinfo['ETH']
  except Exception:
    try:
      data = rget("https://api.etherscan.io/api?module=account&action=balance&address=%s&tag=latest&apikey=%s" % (address, CONFIG['keys'].get('etherscan', '')))
      if 'result' in data.keys():
        try:
          tinfo['ETH'] = float(data['result']) / 1e18
        except Exception:
          apidown = True
    except Exception:
      apidown = True
  if apidown and 'tokens' in data.keys():
    if 'tokens' in data.keys():
      for tok in data['tokens']:
        if tok['tokenInfo']['symbol'] not in BLACKLIST:
          tinfo[tok['tokenInfo']['symbol']], _ = get_erc20_balance(tok['tokenInfo']['symbol'], address)
    try:
      balance = rget("https://api.etherscan.io/api?module=account&action=balance&address=%s&tag=latest&apikey=%s" % (address, CONFIG['keys'].get('etherscan', '')))
    except Exception:
      return tokens
    if 'result' in balance.keys():
      try:
        tinfo['ETH'] = float(balance['result']) / 1e18
      except:
        return tokens
  elif 'tokens' in data.keys():
    for tok in data['tokens']:
      if tok['tokenInfo']['symbol'] not in BLACKLIST:
        if True: #time.time() - tok['tokenInfo']['lastUpdated'] > 60 * 60:
          try:
            r = rget('https://api.etherscan.io/api?module=account&action=tokenbalance&contractaddress=%s&address=%s&tag=latest&apikey=%s' % (tok['tokenInfo']['address'], address, CONFIG['keys'].get('etherscan', '')))
            tinfo[tok['tokenInfo']['symbol']] = float(r['result'] or 0) / 10**int(tok['tokenInfo']['decimals'])
          except:
            tinfo[tok['tokenInfo']['symbol']] = float(tok['balance']) / 10**int(tok['tokenInfo']['decimals'])
        else:
          tinfo[tok['tokenInfo']['symbol']] = float(tok['balance']) / 10**int(tok['tokenInfo']['decimals'])
  tokens[address] = tinfo
  return tokens

def ethereum(address):
  if not address in tokens.keys():
    tokens[address] = {}
    tokens[address]['.'] = time.time()
    return get_ethereum(address)
  elif time.time() - tokens[address]['.'] > 180:
    tokens[address]['.'] = time.time()
    _thread.start_new_thread(get_ethereum, (address,))
  return tokens

btcb = {}
def get_bitcoin(address):
  binfo = {'.' : time.time(), 'balance' : 0}
  if not address in btcb:
    btcb[address] = binfo
  try:
    binfo['balance'] = float(rget('https://blockchain.info/q/addressbalance/%s' % address)) / 1e8
  except:
    log("error: get_bitcoin | blockchain.info") 
    try:
      binfo['balance'] = float(rget('https://api.blockcypher.com/v1/btc/main/addrs/%s/balance' % address)['final_balance']) / 1e8
    except:
      log("error: get_bitcoin | blockcypher")
      try:
        binfo['balance'] = float(rget('https://api.blockchair.com/bitcoin/dashboards/address/%s' % address)['data'][address]['address']['balance']) / 1e8
      except:
        log("error: get_bitcoin | blockchair")
  btcb[address] = binfo
  return btcb

def bitcoin(address):
  if not address in btcb.keys():
    return get_bitcoin(address)[address]['balance']
  elif time.time() - btcb[address]['.'] > 60:
    btcb[address]['.'] = time.time()
    _thread.start_new_thread(get_bitcoin, (address,))
  return btcb[address]['balance']

last_price = {}
def get_price_old(coin, curr=None):
  '''Get the data on coins'''
  # curr = curr or CONFIG['api'].get('currency', 'USD')
  if curr is None:
    global CURRENCY
    curr = CURRENCY
  fmt = 'https://min-api.cryptocompare.com/data/pricemultifull?fsyms={}&tsyms={}'

  if not coin in last_price.keys():
    last_price[coin] = [(0,0,0) for c in coin.split(',')]

  try:
    r = rget(fmt.format(coin, curr))
  except requests.exceptions.RequestException:
    return last_price[coin]

  try:
    data_raw = r['RAW']
    last_price[coin] = [(data_raw[c][curr]['PRICE'],
        data_raw[c][curr]['MKTCAP'] / 1e6,
        data_raw[c][curr]['CHANGEPCT24HOUR'] or 0.0) for c in coin.split(',')]
    return last_price[coin]
  except:
    return last_price[coin]

def get_price(coin, curr=None):
  if curr is None:
    global CURRENCY
    curr = CURRENCY

  res = []
  for v in coin.split(','):
    if not v in coinstats.keys():
      res.append((0,0,0,0,0))
      continue
    tok = coinstats[v]
    sf = lambda x: float(x) if x is not None else 0
    price, volume, c1h, c24h, c7d = sf(tok['price_usd']), sf(tok['24h_volume_usd']), sf(tok['percent_change_1h'])/100., sf(tok['percent_change_24h'])/100., sf(tok['percent_change_7d'])/100.
    if curr != 'USD' and 'price_usd' in coinstats[curr]:
      price /= sf(coinstats[curr]['price_usd'])
      volume /= sf(coinstats[curr]['price_usd'])
      s1h = sf(coinstats[curr]['percent_change_1h'])/100.
      try:
        s24h = sf(coinstats[curr]['percent_change_24h'])/100.
        s7d = sf(coinstats[curr]['percent_change_7d'])/100.
      except:
        s24h = sf(tok['24h_volume_usd']) / 100.
        s7d = sf(coinstats[curr]['percent_change_7d'])/100.
      conv = lambda dx,dy: 1. - (1.-dx) / (1.-dy)
      c1h,c24h,c7d = conv(c1h,s1h), conv(c24h,s24h), conv(c7d,s7d)
    res.append((price, volume, c1h * 100, c24h * 100, c7d * 100))
  return res

def get_theme_colors():
  ''' Returns curses colors according to the config'''
  def get_curses_color(name_or_value):
    try:
      return getattr(curses, 'COLOR_' + name_or_value.upper())
    except AttributeError:
      return int(name_or_value)

  theme_config = CONFIG['theme']
  return (get_curses_color(theme_config.get('text', 'yellow')),
    get_curses_color(theme_config.get('banner', 'yellow')),
    get_curses_color(theme_config.get('banner_text', 'black')),
    get_curses_color(theme_config.get('background', -1)),
    )


def conf_scr():
  '''Configure the screen and colors/etc'''
  stripe = 235
  curses.curs_set(0)
  curses.start_color()
  curses.use_default_colors()
  text, banner, banner_text, background = get_theme_colors()
  curses.init_pair(1, banner_text, banner)
  curses.init_pair(2, text, -1)
  curses.init_pair(3, text, stripe)
  curses.init_pair(4, getattr(curses, 'COLOR_GREEN'), -1)
  curses.init_pair(5, getattr(curses, 'COLOR_GREEN'), stripe)
  curses.init_pair(6, getattr(curses, 'COLOR_RED'), -1)
  curses.init_pair(7, getattr(curses, 'COLOR_RED'), stripe)
  curses.init_pair(8, 240, -1)
  curses.init_pair(9, 240, stripe)
  curses.init_pair(10, getattr(curses, 'COLOR_YELLOW'), stripe)
  curses.init_pair(11, getattr(curses, 'COLOR_YELLOW'), stripe)
  curses.halfdelay(12)

def terminal_size():
  import os, sys, io
  if not hasattr(sys.stdout, "fileno"):
    return -1, -1
  try:
    if not os.isatty(sys.stdout.fileno()):
      return -1, -1
  except io.UnsupportedOperation:
    return -1, -1
  env = os.environ
  def ioctl_GWINSZ(fd):
    try:
      import fcntl, termios, struct, os
      cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,'1234'))
    except Exception:
      return
    return cr
  cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
  if not cr:
    try:
      fd = os.open(os.ctermid(), os.O_RDONLY)
      cr = ioctl_GWINSZ(fd)
      os.close(fd)
    except Exception:
      pass
  if not cr:
    cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
  return int(cr[1]), int(cr[0])

def str_formatter(coin, val, held, ticks):
  '''Prepare the coin strings as per ini length/decimal place values'''
  global SYMBOL
  global CURRENCY
  ticks = { "t%d" % i : t for i,t in enumerate(ticks) }
  if held > 0:
    return '{:<{t0}} {:>{t1}.2f} {:>{t2}.{prec}f} {} {:>{t3}.{prec}f} {} {:>{t5}.3f}M {}'.format(
      coin, float(held), val[0], SYMBOL, float(held)*val[0],
      SYMBOL, val[1] / 1e6, SYMBOL, prec=2 if CURRENCY in ['EUR','USD'] else NROFDECIMALS,**ticks)
  else:
    ticks['t5'] += 2
    return '{:<{t0}} {:>{t1}} {:>{t2}.{prec}f} {} {:>{t3}} {:>{t5}.3f}M {}'.format(
      coin, '', val[0], SYMBOL, '', val[1] / 1e6, SYMBOL,
      prec=2 if CURRENCY in ['EUR','USD'] else NROFDECIMALS,**ticks)

def write_coins(name, coins, held, stdscr, x, y, off=0):
  width, _ = terminal_size()
  width -= 5
  ticks = [FIELD + 12 - FIELD_OFFSET,FIELD + 12,FIELD + 12,FIELD + 12,FIELD + 12,FIELD + 12,FIELD + 12,FIELD + 12]
  diffs = [0,0,2,2,2,3,3,3]
  scale = max(width / float(sum(ticks)), 1.0)
  hticks = [int(t * scale) for t in ticks]
  sticks = [int(t * scale - d) for t,d in zip(ticks,diffs)]
  total = 0

  if y > off + 1:
    coinvl = get_price(','.join(coins))
    s = sorted(list(zip(coins, coinvl, held)), key=SORT_FNS[SORTS[COLUMN]], reverse=ORDER)
    coinb = list(x[0] for x in s)
    coinvl = list(x[1] for x in s)
    heldb = list(x[2] for x in s)
    counter = 0
    for coin, val, held in zip(coinb, coinvl, heldb):
      if off + coinb.index(coin) + 4 < y:
        if float(held) > 0.0:
          stdscr.addnstr(off + coinb.index(coin) + 1, 0, str_formatter(coin, val, held, sticks), x, curses.color_pair(2 + counter % 2))
        else:
          stdscr.addnstr(off + coinb.index(coin) + 1, 0, str_formatter(coin, val, held, sticks), x, curses.color_pair(2 + counter % 2))
        for i in [2,3,4]:
          xs = hticks[0] + hticks[1] + 1 + (3+i-2)*(hticks[2]+1)
          if val[i] > 0:
            stdscr.addnstr(off + coinb.index(coin) + 1, xs, '  {:>{t6}.2f} %'.format(val[i], t6=sticks[-1]), x, curses.color_pair(4 + counter % 2))
          elif val[i] < 0:
            stdscr.addnstr(off + coinb.index(coin) + 1, xs, '  {:>{t6}.2f} %'.format(val[i], t6=sticks[-1]), x, curses.color_pair(6 + counter % 2))
          else:
            stdscr.addnstr(off + coinb.index(coin) + 1, xs, '  {:>{t6}.2f} %'.format(val[i], t6=sticks[-1]), x, curses.color_pair(2 + counter % 2))
      total += float(held) * val[0]
      counter += 1

  if y > off:
    global SYMBOL
    header = '{:<%d} {:>%d} {:>%d} {:>%d} {:>%d} {:>%d} {:>%d} {:>%d}' % tuple(hticks)
    if off == 0:
      if total == 0:
        header = header.format(
          name, '', 'PRICE ' +  SYMBOL, 'TOTAL ' + SYMBOL, 'VOLUME ' + SYMBOL, 'HOURLY', 'DAILY', 'WEEKLY')
      else:
        header = header.format(
          name, ("%.2f " % total) + SYMBOL, 'PRICE ' +  SYMBOL, 'TOTAL ' + SYMBOL, 'VOLUME ' + SYMBOL, 'HOURLY', 'DAILY', 'WEEKLY')
    else:
      header = header.format(
        name, ("%.2f " % total) + SYMBOL, '', '', '', '', '', '', '')
    stdscr.addnstr(off, 0, header, x, curses.color_pair(1))

  return total

def write_scr(stdscr, wallet, y, x):
  '''Write text and formatting to screen'''

  total = 0
  coinl = list(wallet.keys())
  heldl = list(wallet.values())

  coin = { 'custom' : [], '' : [] }
  held = { 'custom' : [], '' : [] }
  labels = []

  global CCSET, SHOW_BALANCES

  for i in range(len(coinl)):
    if coinl[i].lower() == 'bittrex':
      if SHOW_BALANCES:
        balance = bittrex(*heldl[i].split(':'))
        coin['bittrex'] = [ c['Currency'].replace('BCC','BCH') for c in balance['result'] if (c['Balance'] or 0) >= 0.01 and not c['Currency'].replace('BCC','BCH') in BLACKLIST ]
        held['bittrex'] = [  (c['Balance'] or 0) for c in balance['result'] if (c['Balance'] or 0) >= 0.01 and not c['Currency'].replace('BCC','BCH') in BLACKLIST ]
      labels.append('bittrex')
    elif coinl[i].lower() == 'binance':
      if SHOW_BALANCES:
        balance = binance(*heldl[i].split(':'))
        coin['binance'] = [ c['asset'].replace('BCC','BCH') for c in balance['balances'] if float(c['free']) + float(c['locked']) >= 0.01 and not c['asset'] in BLACKLIST ]
        held['binance'] = [ float(c['free']) + float(c['locked']) for c in balance['balances'] if float(c['free']) + float(c['locked']) >= 0.01 and not c['asset'] in BLACKLIST ]
      labels.append('binance')
    elif heldl[i].lower().startswith('0x'):
      if SHOW_BALANCES:
        tokens = ethereum(heldl[i])
        coin[coinl[i].lower()] = [ tok for tok in tokens[heldl[i]].keys() if tok != '.' and tok in coinstats.keys() and tokens[heldl[i]][tok] >= 0.01 ]
        held[coinl[i].lower()] = [ tokens[heldl[i]][tok] for tok in tokens[heldl[i]].keys() if tok != '.' and tok in coinstats.keys() and tokens[heldl[i]][tok] >= 0.01 ]
      labels.append(coinl[i].lower())
    elif (heldl[i][0] == '3' or heldl[i][0] == '1') and len(heldl[i]) > 24:
      if SHOW_BALANCES:
        coin[coinl[i].lower()] = ['BTC']
        assert not isinstance(bitcoin(heldl[i]),dict), str(bitcoin(heldl[i]).keys())
        held[coinl[i].lower()] = [bitcoin(heldl[i])]
      labels.append(coinl[i].lower())
    elif float(heldl[i]) >= 0.01 and SHOW_BALANCES:
      coin['custom'].append(coinl[i])
      held['custom'].append(float(heldl[i]))
  for i in range(len(coinl)):
    if not coinl[i].lower() in labels and not any([ coinl[i] in coin[k] for k in coin.keys() ]):
      coin[''].append(coinl[i])
      held[''].append(0)

  CCSET= set([])
  for k in coin.keys():
    CCSET = set(list(CCSET) + coin[k])

  off = 0
  stdscr.erase()
  default_keys = ['', 'custom', 'bittrex', 'etherdelta']
  for key in default_keys:
    if key in coin and coin[key]:
      total += write_coins(key.upper(), coin[key], held[key], stdscr, x, y, off)
      off += len(coin[key]) + 1
  for key in sorted(coin.keys()):
    if key not in default_keys and coin[key]:
      total += write_coins(key.upper(), coin[key], held[key], stdscr, x, y, off)
      off += len(coin[key]) + 1

  if y > off:
    if total > 0 and SHOW_BALANCES:
      stdscr.addnstr(y - 2, 0, 'Total Holdings: {:10.2f} {}  '
        .format(total, CURRENCY), x, curses.color_pair(11))
    stdscr.addnstr(y - 1, 0,
      '[A] Add coin [R] Remove coin [F] Switch currency [S] Sort [C] Cycle sort [H] %s balances [Q] Exit' % ['Show','Hide'][SHOW_BALANCES], x,
      curses.color_pair(2))

  #global LOGTIME, LOGFILE
  #if time.time() - LOGTIME > 60:
  #  LOGTIME = time.time()
  #  log = { key or str(int(time.time())) : dict(zip(coin[key],held[key])) for key in sorted(coin.keys())}
  #  with open(LOGFILE, 'a') as logfile:
  #    print(json.dumps(log),file=logfile)

def read_wallet():
  ''' Reads the wallet data from its json file '''
  try:
    with open(WALLETFILE, 'r') as f:
      return {k.upper():v for k,v in json.load(f).items()}
  except (FileNotFoundError, ValueError):
    # missing or malformed wallet
    write_wallet({})
    return {}

def write_wallet(wallet):
  ''' Writes the wallet data to its json file '''
  with open(WALLETFILE, 'w') as f:
    json.dump(wallet, f)

def get_string(stdscr, prompt):
  '''Requests and string from the user'''
  curses.echo()
  stdscr.clear()
  stdscr.addnstr(0, 0, prompt, -1, curses.color_pair(2))
  curses.curs_set(1)
  stdscr.refresh()
  in_str = stdscr.getstr(1, 0, 250).decode()
  curses.noecho()
  curses.curs_set(0)
  stdscr.clear()
  curses.halfdelay(10)
  return in_str

def add_coin(coin_amount, wallet):
  if not coin_amount.strip():
    return wallet
  if not ',' in coin_amount:
    if coin_amount.upper() in wallet.keys():
      return wallet
    coin_amount = "%s,%d" % (coin_amount, 0)
  coin,amount = coin_amount.split(',')
  wallet[coin.upper()] = amount
  return wallet

def remove_coin(coin, wallet):
  ''' Remove a coin and its amount from the wallet '''
  # coin = '' if window is resized while waiting for string
  if coin:
    coin = coin.upper()
    wallet.pop(coin, None)
  return wallet

def mainc(stdscr):
  inp = 0
  wallet = read_wallet()
  y, x = stdscr.getmaxyx()
  conf_scr()
  stdscr.bkgd(' ', curses.color_pair(2))
  stdscr.clear()
  while inp not in {KEY_ZERO, KEY_ESCAPE, KEY_Q, KEY_q}:
    while True:
      try:
        write_scr(stdscr, wallet, y, x)
      except curses.error:
        pass

      inp = stdscr.getch()
      if inp != curses.KEY_RESIZE:
        break
      y, x = stdscr.getmaxyx()

    if inp in {KEY_a, KEY_A, KEY_ENTER}:
      if y > 2:
        data = get_string(stdscr,
          'Enter in format Symbol e.g. ETH or Symbol,Amount e.g. BTC,10 or Name,Address e.g. TREZOR,0xab5801a7d398351b8be11c439e05c5b3259aec9b')
        wallet = add_coin(data, wallet)
        write_wallet(wallet)

    if inp in {KEY_r, KEY_R, KEY_BACKSPACE}:
      if y > 2:
        data = get_string(stdscr,
          'Enter coin or address to be removed, e.g. BTC')
        wallet = remove_coin(data, wallet)
        write_wallet(wallet)

    if inp in {KEY_s, KEY_S}:
      if y > 2:
        global ORDER
        ORDER = not ORDER

    if inp in {KEY_c, KEY_C}:
      if y > 2:
        global COLUMN
        COLUMN = (COLUMN + 1) % len(SORTS)

    if inp in {KEY_f, KEY_F, KEY_SPACE}:
      if y > 2:
        global CURRENCY, NROFDECIMALS, CURRENCYCOUNTER, CURRENCYLIST, SYMBOL, SYMBOLLIST
        CURRENCYCOUNTER = (CURRENCYCOUNTER + 1) % len(CURRENCYLIST)
        CURRENCY = CURRENCYLIST[CURRENCYCOUNTER]
        SYMBOL = SYMBOLMAP[CURRENCY]
        NROFDECIMALS = 2 if isfiat(CURRENCY) else 6

    if inp in {KEY_h, KEY_H}:
      global SHOW_BALANCES
      SHOW_BALANCES = 1 - SHOW_BALANCES

def main():
  if os.path.isfile(BASEDIR):
    sys.exit('Please remove your old configuration file at {}'.format(BASEDIR))
  os.makedirs(BASEDIR, exist_ok=True)

  global BLACKLIST, CURRENCYLIST, CURRENCY, SYMBOL, SYMBOLMAP
  BLACKLIST = CONFIG['api'].get('blacklist', '').split(',')
  CURRENCYLIST = CONFIG['api'].get('currency', 'USD,ETH,BTC,EUR').split(',')
  assert CURRENCYLIST, "list of currency must not be empty"
  CURRENCY = CURRENCYLIST[0]
  for cur in CURRENCYLIST:
    if not cur in SYMBOLMAP:
      if not cur[0] in SYMBOLMAP:
        SYMBOLMAP[cur] = cur[0]
      else:
        SYMBOLMAP[cur] = cur

  update_coins()
  _thread.start_new_thread(ticker, ())

  global FIELD, FIELD_OFFSET
  FIELD = float(CONFIG['theme'].get('field_length', 0))
  FIELD_OFFSET = float(CONFIG['theme'].get('field_offset', 4))

  #requests_cache.install_cache(cache_name='api_cache', backend='memory',
  #  expire_after=int(CONFIG['api'].get('cache', 60)))

  global WALLETFILE
  if len(sys.argv) > 1:
    WALLETFILE = os.path.join(BASEDIR, '%s.json' % sys.argv[1])
    CCSET = set(read_wallet().keys())

  curses.wrapper(mainc)

if __name__ == "__main__":
  main()
