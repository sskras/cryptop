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

# GLOBALS!
BASEDIR = os.path.join(os.path.expanduser('~'), '.cryptop')
WALLETFILE = os.path.join(BASEDIR, 'wallet.json')
CONFFILE = os.path.join(BASEDIR, 'config.ini')
LOGFILE = os.path.join(BASEDIR, 'trace.log')
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
BALANCE_TIME = 0

KEY_ESCAPE = 27
KEY_ENTER = 13
KEY_BACKSPACE = 8
KEY_SPACE = 32
KEY_ZERO = 48
KEY_A = 65
KEY_F = 70
KEY_Q = 81
KEY_R = 82
KEY_S = 83
KEY_C = 67
KEY_T = 84
KEY_V = 86
KEY_a = 97
KEY_f = 102
KEY_q = 113
KEY_r = 114
KEY_s = 115
KEY_c = 99
KEY_t = 116
KEY_v = 118

def read_configuration(confpath):
  # copy our default config file
  if not os.path.isfile(confpath):
    defaultconf = pkg_resources.resource_filename(__name__, 'config.ini')
    shutil.copyfile(defaultconf, CONFFILE)
  CONFIG.read(confpath)
  return CONFIG

def if_coin(coin, url='https://www.cryptocompare.com/api/data/coinlist/'):
  '''Check if coin exists'''
  return coin in requests.get(url).json()['Data']

coinstats = {}
coinmap = {'KNC' : 'kyber-network', 'BTG' : 'bitcoin-gold'}
def update_coins():
  global CURRENCYLIST
  cmclist = set([])
  for coin in [ c for c in CURRENCYLIST if not isfiat(c) ]:
    try:
      ret = requests.get('https://min-api.cryptocompare.com/data/histohour?fsym=%s&tsym=USD&toTs=%d&limit=175' % (coin.upper(),int(time.time()))).json()
    except:
      cmclist.add(coin.upper())
      continue
    if not coin in coinstats.keys():
      coinstats[coin] = {}
    coinstats[coin]['price_usd'] = ret['Data'][-1]['close']
    coinstats[coin]['percent_change_1h'] = 100. - 100. * (ret['Data'][-2]['close'] / ret['Data'][-1]['close'])
    coinstats[coin]['percent_change_24h'] = 100. - 100. * (ret['Data'][-25]['close'] / ret['Data'][-1]['close'])
    coinstats[coin]['percent_change_7d'] = 100. - 100. * (ret['Data'][-169]['close'] / ret['Data'][-1]['close'])

  cmc = http.client.HTTPSConnection("api.coinmarketcap.com")
  try:
    cmc.request("GET", '/v1/ticker/?convert=EUR&limit=2000', {}, {})
    data = cmc.getresponse()
    data = json.loads(data.read().decode())
  except:
    return
  for item in data[::-1]:
    if item['symbol'] in coinmap.keys() and coinmap[item['symbol']] != item['id']:
      continue
    if item['symbol'] in CURRENCYLIST and not isfiat(item['symbol']) and not item['symbol'] in cmclist:
      coinstats[item['symbol']]['24h_volume_usd'] = item['24h_volume_usd']
    else:
      coinstats[item['symbol']] = item
  from datetime import date, timedelta
  for fiat in [ f for f in CURRENCYLIST if isfiat(f) and f != 'USD']:
    if not fiat in coinstats.keys():
      coinstats[fiat] = {}
    try:
      coinstats[fiat]['price_usd'] = 1. / requests.get('https://api.fixer.io/latest?base=USD').json()['rates'][fiat]
      d24h = date.today() - timedelta(1)
      r24h = 1. / requests.get('https://api.fixer.io/' + d24h.strftime('%Y-%m-%d') + '?base=USD').json()['rates'][fiat]
      coinstats[fiat]['percent_change_24h'] = 100. - 100. * r24h / coinstats[fiat]['price_usd']
      coinstats[fiat]['percent_change_1h'] = coinstats[fiat]['percent_change_24h'] / 24.

      d7d = date.today() - timedelta(7)
      r7d = 1. / requests.get('https://api.fixer.io/' + d7d.strftime('%Y-%m-%d') + '?base=USD').json()['rates'][fiat]
      coinstats[fiat]['percent_change_7d'] = 100. - 100. * r7d / coinstats[fiat]['price_usd']
    except:
      try:
        rates = requests.get('https://www.quandl.com/api/v3/datasets/ECB/EURUSD').json()['dataset']['data']
      except:
        continue
      coinstats[fiat]['price_usd'] = rates[0][1]
      coinstats[fiat]['percent_change_24h'] = 100. - 100. * rates[1][1] / rates[0][1]
      coinstats[fiat]['percent_change_1h'] = coinstats[fiat]['percent_change_24h'] / 24.
      coinstats[fiat]['percent_change_7d'] = 100. - 100. * rates[7][1] / rates[0][1]

def ticker():
  import time
  ts = 0
  while True:
    time.sleep(max(min(int(time.time()) - ts, 15), 2))
    ts = int(time.time())
    update_coins()

bittrex_tokens = {}
bittrex_time = 0
def update_bittrex(key, secret):
  global bittrex_time
  if time.time() - bittrex_time > 15:
    bittrex_time = time.time()
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

def bittrex(key, secret):
  if not key in bittrex_tokens.keys():
    return update_bittrex(key, secret)
  _thread.start_new_thread(update_bittrex, (key,secret))
  return bittrex_tokens[key]

binance_tokens = {}
binance_time = 0
def update_binance(key, secret):
  global binance_time
  if time.time() - binance_time > 15:
    binance_time = time.time()
    try:
      url = "https://api.binance.com/api/v3/account?timestamp=%d" % int(time.time() * 1000)
      url += '&signature=' + hmac.new(secret.encode('utf-8'), url.split('?')[1].encode('utf-8'), hashlib.sha256).hexdigest()
      ret = requests.get(url,
          headers={'X-MBX-APIKEY': key, 'Accept': 'application/json', 'User-Agent': 'binance/python'},
          timeout=5).json()
      if 'balances' in ret and ret['balances'] is not None:
        binance_tokens[key] = ret
    except:
      if not key in binance_tokens.keys():
        binance_tokens[key] = { 'balances' : [] }
  return binance_tokens[key]

def binance(key, secret):
  if not key in binance_tokens.keys():
    return update_binance(key, secret)
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

def update_erc20_balance(address): # doesn't catch OMG :(
  import json
  import time
  global erc20_time, erc20_block, erc20, erc20_contracts

  def er(method, default):
    try:
      etherscan_conn = http.client.HTTPSConnection("api.etherscan.io")
      etherscan_conn.request("GET", method, {}, {})
      data = etherscan_conn.getresponse()
      data = json.loads(data.read().decode())
      if 'result' in data.keys():
        return data['result']
      return default
    except:
      return default

  if not address in erc20:
    erc20[address] = {'ETH' : 0}
    erc20_block[address] = 0

  current_block = int(er("/api?module=proxy&action=eth_blockNumber&apikey=", "0xFFFFF"),0)
  if current_block - erc20_block[address] >= 10:
    erc20[address]['ETH'] = float(er("/api?module=account&action=balance&address=%s&tag=latest&apikey=" % address, 0)) / 1e18
    transactions = er("/api?module=account&action=txlist&address=%s&startblock=%d&endblock=%d&sort=asc&apikey=" % (address,erc20_block[address],current_block),[])
    erc20_block[address] = current_block
    erc20_contracts = set(list(erc20_contracts) + [tx['to'] for tx in transactions if tx['to'] != address])
    erc20_contracts = set(list(erc20_contracts) + [tx['contractAddress'] for tx in transactions if tx['contractAddress'] != ''])
    conn = http.client.HTTPSConnection("api.tokenbalance.com")
    for tok in erc20_contracts:
      conn.request("GET", "/token/%s/%s" % (tok,address), {}, {})
      res = conn.getresponse()
      data = json.loads(res.read().decode())
      if 'symbol' in data.keys():
        try:
          conn.request("GET", "/balance/%s/%s" % (tok,address), {}, {})
          res = conn.getresponse()
          balance = json.loads(res.read().decode())
        except:
          continue
        if balance > 0:
          erc20[address][data['symbol']] = balance

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

  if True: #time.time() - tokens[address][token]['time'] > 60:
    #tokens[address][token]['time'] = time.time()
    try:
      if token.lower() ==  'eth':
        r = requests.get("https://etherscan.io/api?module=account&action=balance&address=%s&tag=latest&apikey=" % address, {}, {})
        data = r.json()
        if 'result' in data and data['result']:
          tokens[address][token]['balance'] = float(data['result']) / 1e18
          tokens[address]['token']['eth_balance'] = float(data['result']) / 1e18
      else:
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
        r = requests.get("https://api.tokenbalance.com/token/%s/%s" % (contract,address))
        tokens[address][token].update(r.json())
    except:
      pass

  return tokens[address][token]['balance'],tokens[address][token]['eth_balance']

tokens = {}
def get_ethereum(address):
  import json

  global BLACKLIST, tokens, ethplorer_conn, etherscan_conn
  if not address in tokens:
    tokens[address] = {'.' : 0}

  if time.time() - tokens[address]['.'] > 60:
    try:
      r = requests.get('https://api.ethplorer.io/getAddressInfo/%s?apiKey=freekey' % address)
      data = r.json()
      tokens[address] = {'.' : time.time(), 'ETH':data['ETH']['balance']}
      apidown = not tokens[address]['ETH']
    except Exception:
      apidown=True
      data={}
    if apidown and 'tokens' in data.keys():
      for tok in data['tokens']:
        if tok['tokenInfo']['symbol'] not in BLACKLIST:
          tokens[address][tok['tokenInfo']['symbol']], _ = get_erc20_balance(tok['tokenInfo']['symbol'], address)
      try:
        r = requests.get("https://api.etherscan.io/api?module=account&action=balance&address=%s&tag=latest&apikey=" % address)
        balance = r.json()
      except Exception:
        return tokens
      if 'result' in balance.keys():
        tokens[address]['ETH'] = float(balance['result']) / 1e18
    elif 'tokens' in data.keys():
      for tok in data['tokens']:
        if tok['tokenInfo']['symbol'] not in BLACKLIST:
          tokens[address][tok['tokenInfo']['symbol']] = tok['balance'] / 10**int(tok['tokenInfo']['decimals'])
  return tokens

def ethereum(address):
  if not address in tokens.keys():
    return get_ethereum(address)
  _thread.start_new_thread(get_ethereum, (address,))
  return tokens

CONTRACTS = {}
ETHERDELTA = {}
def get_etherdelta(token, address):
  import json
  conn = http.client.HTTPSConnection("api.etherdelta.com")
  global CONTRACTS, ETHERELTA
  token = token.upper()
  address='0xB7edBA4d444Ebca43A41880A8189c03bf52d1152'.lower()
  token = 'LINK'

  if not address in ETHERDELTA.keys():
    ETHERDELTA[address] = {}
  if not token in ETHERDELTA[address]:
    ETHERDELTA[address][token] = {'.' : 0, 'balance' :  0.0}

  if not CONTRACTS.keys():
    conn.request('GET', '/returnTicker')
    data = json.loads(conn.getresponse().read().decode())
    for k in data.keys():
      if k.startswith('ETH_'):
        CONTRACTS[k[4:]] = data[k]['tokenAddr']

  if not token in CONTRACTS.keys():
    return 0.0

  if time.time() - ETHERDELTA[address][token]['.'] > 60:
    ETHERDELTA[address][token]['.'] = time.time()
    conn.request('GET','/funds/%s/%s/0' % (address,CONTRACTS[token]))

    data = json.loads(conn.getresponse().read().decode())
    #try:
    #  data = json.loads(conn.getresponse().read().decode())
    #except:
    #  return 0.0
    assert False, str(data)
  return ETHERDELTA[address][token]

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
    r = requests.get(fmt.format(coin, curr))
  except requests.exceptions.RequestException:
    return last_price[coin]

  try:
    data_raw = r.json()['RAW']
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
    if curr != 'USD':
      price /= sf(coinstats[curr]['price_usd'])
      volume /= sf(coinstats[curr]['price_usd'])
      s1h = sf(coinstats[curr]['percent_change_1h'])/100.
      s24h = sf(coinstats[curr]['percent_change_24h'])/100.
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
  ticks = [FIELD + 8,FIELD + 12,FIELD + 12,FIELD + 12,FIELD + 12,FIELD + 12,FIELD + 12,FIELD + 12]
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

  for i in range(len(coinl)):
    if coinl[i].lower() == 'bittrex':
      balance = bittrex(*heldl[i].split(':'))
      coin['bittrex'] = [ c['Currency'].replace('BCC','BCH') for c in balance['result'] if c['Balance'] >= 0.01 ]
      held['bittrex'] = [ c['Balance'] for c in balance['result'] if c['Balance'] >= 0.01 ]
      labels.append('bittrex')
    elif coinl[i].lower() == 'binance':
      balance = binance(*heldl[i].split(':'))
      coin['binance'] = [ c['asset'].replace('BCC','BCH') for c in balance['balances'] if float(c['free']) + float(c['locked']) >= 0.01 ]
      held['binance'] = [ float(c['free']) + float(c['locked']) for c in balance['balances'] if float(c['free']) + float(c['locked']) >= 0.01 ]
      labels.append('binance')
    elif heldl[i].lower().startswith('0x'):
      tokens = ethereum(heldl[i])
      coin[coinl[i].lower()] = [ tok for tok in tokens[heldl[i]].keys() if tok != '.' and tok in coinstats.keys() and tokens[heldl[i]][tok] >= 0.01 ]
      held[coinl[i].lower()] = [ tokens[heldl[i]][tok] for tok in tokens[heldl[i]].keys() if tok != '.' and tok in coinstats.keys() and tokens[heldl[i]][tok] >= 0.01 ]
      labels.append(coinl[i].lower())
    elif float(heldl[i]) >= 0.01:
      coin['custom'].append(coinl[i])
      held['custom'].append(float(heldl[i]))
  for i in range(len(coinl)):
    if not coinl[i].lower() in labels and not any([ coinl[i] in coin[k] for k in coin.keys() ]):
      coin[''].append(coinl[i])
      held[''].append(0)

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
    stdscr.addnstr(y - 2, 0, 'Total Holdings: {:10.2f} {}  '
      .format(total, CURRENCY), x, curses.color_pair(11))
    stdscr.addnstr(y - 1, 0,
      '[A] Add coin [R] Remove coin [F] Switch currency [S] Sort [C] Cycle sort [Q] Exit', x,
      curses.color_pair(2))

  global LOGTIME, LOGFILE
  if time.time() - LOGTIME > 60:
    LOGTIME = time.time()
    log = { key or str(int(time.time())) : dict(zip(coin[key],held[key])) for key in sorted(coin.keys())}
    with open(LOGFILE, 'a') as logfile:
      print(json.dumps(log),file=logfile)

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

def main():
  if os.path.isfile(BASEDIR):
    sys.exit('Please remove your old configuration file at {}'.format(BASEDIR))
  os.makedirs(BASEDIR, exist_ok=True)

  global CONFIG
  CONFIG = read_configuration(CONFFILE)

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

  global FIELD
  FIELD = int(CONFIG['theme'].get('field_length'))

  requests_cache.install_cache(cache_name='api_cache', backend='memory',
    expire_after=int(CONFIG['api'].get('cache', 60)))

  global WALLETFILE
  if len(sys.argv) > 1:
    WALLETFILE = os.path.join(BASEDIR, '%s.json' % sys.argv[1])

  curses.wrapper(mainc)

if __name__ == "__main__":
  main()
