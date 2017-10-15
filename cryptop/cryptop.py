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

# GLOBALS!
BASEDIR = os.path.join(os.path.expanduser('~'), '.cryptop')
WALLETFILE = os.path.join(BASEDIR, 'wallet.json')
LEDGERFILE = os.path.join(BASEDIR, 'ledger.json')
CONFFILE = os.path.join(BASEDIR, 'config.ini')
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

VIEW = 'WALLET'
FIAT = 'EUR'
CURRENCYLIST = [FIAT, 'ETH', 'BTC']
SYMBOL = '€'
SYMBOLLIST = ['€','Ξ','Ƀ']
CURRENCYCOUNTER = 0
CURRENCY = FIAT
NROFDECIMALS = 2
BALANCE_TIME = 0

KEY_ESCAPE = 27
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

  if time.time() - tokens[address][token]['time'] > 60:
    tokens[address][token]['time'] = time.time()
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

  global tokens, ethplorer_conn, etherscan_conn
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
        tokens[address][tok['tokenInfo']['symbol']], _ = get_erc20_balance(tok['tokenInfo']['symbol'], address)
      try:
        r = requests.get("https://api.etherscan.io/api?module=account&action=balance&address=%s&tag=latest&apikey=" % address)
        balance = r.json()
      except Exception:
        return tokens
      if 'result' in balance.keys():
        tokens[address]['ETH'] = float(balance['result']) / 1e18
    else:
      for tok in data['tokens']:
        tokens[address][tok['tokenInfo']['symbol']] = tok['balance'] / 10**int(tok['tokenInfo']['decimals'])
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
def get_price(coin, curr=None):
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
  ticks = { "t%d" % i : t for i,t in enumerate(ticks) }
  return '{:<{t0}} {:>{t1}.2f} {:>{t2}.{prec}f} {} {:>{t3}.{prec}f} {} {:>{t5}.3f}M {}'.format(
    coin, float(held), val[0], SYMBOL, float(held)*val[0],
    SYMBOL, val[1], SYMBOL, prec=NROFDECIMALS,**ticks)

def write_scr(stdscr, wallet, y, x):
  '''Write text and formatting to screen'''
  from math import ceil
  width, _ = terminal_size()
  width -= 5
  ticks = [7,15,15,15,15,15]
  diffs = [0,0,2,2,3,3]
  scale = max(width / float(sum(ticks)),1.0)
  hticks = [int(t * scale) for t in ticks]
  sticks = [int(t * scale - d) for t,d in zip(ticks,diffs)]

  stdscr.erase()
  if y >= 0:
    header = '{:<%d} {:>%d} {:>%d} {:>%d} {:>%d} {:>%d}' % tuple(hticks)
    header = header.format(
      'TREZOR', 'HODLING', 'CURRENT PRICE', 'TOTAL VALUE', 'MARKET CAP', '24H CHANGE')
    stdscr.addnstr(0, 0, header, x, curses.color_pair(1))

  totall = 0
  totalb = 0
  coinlr = list(wallet.keys())
  heldlr = list(wallet.values())

  coinl = []
  heldl = []

  coinb = []
  heldb = []

  for i in range(len(coinlr)):
    if coinlr[i].startswith('bittrex:'):
      nonce = str(int(time.time() * 1000))
      url = "https://bittrex.com/api/v1.1/account/getbalances?apikey=%s&nonce=%s&" % (coinlr[i].split(':')[1], nonce)
      global balance
      global BALANCE_TIME
      if time.time() - BALANCE_TIME > 60:
        try:
          ret = requests.get(
              url,
              headers={"apisign": hmac.new(heldlr[i].encode(), url.encode(), hashlib.sha512).hexdigest()},
              timeout=5
          ).json()
          balance = ret
          BALANCE_TIME = time.time()
        except:
          continue
      for c in balance['result']:
        if c['Balance'] >= 0.01:
          coinb.append(c['Currency'].replace('BCC','BCH'))
          heldb.append(c['Balance'])
    elif str(coinlr[i]).lower().strip().startswith('0x'):
      address = str(coinlr[i]).lower().strip()
      tokens = get_ethereum(address)
      for tok in tokens[address].keys():
        if tok != '.':
          coinl.append(tok)
          heldl.append(tokens[address][tok])
          #delta = get_etherdelta(tok, address)
    elif str(heldlr[i]).lower().strip().startswith('0x'):
      coinl.append(coinlr[i])
      tok_balance, eth_balance = get_erc20_balance(coinlr[i], heldlr[i].lower().strip())
      heldl.append(tok_balance)
      if not 'ETH' in coinlr and not 'ETH' in coinl:
        coinl.append('ETH')
        heldl.append(eth_balance)
    else:
      coinl.append(coinlr[i])
      heldl.append(heldlr[i])

  ncl = len(coinl)
  ncb = len(coinb)

  if coinl:
    coinvl = get_price(','.join(coinl))

    if y > 2:
      s = sorted(list(zip(coinl, coinvl, heldl)), key=SORT_FNS[SORTS[COLUMN]], reverse=ORDER)
      coinl = list(x[0] for x in s)
      coinvl = list(x[1] for x in s)
      heldl = list(x[2] for x in s)
      counter = 0
      for coin, val, held in zip(coinl, coinvl, heldl):
        if coinl.index(coin) + 1 < y:
          if float(held) > 0.0:
            stdscr.addnstr(coinl.index(coin) + 1, 0, str_formatter(coin, val, held, sticks), x, curses.color_pair(2 + counter % 2))
          else:
            stdscr.addnstr(coinl.index(coin) + 1, 0, str_formatter(coin, val, held, sticks), x, curses.color_pair(8 + counter % 2))

          if val[2] > 0:
            stdscr.addnstr(coinl.index(coin) + 1, hticks[0] + hticks[1] + 1 + 3*(hticks[2]+1),
            '  {:>{t6}.2f} %'.format(val[2], t6=sticks[-1]), x, curses.color_pair(4 + counter % 2))
          elif val[2] < 0:
            stdscr.addnstr(coinl.index(coin) + 1, hticks[0] + hticks[1] + 1 + 3*(hticks[2]+1),
            '  {:>{t6}.2f} %'.format(val[2], t6=sticks[-1]), x, curses.color_pair(6 + counter % 2))
          else:
            stdscr.addnstr(coinl.index(coin) + 1, hticks[0] + hticks[1] + 1 + 3*(hticks[2]+1),
            '  {:>{t6}.2f} %'.format(val[2], t6=sticks[-1]), x, curses.color_pair(2 + counter % 2))
        totall += float(held) * val[0]
        counter += 1

  if y > ncl + 1:
    stdscr.addnstr(ncl + 1, 0, 'Value: {:10.2f} {}  '
      .format(totall, CURRENCY), x, curses.color_pair(10))

  if coinb:
    if y > ncl + 2:
      header = '{:<%d} {:>%d} {:>%d} {:>%d} {:>%d} {:>%d}' % tuple(hticks)
      header = header.format(
        'BITTREX', 'HODLING', 'CURRENT PRICE', 'TOTAL VALUE', 'MARKET CAP', '24H CHANGE')
      stdscr.addnstr(ncl + 2, 0, header, x, curses.color_pair(1))
    if y > ncl + 3:
      coinvl = get_price(','.join(coinb))
      s = sorted(list(zip(coinb, coinvl, heldb)), key=SORT_FNS[SORTS[COLUMN]], reverse=ORDER)
      coinb = list(x[0] for x in s)
      coinvl = list(x[1] for x in s)
      heldb = list(x[2] for x in s)
      counter = 0
      for coin, val, held in zip(coinb, coinvl, heldb):
        if ncl + coinb.index(coin) + 4 < y:
          if float(held) > 0.0:
            stdscr.addnstr(ncl + coinb.index(coin) + 3, 0, str_formatter(coin, val, held, sticks), x, curses.color_pair(2 + counter % 2))
          else:
            stdscr.addnstr(ncl + coinb.index(coin) + 3, 0, str_formatter(coin, val, held, sticks), x, curses.color_pair(8 + counter % 2))

          if val[2] > 0:
            stdscr.addnstr(ncl + coinb.index(coin) + 3, hticks[0] + hticks[1] + 1 + 3*(hticks[2]+1),
            '  {:>{t6}.2f} %'.format(val[2], t6=sticks[-1]), x, curses.color_pair(4 + counter % 2))
          elif val[2] < 0:
            stdscr.addnstr(ncl + coinb.index(coin) + 3, hticks[0] + hticks[1] + 1 + 3*(hticks[2]+1),
            '  {:>{t6}.2f} %'.format(val[2], t6=sticks[-1]), x, curses.color_pair(6 + counter % 2))
          else:
            stdscr.addnstr(ncl + coinb.index(coin) + 3, hticks[0] + hticks[1] + 1 + 3*(hticks[2]+1),
            '  {:>{t6}.2f} %'.format(val[2], t6=sticks[-1]), x, curses.color_pair(2 + counter % 2))
        totalb += float(held) * val[0]
        counter += 1


  if y > ncl + ncb + 3:
    stdscr.addnstr(ncl + ncb + 3, 0, 'Value: {:10.2f} {}  '
      .format(totalb, CURRENCY), x, curses.color_pair(10))
    stdscr.addnstr(y - 2, 0, 'Total Holdings: {:10.2f} {}  '
      .format(totall+totalb, CURRENCY), x, curses.color_pair(11))
    stdscr.addnstr(y - 1, 0,
      '[A] Add coin [R] Remove coin [T] Add transaction [F] Switch FIAT/ETH [V] View ledger [S] Sort [C] Cycle sort [Q] Exit', x,
      curses.color_pair(2))


def read_wallet():
  ''' Reads the wallet data from its json file '''
  try:
    with open(WALLETFILE, 'r') as f:
      return json.load(f)
  except (FileNotFoundError, ValueError):
    # missing or malformed wallet
    write_wallet({})
    return {}

def read_ledger():
  ''' Reads the transaction ledger data from its json file '''
  try:
    with open(LEDGERFILE, 'r') as f:
      return json.load(f)
  except (FileNotFoundError, ValueError):
    # missing or malformed wallet
    write_ledger({})
    return {}

def write_wallet(wallet):
  ''' Writes the wallet data to its json file '''
  with open(WALLETFILE, 'w') as f:
    json.dump(wallet, f)

def write_ledger(ledger):
  ''' Writes the ledger data to its json file '''
  with open(LEDGERFILE, 'w') as f:
    json.dump(ledger, f)


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
  ''' Add a coin and its amount to the wallet '''
  if not ',' in coin_amount and coin_amount.lower().strip().startswith('0x'):
    coin = coin_amount.lower().strip()
    amount = 0
  elif not ',' in coin_amount and coin_amount.lower().strip().startswith('etherdelta:'):
    coin = coin_amount.lower().strip()
    amount = 0
  elif coin_amount.split(',')[-1].lower().strip().startswith('0x'):
    coin, amount = coin_amount.split(',')
    coin = coin.upper()
  elif coin_amount.startswith('bittrex:'):
    coin, amount = coin_amount.split(',')
  else:
    coin_amount = coin_amount.upper()
    if not COIN_FORMAT.match(coin_amount):
      return wallet
    coin, amount = coin_amount.split(',')
    if not if_coin(coin):
      return wallet

  wallet[coin] = amount
  return wallet


def add_transaction(transaction, wallet, ledger):
  ''' Add a transaction to ledger and update wallet accordingly '''

  transaction = transaction.upper()
  if not COIN_FORMAT.match(transaction):
    return wallet, ledger

  coin_out, amount_out, coin_in, amount_in = transaction.split(',')
  if (not if_coin(coin_out)) or (not if_coin(coin_out)):
    return wallet, ledger

  # Add transaction to ledger
  now = datetime.datetime.now()
  ledger[now.strftime("%Y-%m-%d %H:%M")] = transaction

  # Update wallet
  current_amount_coin_out = float(wallet[coin_out])
  current_amount_coin_in = float(wallet[coin_in])
  wallet[coin_out] = current_amount_coin_out - float(amount_out)
  wallet[coin_in] = current_amount_coin_in + float(amount_in)

  return wallet, ledger


def remove_coin(coin, wallet):
  ''' Remove a coin and its amount from the wallet '''
  # coin = '' if window is resized while waiting for string
  if coin:
    coin = coin.lower()
    wallet.pop(coin, None)
  return wallet

def view_ledger(stdscr, ledger, x, y):
  '''Write transactions to screen'''
  stdscr.erase()

  if y >= 1:
    stdscr.addnstr(0, 0, ntpath.basename(WALLETFILE), x, curses.color_pair(2))
  if y >= 2:
    width, _ = terminal_size()
    width -= 5
    ticks = [16,5,14,5,14,18,18]
    scale = max(width / float(sum(ticks)),1.0)
    ticks = [int(t * scale) for t in ticks]
    header = '{:<%d} {:<%d} {:<%d} {:<%d} {:<%d} {:<%d} {:>%d}' % tuple(ticks)
    header = header.format(
      'DATE', 'OUT', 'AMOUNT', 'IN', 'AMOUNT', 'RATE OUT/IN', 'RATE  IN/OUT')
    stdscr.addnstr(1, 0, header, x, curses.color_pair(1))


  dates = list(ledger.keys())
  transactions = list(ledger.values())

  if transactions:
    if y > 3:
      counter = 0
      ticks = [17,10,11,8,12,12,12]
      scale = max(width / float(sum(ticks)),1.0)
      ticks = [int(t * scale) for t in ticks]
      header = '{:<%d} {:>%d} {:>%d.6f} {:>%d} {:>%d.6f} {:>%d.6f} {}/{} {:>%d.6f} {}/{}' % tuple(ticks)
      for date, transaction in list(zip(dates, transactions)):
        info = transaction.split(',')
        printme = header.format(
          date, info[0], float(info[1]), info[2], float(info[3]),
          float(info[1])/float(info[3]), info[0], info[2],
          float(info[3])/float(info[1]), info[2], info[0])
        stdscr.addnstr(dates.index(date) + 2, 0, printme, x, curses.color_pair(2 + counter % 2))
        counter += 1

  if y > len(transactions) + 3:
    stdscr.addnstr(y - 1, 0,
      '[V] View wallet [T] Add transaction [Q] Exit', x,
      curses.color_pair(2))


def mainc(stdscr):
  inp = 0
  wallet = read_wallet()
  ledger = read_ledger()
  y, x = stdscr.getmaxyx()
  conf_scr()
  stdscr.bkgd(' ', curses.color_pair(2))
  stdscr.clear()
  #stdscr.nodelay(1)
  # while inp != 48 and inp != 27 and inp != 81 and inp != 113:
  while inp not in {KEY_ZERO, KEY_ESCAPE, KEY_Q, KEY_q}:
    global VIEW
    while True:
      try:
        if VIEW is 'WALLET':
          write_scr(stdscr, wallet, y, x)
        elif VIEW is 'LEDGER':
          view_ledger(stdscr, ledger, x, y)
      except curses.error:
        pass

      inp = stdscr.getch()
      if inp != curses.KEY_RESIZE:
        break
      y, x = stdscr.getmaxyx()

    if inp in {KEY_a, KEY_A}:
      if y > 2:
        data = get_string(stdscr,
          'Enter in format Symbol,Amount e.g. BTC,10')
        wallet = add_coin(data, wallet)
        write_wallet(wallet)

    if inp in {KEY_r, KEY_R}:
      if y > 2:
        data = get_string(stdscr,
          'Enter the symbol of coin to be removed, e.g. BTC')
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

    if inp in {KEY_f, KEY_F}:
      if y > 2:
        global CURRENCY, NROFDECIMALS, FIAT, CURRENCYCOUNTER, CURRENCYLIST, SYMBOL, SYMBOLLIST
        CURRENCYCOUNTER = (CURRENCYCOUNTER + 1) % len(CURRENCYLIST)
        CURRENCY = CURRENCYLIST[CURRENCYCOUNTER]
        SYMBOL = SYMBOLLIST[CURRENCYCOUNTER]

        if CURRENCY is FIAT:
          NROFDECIMALS = 2
        else:
          NROFDECIMALS = 6

    if inp in {KEY_t, KEY_T}:
      if y > 2:
        data = get_string(stdscr,
          'Enter transaction (Out,Amount,In,Amount), e.g. BTC,10,ETH,10')
        wallet, ledger = add_transaction(data, wallet, ledger)
        write_wallet(wallet)
        write_ledger(ledger)

    if inp in {KEY_v, KEY_V}:
      if VIEW is 'WALLET':
        VIEW = 'LEDGER'
      elif VIEW is 'LEDGER':
        VIEW = 'WALLET'


def main():
  if os.path.isfile(BASEDIR):
    sys.exit('Please remove your old configuration file at {}'.format(BASEDIR))
  os.makedirs(BASEDIR, exist_ok=True)

  global CONFIG
  CONFIG = read_configuration(CONFFILE)

  global FIAT, CURRENCYLIST, CURRENCY, SYMBOL, SYMBOLLIST
  FIAT = CONFIG['api'].get('currency', 'EUR')
  CURRENCY = FIAT
  CURRENCYLIST = [FIAT, 'ETH', 'BTC'] + (['USD'] if FIAT != 'USD' else [])

  if FIAT == 'EUR':
    SYMBOL = '€'
    SYMBOLLIST = ['€','Ξ','Ƀ','$']
  elif FIAT == 'USD':
    SYMBOL = '$'
    SYMBOLLIST = ['$','Ξ','Ƀ']

  requests_cache.install_cache(cache_name='api_cache', backend='memory',
    expire_after=int(CONFIG['api'].get('cache', 60)))

  global WALLETFILE
  if len(sys.argv) > 1:
    WALLETFILE = WALLETFILE = os.path.join(BASEDIR, '%s.json' % sys.argv[1])

  curses.wrapper(mainc)


if __name__ == "__main__":
  main()
