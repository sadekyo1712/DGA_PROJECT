# Bui Duc Hung - KSCLC HTTT&TT K57 - BKHN - 5/2017
# DGA Classify Project

from datetime import datetime
from StringIO import StringIO
from urllib import urlopen
from zipfile import ZipFile
import cPickle as pickle
import os
import random
import tldextract
from dga_genarator import corebot, banjori, cryptolocker, dircrypt, kraken, lockyv2, pykspa, \
    qakbot, ramdo, ramnit, simda


ALEXA_1M_DOMAIN = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'
DATA_GENERATED = './data_200k_gen/data_training.pkl'

def gen_dga_malicious_domain(num_domain_per_dga=10000):

    domains = []
    labels = []

    # banjori generator
    banjori_seeds = ['quangcao', 'raovat', 'tusuong', 'quangcao1234', 'trungthuong',
                     'vietlott', 'laptop', 'laptoppc', 'malwareisbad', 'trungthuongdedang',
                     'tietkiem', 'bitcoin', 'footballchampion', 'football', 'banhang',
                     'banhangtrugthuong', 'taitro', 'muaban24', 'dacap', 'kinhdoanh',
                     'dienthoaicu', 'dienthoai', 'maytinh', 'gameonline',
                     'luadao', 'timvieclam', 'hondacivic', 'toyotaprius',
                     'muabannhadat', 'muavaban', 'dienthoaiapple', 'vovgiaothong', 'mangxahoi',
                     'tinhot', 'tacduong', 'sanbay', 'dienthoaigiare', 'muanha',
                     'muadat', 'lacduong', 'hanoi765', 'vietnam8765', 'hochiminh34567',
                     'namdinh', 'gamcau', 'sandienthoai', 'duhoc', 'hocbonggiare',
                     'giarenhat', 'muabandienthoai', 'dienthoaisamsung', 'raovathanghoa']

    segment_size = max(1, num_domain_per_dga / len(banjori_seeds))
    for banjori_seed in banjori_seeds:
        domains += banjori.generate_domains(segment_size, banjori_seed)
        labels += ['banjori'] * segment_size


    # corebot generator
    domains += corebot.generate_domains(num_domain_per_dga)
    labels += ['corebot'] * num_domain_per_dga

    # cryptolocker generator
    length_cryptolocker_domain = range(8, 32)
    segment_size = max(1, num_domain_per_dga / len(length_cryptolocker_domain))
    for crypto_length in length_cryptolocker_domain:
        domains += cryptolocker.generate_domains(segment_size,
                                                 seed_num=random.randint(1, 1000000),
                                                 length=crypto_length)
        labels += ['cryptolocker'] * segment_size

    # dircrypt generator
    domains += dircrypt.generate_domains(num_domain_per_dga)
    labels += ['dircrypt'] * num_domain_per_dga

    # kraken generator
    num_kraken_generated_domain = max(1, num_domain_per_dga / 2)
    domains += kraken.generate_domains(num_kraken_generated_domain, datetime(2017, 5, 12), 'a', 3)
    domains += kraken.generate_domains(num_kraken_generated_domain, datetime(2017, 5, 13), 'b', 3)
    labels += ['kraken'] * num_kraken_generated_domain * 2

    # lockyv2 generator
    num_locky_gen = max(1, num_kraken_generated_domain/11)
    for i in range(1, 12):
        domains += lockyv2.generate_domains(num_locky_gen, config=i)
        labels += ['locky'] * num_locky_gen

    # pykspa generator
    domains += pykspa.generate_domains(num_domain_per_dga, datetime(2017, 5, 12))
    labels += ['pykspa'] * num_domain_per_dga

    # qakbot generator
    domains += qakbot.generate_domains(num_domain_per_dga, tlds=[])
    labels += ['qakbot'] * num_domain_per_dga

    # ramdo generator
    ramdo_lengths = range(8, 32)
    segment_size = max(1, num_domain_per_dga / len(ramdo_lengths))
    for ramdo_length in ramdo_lengths:
        domains += ramdo.generate_domains(segment_size,
                                          seed_num=random.randint(1, 1000000),
                                          length=ramdo_length)
        labels += ['ramdo'] * segment_size

    # ramnit generator
    domains += ramnit.generate_domains(num_domain_per_dga, seed=0x17121994)
    labels += ['ramnit'] * num_domain_per_dga

    # simda generator
    simda_lengths = range(8, 32)
    segment_size = max(1, num_domain_per_dga/len(simda_lengths))
    for simda_length in simda_lengths:
        domains += simda.generate_domains(segment_size, length=simda_length,
                                          tld=None, base=random.randint(2, 2**32))
        labels += ['simda'] * segment_size

    return domains, labels

def get_alexa_domain(num, address=ALEXA_1M_DOMAIN, filename='top-1m.csv' ):

    url = urlopen(address)
    zipfile = ZipFile(StringIO(url.read()))
    alexa_domain = [tldextract.extract(x.split(',')[1]).domain for x in zipfile.read(filename).split()[:num]]
    return alexa_domain

def gen_training_data(gate=False):

    if gate or (not os.path.isfile(DATA_GENERATED)):
        domains, labels = gen_dga_malicious_domain()

        domains += get_alexa_domain(len(domains))
        labels += ['legit'] * len(labels)

        pickle.dump(zip(domains, labels), open(DATA_GENERATED, 'wb'))

def get_training_data(gate=False):

    gen_training_data(gate=gate)
    return pickle.load(open(DATA_GENERATED, mode='rb'))
