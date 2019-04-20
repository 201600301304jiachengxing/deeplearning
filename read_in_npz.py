# coding: utf-8
import numpy as np
import pickle

f = open('char-rnn-snapshot.pkl','rb')
a = pickle.load(f,encoding='bytes')
Wxh = a[b"Wxh"]
Whh = a[b"Whh"]
Why = a[b"Why"]
bh = a[b"bh"]
by = a[b"by"]
mWxh, mWhh, mWhy = a[b"mWxh"], a[b"mWhh"], a[b"mWhy"]
mbh, mby = a[b"mbh"], a[b"mby"]
chars, data_size, vocab_size, char_to_ix, ix_to_char = a[b"chars"].tolist(), \
                                                       a[b"data_size"].tolist(), \
                                                       a[b"vocab_size"].tolist(), \
                                                       a[b"char_to_ix"].tolist(), \
                                                       a[b"ix_to_char"].tolist()

def decoder(h, seed_ix, n, temp=1):
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    y = (y - np.max(y,axis=0)) / temp
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

def encoder(h, inputs):
  for t in range(len(inputs)):
    x = np.zeros((vocab_size, 1))
    x[inputs[t]] = 1
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
  return h


hidden_size = 250
def sample():
  data = str(open('samples.txt', 'rb').read())
  inputs = [char_to_ix[str.encode(seg)] for seg in data if str.encode(seg) in char_to_ix.keys()]
  h = np.zeros((hidden_size, 1))
  h = encoder(h, inputs)
  s = decoder(h, inputs[-1], 400, temp=1)
  txt = ''.join(bytes.decode(ix_to_char[ix]) for ix in s)
  string = " irst incitly dist of guwer ollome to thee voite, es you she! hands.\n\nMENENIUS:\nI geon, with dity with it him how ed so have doldeus?\n\nCAUCIIUS:\nThe so cowar to hits dat: thes! in, now toke love.\n\nCORI \n----\n\n\n----\n irsham!\nages! Who shere Ede,\nTo preed as o shall detwake it to gups your.\n\nMENENIUS:\nHes, the in.\n\nCORIOLANUS:\nThe say a kees,\nDeaven ager he,\nAnd out, my a Cixice?--'ll and mabt Mase,\nMore merse.\n\nVI \n----\n\n\n----\n irst Conbed you fle\nCor sane, on;\nI: and caem as on owat me love,\nThy for will foldingain be sun dik, hil this or have show wounk\n\nVOLIAIA RCININI:\nThere my che sold to-'dttand thee nepe,\nOhy on pees\n \n"

  print(string + txt)


def test1():
  sample1 = 'test of an apple:'
  inputs = [char_to_ix[str.encode(seg)] for seg in sample1 if str.encode(seg) in char_to_ix.keys()]
  h = np.zeros((hidden_size, 1))
  h = encoder(h, inputs)

  a1 = np.zeros_like(h)
  index1 = np.where(np.abs(h) > 0.5)
  a1[index1] = 1

  #s = decoder(h, inputs[-1] ,1, temp=1)

  temp = 1
  x = np.zeros((vocab_size, 1))
  x[inputs[-1]] = 1
  h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
  y = np.dot(Why, h) + by
  y = (y - np.max(y, axis=0)) / temp
  p = np.exp(y) / np.sum(np.exp(y))
  ix = np.random.choice(range(vocab_size), p=p.ravel())

  a2 = np.zeros_like(h)
  index2 = np.where(np.abs(h) > 0.5)
  a2[index2] = 1

  a3 = np.zeros_like(p)
  index3 = np.where(np.abs(p) > 0.5)
  a3[index3] = 1

  a1 = a1.reshape((-1,1))
  a2 = a2.reshape((-1,1))
  a3 = a3.reshape((-1,1))

  print(np.dot(a1, a2.T))
  print(np.dot(a2, a3.T))

  x = np.zeros((vocab_size, 1))
  x[ix] = 1

  string = " irst incitly dist of guwer ollome to thee voite, es you she! hands.\n\nMENENIUS:\nI geon, with dity with it him how ed so have doldeus?\n\nCAUCIIUS:\nThe so cowar to hits dat: thes! in, now toke love.\n\nCORI \n----\n\n\n----\n irsham!\nages! Who shere Ede,\nTo preed as o shall detwake it to gups your.\n\nMENENIUS:\nHes, the in.\n\nCORIOLANUS:\nThe say a kees,\nDeaven ager he,\nAnd out, my a Cixice?--'ll and mabt Mase,\nMore merse.\n\nVI \n----\n\n\n----\n irst Conbed you fle\nCor sane, on;\nI: and caem as on owat me love,\nThy for will foldingain be sun dik, hil this or have show wounk\n\nVOLIAIA RCININI:\nThere my che sold to-'dttand thee nepe,\nOhy on pees\n \n"

  txt = ''.join(bytes.decode(ix_to_char[ix]))
  print(string + txt)

for i in range(1):

  print(i)
  #sample()
  test1()




