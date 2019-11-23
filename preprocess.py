import os
import numpy as np
import pandas as pd
import cv2 as cv

class prep:
  def print_hello(self,h):
    print(h)
    
  def draw_rect(self,img,time_series,rect_value):
    START_X = rect_value[0]
    START_Y = rect_value[1]
    WIDTH = rect_value[2]
    HEIGHT = rect_value[3]
    DELTA_X = WIDTH + rect_value[4]
    DELTA_Y = HEIGHT + rect_value[5]
    pos = ['A','B','C','D','E']
    for i in range(10):
      for j in range(5):
        x = START_X+(DELTA_X*i)
        y = START_Y+(DELTA_Y*j)
        cv.rectangle(img,(x,y),(x+WIDTH,y+HEIGHT),(0,255,0),1)
        if(pos[j]+str(i+1) not in time_series):
          time_series[pos[j]+str(i+1)] = [(img[y:y+HEIGHT,x:x+WIDTH],)]
        else:
          time_series[pos[j]+str(i+1)].append((img[y:y+HEIGHT,x:x+WIDTH],))

  def get_time_series(self,case,rect_value):
    # read path
    path_dir = '/content/gdrive/My Drive/Year4/Project/top_view/'+str(case)+'/'
    # path_dir = '/top_view/'+str(case)+'/'
    print('getting...')
    path = []
    for p in os.listdir(path_dir):
      path.append(p)
      path.sort()
    print(path[0:5])
    # load data
    data = []
    for p in path:
      img = cv.imread(path_dir+'/'+p)
      shape = img.shape
      img_resize = cv.resize(img,(int(shape[1]/2),int(shape[0]/2)))
      data.append(img_resize)

    # clean data
    clean_data = []
    for i in range(len(data)):
      print('.',end='')
      img = cv.fastNlMeansDenoisingColored(data[i],None,10,10,7,21)
      gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
      eq = cv.equalizeHist(gray)
      clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
      cl1 = clahe.apply(eq)
      clean_data.append(cl1)
    print(case,'complete')

    # draw rect
    rect = []
    series = dict()
    for i in range(len(clean_data)):
      tmp = np.array(clean_data[i])
      self.draw_rect(tmp,series,rect_value)
      rect.append(tmp)

    # threshold
    for key in series:
      for i in range(len(series[key])):
        tmp = np.array(series[key][i][0])
        if(np.mean(tmp) > 50):
          tmp = cv.convertScaleAbs(tmp, alpha=1, beta=-(np.mean(tmp)-50))
        bll = cv.medianBlur(tmp,3)
        _, th1 = cv.threshold(bll, 70, 255,cv.THRESH_BINARY)
        series[key][i]  = series[key][i] + (th1,np.mean(th1))

    # calculate time series
    time_series = dict()
    for key in series:
      arr_diff = []
      arr_white = []
      arr_black = []
      arr = []
      for i in range(len(series[key])-1):
        # add diff
        diff = np.count_nonzero(series[key][i+1][1] - series[key][i][1])
        arr_diff.append(diff)
        # add white
        white = np.count_nonzero(series[key][i][1])
        arr_white.append(white)
        # add black
        black = np.count_nonzero(series[key][i][1] == 0)
        arr_black.append(black)
      arr.append(arr_diff)
      arr.append(arr_white)
      # arr.append(arr_black)
      time_series[key] = arr
    return time_series