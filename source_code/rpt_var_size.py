
# coding: utf-8

# In[5]:


## rpt_var_size()
## Author: David Yin 
## Contact: yinzhen@stanford.edu
## Date: Nov 03, 2018

### The functions evaluates the size of all the defined variables 
import sys

def rpt_var_size():
    def sizeof_fmt(num, suffix='B'):
        ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))

