import pickle

class DataCollector:
  _instance = None

  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
    return cls._instance
  
  def __init__(self):
    self.name = None
    self.data = {}
  
  def setNextDataName(self, name):
    self.name = name
  
  def setData(self, data, name):
    if name:
      self.data[name] = data
    else:
      assert self.name, "no Name for data"
      self.data[self.name] = data
  
  def export(self, ofname):
    pickle.dump(self.data, ofname)
  
  def load(self, ifname):
    data = pickle.load(ifname)
    self.data = data