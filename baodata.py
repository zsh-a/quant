import pickle

c = pickle.loads(open('model_v3.0.pkl','rb').read())
print(c)