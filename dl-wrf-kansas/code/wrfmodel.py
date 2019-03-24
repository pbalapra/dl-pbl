import sys
import os
import time
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot
from keras.utils import plot_model
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
print(here)
root = os.path.dirname(here)
print(root)

start = time.time()
import numpy as np
import pandas as pd
np.random.seed(10)

from numpy import concatenate
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.pipeline import Pipeline
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras import backend as K
from importlib import reload
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import layers
import keras_cmdline
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import os.path

load_time = time.time() - start
print(f"module import time: {load_time:.3f} seconds")

fpath = os.path.dirname(os.path.abspath(__file__))
print(fpath)

def defaults():
    def_parser = keras_cmdline.create_parser()
    def_parser = augment_parser(def_parser)
    return vars(def_parser.parse_args(''))

class SurrogateModel():
    def __init__(self, param_dict):
        self.set_keras_backend("tensorflow")
        self.plots_dir = '%s/plots/' % (root)
        self.results_dir = '%s/results/' % (root)
        self. param_dict =  param_dict
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        self.callbacks_list = [self.early_stopping]
        self.build_time = 0.0
        self.training_time = 0.0
        self.inference_time = 0.0
        self.read_params()
        if self.check_results_file():
            print('results file exists')
            return None
        self.load_data()  
        self.preprocess_data()
        self.split_data()   
        self.build_model()
        #self.draw_model()
        self.train_model()
        #sys.exit(0)
        
        self.test_model()
        self.plot_save_results()
        self.return_result()

    def set_keras_backend(self, backend):
        if K.backend() != backend:
            os.environ['KERAS_BACKEND'] = backend
            reload(K)
            assert K.backend() == backend
    
    def check_results_file(self):
        status =  False
        print(self.tag)
        fname = self.results_dir+self.tag+'_predicted.csv'
        print(fname)
        if os.path.isfile(fname):
            status =  True
        return status

    def read_params(self):
        default_params = defaults()
        for key in default_params:
            if key not in self.param_dict:
                self.param_dict[key] = default_params[key]
        self.optimizer = keras_cmdline.return_optimizer(param_dict)       
        self.model_type = self.param_dict['model_type']
        self.station_id = self.param_dict['sid']
        self.batch_size = self.param_dict['batch_size']
        self.hidden_size = self.param_dict['nhidden']
        self.nunits = self.param_dict['nunits']
        self.dropout = self.param_dict['dropout']
        self.start_id = self.param_dict['start_id'] - 1
        self.tag = '{}-{:03d}-{:05d}'.format(self.model_type, self.station_id, self.start_id)
        self.epochs = self.param_dict['epochs']     


    def load_data(self):
        inp_df = pd.read_csv(root+'/data/Input_038_1984-2005.csv', sep=r",", header=None, engine='python')
        out_df = pd.read_csv(root+'/data/Output_038_1984-2005.csv', sep=r",", header=None, engine='python')
        inp_df = inp_df.iloc[:,range(16)]
        print(inp_df.shape)
        print(out_df.shape)
        print(inp_df.head())

        self.rlevels = 17 # we need 0 to 16 layers
        self.nlevels = 17 #out_df1.shape[1]
        self.nseries = int(out_df.shape[1]/self.nlevels)
        self.tlevels = out_df.shape[1]
        indices = range(self.tlevels)

        res = []
        for i in range(self.rlevels):
            for j in range(self.nseries):
                res.append(indices[i+(self.nlevels*j)])  
        print(res)
        print(len(res))
        self.req_outputs = res
        self.req_outputs = [int(x) for x in self.req_outputs]
        print(self.req_outputs)

        # load dataset
        dataset = out_df
        print(dataset.shape)
        values = dataset.values
        print(values.shape)

        out_df_sel = out_df.iloc[:,self.req_outputs]
        self.n_input = inp_df.shape[1]
        self.n_output = out_df_sel.shape[1]

        dataset = pd.concat([inp_df, out_df_sel], axis=1)
        print(dataset.shape)
        values = dataset.values
        # ensure all data is float
        self.req_data = values.astype('float32')

    def preprocess_data(self):
        # normalize features
        self.preprocessor = Pipeline([('stdscaler', StandardScaler()), ('minmax', MinMaxScaler(feature_range=(0, 1)))])
        self.preproc_data = self.preprocessor.fit_transform(self.req_data)

    def split_data(self):
        # split into train and test sets
        values = self.preproc_data
        train_idx = range(self.start_id, 58365-1)
        val_idx = range(58365, 61290-1)
        test_idx = range(61290, 64207)
        
        train = values[train_idx, :]
        test = values[test_idx, :]
        validation = values[val_idx, :]    

        # split into input and outputs
        self.train_X, self.train_y = train[:, range(self.n_input)], train[:, range(self.n_input, self.n_input+self.n_output)]
        self.validation_X, self.validation_y = validation[:, range(self.n_input)], validation[:, range(self.n_input, self.n_input+self.n_output)]
        self.test_X, self.test_y = test[:, range(self.n_input)], test[:, range(self.n_input, self.n_input+self.n_output)]
        print(self.train_X.shape, self.train_y.shape, self.validation_X.shape, self.validation_X.shape, self.test_X.shape, self.test_y.shape)
        self.ntrain = self.train_X.shape[0]
        self.ntest = self.test_X.shape[0]
        self.nval = self.validation_X.shape[0]
        #self.tag = '{}-{:05d}'.format(self.tag, self.ntrain)

    def build_model(self):
        if self.model_type == 'hpc':
            self.build_model_hc(type_conn='hpc')
        elif self.model_type == 'hpc_staged':
            self.build_model_hc_staged(type_conn='hpc')
        elif self.model_type == 'hac':
            self.build_model_hc(type_conn='hac')
        elif self.model_type == 'hac_staged':
            self.build_model_hc_staged(type_conn='hac')
        elif self.model_type == 'mlp':
            self.build_model_mlp()
        else:
            print('%s not implemented' % model_type)
            sys.exit(0)
        self.model.summary()
        #sys.exit(0)

    def draw_model(self):
        filename=self.plots_dir+self.tag+'_model_plot.png'
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)

    def build_model_hc(self, type_conn):
        start = time.time()
        input_shape = self.train_X.shape[1:] 
        x = Input(shape=input_shape)
        inputs = x
        for j in range(self.hidden_size):
            x = Dense(self.nunits, activation='relu')(x)
            x = Dropout(self.dropout)(x)
        x = Dense(self.nunits, activation='relu')(x)
        level_0 = Dense(self.nseries)(x)
        prev = level_0
        outputs_list = []
        outputs_list.append(prev) 
        for i in range(self.rlevels-1):
            if type_conn == 'hpc':
                prev_conn = [prev]
            elif type_conn == 'hac':
                prev_conn = outputs_list
            else:
                print('error; exit')
                sys.exit(0)
            x = concatenate([inputs] + prev_conn)
            for j in range(self.hidden_size):
                x = Dense(self.nunits, activation='relu')(x)
                x = Dropout(self.dropout)(x)
            x = Dense(self.nunits, activation='relu')(x)
            prev = Dense(self.nseries)(x)
            outputs_list.append(prev)
        self.model = Model(inputs=inputs, outputs=concatenate(outputs_list))
        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae'])
        self.model.summary()
        end = time.time()
        self.build_time = self.build_time + (end-start)

    def build_model_hc_staged(self, type_conn):
        build_time = 0
        train_time = 0
        cntr = 1
        start = time.time() 
        input_shape = self.train_X.shape[1:]
        x = Input(shape=input_shape)
        inputs = x
        for j in range(self.hidden_size):
            x = Dense(self.nunits, activation='relu')(x)
            x = Dropout(self.dropout)(x)
        x = Dense(self.nunits, activation='relu')(x)
        level_0 = Dense(self.nseries, name='level_%02d'%(cntr))(x)
        prev = level_0
        outputs_list = []
        outputs_list.append(prev) 
        #model = Model(inputs=inputs, outputs=outputs_list)
        model = Model(inputs=inputs, outputs=prev)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae'])
        model.summary()
        end = time.time() 
        self.buid_time = self.build_time + (end - start)
        start_index = (cntr-1)*self.nseries
        end_index = cntr*self.nseries
        start = time.time()
        history = model.fit(self.train_X, self.train_y[:,range(start_index, end_index)], 
                    validation_data=(self.validation_X, self.validation_y[:,range(start_index, end_index)]), 
                    callbacks=self.callbacks_list, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        end = time.time()
        self.training_time = self.training_time + (end - start)
        for layer in model.layers:
            layer.trainable = False
        for i in range(self.rlevels-1):
            cntr = cntr + 1
            start = time.time()
            if type_conn == 'hpc':
                prev_conn = [prev]
            elif type_conn == 'hac':
                prev_conn = outputs_list
            else:
                print('error; exit')
                sys.exit(0) 
            x = concatenate([inputs] + prev_conn)
            for j in range(self.hidden_size):
                x = Dense(self.nunits, activation='relu')(x)
                x = Dropout(self.dropout)(x)
            x = Dense(self.nunits, activation='relu')(x)
            prev = Dense(self.nseries, name='level_%02d'%(cntr))(x)
            outputs_list.append(prev)
            model = Model(inputs=inputs, outputs=prev)
            model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae'])
            model.summary()
            end = time.time() 
            self.buid_time = self.build_time + (end - start)
            start_index = (cntr-1)*self.nseries
            end_index = cntr*self.nseries
            start = time.time()
            history = model.fit(self.train_X, self.train_y[:,range(start_index, end_index)],
                        validation_data=(self.validation_X, self.validation_y[:,range(start_index, end_index)]), 
                        callbacks=self.callbacks_list, epochs=self.epochs, batch_size=self.batch_size, 
                        verbose=1)
            end = time.time()
            self.training_time = self.training_time + (end - start)
            for layer in model.layers:
                layer.trainable = False
        start = time.time()
        model = Model(inputs=inputs, outputs=concatenate(outputs_list))
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae'])
        model.summary()
        end = time.time() 
        self.buid_time = self.build_time + (end - start)
        self.model = model

    def build_model_mlp(self):
        start = time.time()
        input_shape = self.train_X.shape[1:]
        inputs = Input(shape=input_shape)
        x = Dense(self.nunits, activation='relu')(inputs)
        x = Dropout(self.dropout)(x)
        for j in range(self.hidden_size*self.rlevels-1):
            x = Dense(self.nunits, activation='relu')(x)
            x = Dropout(self.dropout)(x)
        level_all = Dense(self.train_y.shape[1])(x)
        self.model = Model(inputs=inputs, outputs=level_all)
        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae'])
        #self.model.summary()
        end = time.time()
        self.build_time = self.build_time + (end-start)

    def train_model(self):
        filepath= self.results_dir+self.tag+'_weights.best.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        self.model.summary()
        start = time.time()
        self.history = self.model.fit(self.train_X, self.train_y, validation_data=(self.validation_X, self.validation_y), 
                                        callbacks=self.callbacks_list, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        end = time.time()
        self.training_time = self.training_time + (end - start)

    def test_model(self):
        start = time.time()
        self.yhat = self.model.predict(self.test_X)
        end = time.time()
        self.inference_time = self.inference_time + (end - start)
        #print(self.yhat)
        print(self.yhat.shape)

    def compute_metric(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred) 
        evs = explained_variance_score(y_true, y_pred) 
        mae = mean_absolute_error(y_true, y_pred)
        return r2, evs, mae

    def geo_mean(self, iterable):
        a = np.array(iterable)
        return a.prod()**(1.0/len(a))

    def compute_tuning_metric(self):
        r2_vals = []
        for level in range(1, self.rlevels):
            start = (5*level) - 5
            end = (5*level)
            groups = range(start,end)
            for group in reversed(groups):
                #print(group)
                r2, evs, mae = self.compute_metric(self.test_y[:, group], self.yhat[:, group])
                r2_vals.append(r2)
                #print(r2)
        print(r2_vals[-5:])
        res = self.geo_mean(r2_vals[-5:])
        return res

    def return_result(self):
        #self.diff = self.test_y.ravel() - self.yhat.ravel()
        #output = np.percentile(abs(self.diff), 95)
        print(self.test_y.shape)
        print(self.yhat.shape)
        output = self.compute_tuning_metric()
        if np.isnan(output):
            output = -100.0
        print('OUTPUT: %f'% -output)
        return output

    def plot_save_results(self):
        # plot history
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='validation')
        pyplot.legend()
        pyplot.savefig(self.plots_dir+self.tag+'_train_loss.png', bbox_inches='tight')
        pyplot.close()

        # plot history
        pyplot.plot(self.history.history['mean_absolute_error'], label='train')
        pyplot.plot(self.history.history['val_mean_absolute_error'], label='validation')
        pyplot.legend()
        pyplot.savefig(self.plots_dir+self.tag+'_train_error.png', bbox_inches='tight')
        pyplot.close()

        i = 1
        diff_results = []
        groups = range(self.train_y.shape[1])
        for group in groups:
            diff = self.test_y[:, group] - self.yhat[:, group]
            diff.ravel()
            diff_results.append(diff)
            i += 1

        req_indices = np.argsort(self.req_outputs).tolist()
        print(req_indices)
        reorder_list = [ diff_results[i] for i in req_indices]
        pyplot.figure(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
        pyplot.boxplot(reorder_list, notch=True, patch_artist=True)
        pyplot.ylim(-0.2, 0.2)
        pyplot.grid(True)
        pyplot.savefig(self.plots_dir+self.tag+'_bwplot.png', bbox_inches='tight')
        pyplot.close()

        for level in range(1, self.rlevels):
            start = (5*level) - 5
            end = (5*level)
            groups = range(start,end)
            i = 1
            pyplot.figure(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
            for group in reversed(groups):
                pyplot.subplot(len(groups), 1, i)
                pyplot.plot(self.test_y[:, group], label='observed')
                pyplot.plot(self.yhat[:, group], label='predicted')
                r2, evs, mae = self.compute_metric(self.test_y[:, group], self.yhat[:, group])
                pyplot.title('level = %d; r2 = %1.3f; evs = %1.3f; mae = %1.3f' % (level, r2, evs, mae))
                pyplot.ylim(0, 1)
                pyplot.legend()
                i += 1
            pyplot.subplots_adjust(hspace=1, wspace=1)
            pyplot.savefig(self.plots_dir+self.tag+'_trend_%d.png'%level, bbox_inches='tight')
            pyplot.close()

        print(self.test_y.shape)
        print(self.yhat.shape)
        print(self.test_X.shape)

        test_pred = np.concatenate((self.test_X, self.yhat), axis=1)
        test_orig = np.concatenate((self.test_X, self.test_y), axis=1)
        test_y_trans = self.preprocessor.inverse_transform(test_pred)
        yhat_trans = self.preprocessor.inverse_transform(test_orig)


        obsr_df = pd.DataFrame(test_y_trans)
        pred_df = pd.DataFrame(yhat_trans)
        
        meta_df = pd.DataFrame()
        meta_df['model_type'] = [self.model_type]  
        meta_df['station_id'] = [self.station_id]  
        meta_df['batch_size'] = [self.batch_size]  
        meta_df['hidden_size'] = [self.hidden_size]  
        meta_df['nunits'] = [self.nunits] 
        meta_df['dropout'] = [self.dropout]  
        meta_df['start_id'] = [self.start_id]  
        meta_df['tag'] = [self.tag]  
        meta_df['epochs'] = [self.epochs]  
        meta_df['ntrain'] = [self.ntrain] 
        meta_df['ntest'] = [self.ntest] 
        meta_df['nval'] = [self.nval] 
        meta_df['build_time'] = [self.build_time]
        meta_df['training_time'] = [self.training_time]
        meta_df['inference_time'] = [self.inference_time] 

        pred_df.to_csv(self.results_dir+self.tag+'_predicted.csv', index=False)
        obsr_df.to_csv(self.results_dir+self.tag+'_observed.csv', index=False)
        meta_df.to_csv(self.results_dir+self.tag+'_meta.csv', index=False)

def augment_parser(parser):
    parser.add_argument('--nhidden', action='store', dest='nhidden',
                        nargs='?', const=2, type=int, default='2',
                        help='number of hidden layers')
    parser.add_argument('--nunits', action='store', dest='nunits',
                        nargs='?', const=2, type=int, default='16',
                        help='number of units per hidden layer')
    parser.add_argument('--sid', action='store', dest='sid',
                        nargs='?', const=2, type=int, default='2',
                        help='station id')
    parser.add_argument('--start_id', action='store', dest='start_id',
                        nargs='?', const=2, type=int, default='1',
                        help='start id')
    parser.add_argument('--model_type', action='store',
                        dest='model_type',
                        nargs='?', const=1, type=str, default='hpc',
                        choices=['hpc', 'hpc_staged', 'hac', 'hac_staged', 'mlp'],
                        help='type of model')
    return parser


if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    SurrogateModel(param_dict)
