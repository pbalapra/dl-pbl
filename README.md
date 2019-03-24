## Fast domain-aware neural network emulation of a planetary boundary layer parameterization in a numerical weather forecast model

### Jiali Wang, Prasanna Balaprakash, and Rao Kotamarthi

Environmental Science Division, Mathematics and Computer Science Division, Argonne National Laboratory, 9700 South Cass Avenue, Argonne IL 60439, USA

### Abstract
Parameterizations for physical processes in weather/climate models are computationally expensive. We use model output from a set of simulations performed using the Weather Research Forecast (WRF) model to train deep neural networks and evaluate if trained models can provide an accurate alternative to the physics-based parameterizations. Specifically, we develop an emulator using deep neural networks for a planetary boundary layer (PBL) parameterization in the WRF model. PBL parameterizations are commonly used in atmospheric models to represent the diurnal variation of the formation and collapse of the atmospheric boundary layer ― the lowest part of the atmosphere. The dynamics of the atmospheric boundary layer, mixing and turbulence within the boundary layer, velocity, temperature and humidity profiles are all critical for determining many of the physical processes in the atmosphere. PBL parameterizations are used to represent these processes that are usually unresolved in a typical numerical weather model that operates at horizontal spatial scales in the 10’s of kilometers. We demonstrate that a domain-aware deep neural network, which takes account of underlying domain structure that are locality-specific (e.g., terrain, spatial dependence vertically), can successfully simulate the vertical profiles within the boundary layer of velocities, temperature and water vapor over the entire diurnal cycle. We then assess the spatial transferability of the domain-aware neural networks by using a trained model from one location to nearby locations. Results show that a single trained model from a location over Midwestern USA, produces predictions of wind components, temperature and water vapor profiles over the entire diurnal cycle and all nearby locations with errors less than a few percent when compared to the WRF simulations used as the training dataset.

### Data
The data we use in this study is output from a regional climate model, WRF version 3.3.1. WRF is a fully compressible, nonhydrostatic, regional numerical prediction system with proven suitability for a broad range of applications. The WRF model configuration and evaluations can be found in Wang and Kotamarthi (2014). There was 38 vertical layers cover all of the troposphere, between the surface to approximately 16 km (100 hPa). The lowest 17 layers covers from the surface to about 2km above the ground. The PBL parameterization we used for this particular WRF simulation is known as the YSU scheme (Yonsei University; Hong et al., 2006). The YSU scheme uses a nonlocal-mixing scheme with an explicit treatment of entrainment at the top of the boundary layer; and a first-order closure for the Reynolds-averaged turbulence equations of momentum of air within the PBL. 
We use the output of WRF model driven by NCEP-R2 for the period of 1984-2005. The 22-year data was partitioned into three parts: a training set consisting of 20 years (1984-2003) of 3hourly data to train the NN; a development set (also called validation set) consisting of 1 year (2004) of 3-hourly data used for tuning algorithm’s hyper-parameters and to control over-fitting (the situation where the trained network predicts quite well on the training data but not on the test data); and a test set consisting of 1 year records (2005) for prediction and evaluations. The goal of the work described here is to develop a NN based parameterization that can be used to replace the PBL parameterization in the WRF model. Thus, we expect the NN sub-model to receive a set of inputs that are equivalent to the inputs provided to the YSU scheme at each timestep. However, a key difference in our approach is that the vertical profiles of various state variables are reconstructed by the NN using only the inputs (near surface variables and 700 hPa geostrophic winds). 


The major inputs are near surface characteristics including: near surface water vapor, near surface air temperature, near surface zonal and meridional wind, ground heat flux, incoming shortwave radiation, incoming longwave radiation, PBL height, sensible heat flux, latent heat flux, surface friction velocity, ground temp, soil temperature at 2m below the ground, soil moisture at 0-0.3cm below the ground, and geostrophic wind component at 700hPa. The major outputs for NN architecture are the vertical profiles (or vectors) of the following model prognostic and diagnostic fields: temperature, water vapor mixing ratio, zonal and meridional wind. In this study we develop NN emulation of PBL parameterization, hence we only focus on predicting the profiles within the PBL layer, which is on average around 200 m during the night of winter, and around 400 m during the afternoon of winter and during the night of summer; and around 1300 m during the afternoon of summer for the locations studied here. The mid and upper troposphere (all layers above the PBL) are considered fully resolved by the dynamics simulated by the model and hence not parameterized. 

### Deep neural networks for PBL parameterization emulation

We consider three variants of DNN (see below). We construct all of them using a neural block that comprises a dense neural layer with N nodes and a rectified linear activation function, where N is user-defined parameters.

#### Naïve DNN:
Deep feed forward neural network (FFN): This is a fully connected feed forward deep neural network constructed as a sequence of K neural blocks, where the input of the ith neural block is from i-1th block, and the output of the ith neural block is given as the input of the i+1th neural block. The sizes of the input and output neural layers are 16 (=near surface variables) and 85 (= 17 vertical levels × 5 output variables). See Figure 1a for an illustration. 

![alt text](https://github.com/pbalapra/dl-pbl/blob/master/images/pbl_fnn.pdf.jpg "FNN")


#### Domain-aware DNN:
While the FFN is a typical way of applying network for finding the non-linear relationship between input and output, a key drawback of the naïve FFN is that it does not consider the underlying PBL domain structure such as the patterns that are locality-specific and the vertical dependence between different vertical levels of each profile. This is not typically needed for neural networks in general and, in fact, is usually avoided. That is because for classification and regression one seeks filters that activate when they find visual features irrespective to their location. For example, a picture can be classified as a certain object even when that object has never appeared in the given position in the training set. In our case, however, location is fixed and represents specific positions in the domain. Consequently, we want to learn the particular influence of location in the forecast. For example, topography plays a role in precipitation and can help refine the output. One could use a fully connected layer to achieve the same result. This dependence may inform the NN and provide better accuracy and data efficiency. To that end, we develop two variants of domain-aware DNNs for PBL emulation. 

##### Hierarchically connected network with previous layer only connection (HPC) 

We assume that the outputs at each altitude level depend not only on the 16 near-surface variables but also on the altitude level below it. To model this explicitly, we develop a domain-aware DNN variant in which 17 neural block are connected as follows: the input to an ith (i>1) neural block comprises input neural layer of 16 near-surface variables and the 5 outputs of the i-1th neural block. The first neural block, which is next to the input layer, receives only inputs from the input neural layer of 16 near-surface variables. See Figure 1b for an example. 

![alt text](https://github.com/pbalapra/dl-pbl/blob/master/images/pbl_hpc.pdf.jpg "HPC")


##### Hierarchically connected network with all previous layers connection (HAC) 

We assume that the outputs at each PBL depend not only on the 16 near-surface variables but also on all altitude levels below it. The input to an ith neural block comprises input neural layer of 16 near-surface variables and all outputs of the {1, 2,…, i-1} neural blocks below it. See Figure 1c for an example. 


![alt text](https://github.com/pbalapra/dl-pbl/blob/master/images/pbl_hac.pdf.jpg "HAC")


### Key software dependencies 

* Python==3.6
* Keras==2.0.8 
* tensorflow==1.3.0
* scikit-learn==0.19.1
* pandas==0.20.3 

### Directory structure
```
dl-wrf-kansas/
    Code to compare FNN, HPC, HAC on Logan, KS data for differnet number of training years. 

dl-wrf-kansas-transfer-learning/
    Code to assess the spatial transferability of the domain-aware neural networks (specifically HAC and HPC) by using a trained model from one location (at Logan, KS as presented above) to other locations within 800 kilometers from the Kansas location with different terrain conditions and vegetation types. We choose ten locations, among which two are (Sites 1 and 2) 300 km away from Logan site; three are (Sites 3, 4, and 5) 430 km away from Logan site; and five are (Sites 6 to 10) 450-800 km away from Logan site, with Sites 9 and 10 the furthest and having the most different elevations from Logan site. 
```

### Running experiments
```
dl-wrf-kansas/
    for i in 55448 43775 29190 14601 1
        do
            python code/wrfmodel.py --model_type=hpc --start_id=$i --optimizer=adam --learning_rate=0.001 --epochs=1000 --batch_size=64

            python code/wrfmodel.py --model_type=hac --start_id=$i --optimizer=adam --learning_rate=0.001 --epochs=1000 --batch_size=64

            python code/wrfmodel.py --model_type=mlp --start_id=$i --optimizer=adam --learning_rate=0.001 --epochs=1000 --batch_size=64
        done

dl-wrf-kansas-transfer-learning/

    for i in 55448 43775 29190 14601 1
        do
            python code/wrfmodel.py --model_type=hpc --start_id=$i --optimizer=adam --learning_rate=0.001 --epochs=1000 --batch_size=64

            python code/wrfmodel.py --model_type=hac --start_id=$i --optimizer=adam --learning_rate=0.001 --epochs=1000 --batch_size=64

            python code/wrfmodel.py --model_type=mlp --start_id=$i --optimizer=adam --learning_rate=0.001 --epochs=1000 --batch_size=64
        done

```
