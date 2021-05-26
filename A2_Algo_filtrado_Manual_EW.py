#%%
import pandas as pd
import datetime
from datetime import timedelta
import time
from numpy.lib.function_base import cov
from pandas import Timestamp
import pickle
import numpy as np
import statistics
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from warnings import simplefilter
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
simplefilter(action='ignore', category=FutureWarning)#PARA IGNORAR LOS WARNINGS DEL SKLERN
pd.options.mode.chained_assignment = None

#CARGO EL PICKLE
t0 = time.time()


data                = pd.read_pickle(r"C:\Users\Usuario\Desktop\DATA\data.p")
benchmark           = pd.read_pickle(r"C:\Users\Usuario\Desktop\DATA\benchmark.p")[:-17]
constituents_by_day = pd.read_pickle(r"C:\Users\Usuario\Desktop\DATA\constituents_by_day.p")
rics_sector         = pd.read_pickle(r"C:\Users\Usuario\Desktop\DATA\rics_sector.p")
error               = pd.read_pickle(r"C:\Users\Usuario\Desktop\DATA\error.p")
vix                 = pd.read_pickle(r"C:\Users\Usuario\Desktop\DATA\vix.p")
ventana             = 30
dates               = list(benchmark.index)
num_activos         = 150

#DF CON LOS DATOS DE COTIZACION---------------------------------------------------------------------------

#GENERA UN DF CON EL ELEMENTO QUE QUERAMOS ('OPEN','HIGH','LOW','CLOSE','COUNT','VOLUME') PARA TODOS LOS COMPONENTES DEL 
#INDICE QUE DISPONGAN DATOS. A LA FUNCION LE TENEMOS QUE PASAR:
# data:diccionario resultante de la extracción de reuters
# element: ('OPEN','HIGH','LOW','CLOSE','COUNT','VOLUME')

def get_data(data, element):

        elements  = pd.DataFrame()
        rics_with_data = []
        for i in list(data.keys()):
                try:
                        df_i = data[i][element]
                        elements = pd.concat([elements,df_i], axis=1)
                        rics_with_data.append(i)
                except TypeError:
                        pass   
        elements.columns = rics_with_data
        

        return elements

#DESCARGO DATOS 
close               = get_data(data, 'CLOSE')
high                = get_data(data, 'HIGH')
low                 = get_data(data, 'LOW')
volume              = get_data(data, 'VOLUME')

#RELLENO LOS NA
close               = close.fillna( method='ffill')
high                = high.fillna( method='ffill')
low                 = low.fillna( method='ffill')

#GENERO FRAMES DE 7 DIAS
frames              = [dates[x:x+ventana] for x in range(0, len(dates), ventana)] #LISTA CON LAS FECHAS EN FRAMES


#GENERO EL DATAFRAME QUE IRE RELLENANDO CON LAS CARACTERISTICAS
#Y EL DICCIONARIO DONDE GUARDARE LAS ALOCS

caracteristicas     = ['ret','volume','beta','volat','sharp','alpha']
alocs               = {}

#SACO LOS RETORNOS LOGARITMICOS DE LOS PRECIOS, EL VOLUMEN Y EL BENCHMArK
price_ret           = (np.log(close / close.shift(1))).astype(float)
low_ret             = (np.log(low / low.shift(1))).astype(float)
high_ret            = (np.log(high / high.shift(1))).astype(float)
vol_ret             = (np.log(volume / volume.shift(1))).astype(float)
benchmark_ret       = (np.log(benchmark['CLOSE'] / benchmark['CLOSE'].shift(1))).astype(float)
vix_ret             = (np.log(vix['CLOSE'] / vix['CLOSE'].shift(1))).astype(float)

#FUNCION QUE ME SACA LOS CLUSTERS
def apply_cluster(df, clusters):

    try:
        df.drop('cluster', axis=1, inplace=True)
    except:
        next
    X = df#.iloc[:,1:]
    rb = RobustScaler()
    X_rb = rb.fit_transform(X)
    kmeans = KMeans(n_clusters=clusters, random_state=10, n_init=10, n_jobs=-1)  
    kmeans.fit(X_rb) 
    score = metrics.silhouette_score(X_rb, kmeans.labels_, random_state=10)
    df['cluster'] = kmeans.labels_
    sse_within_cluster = kmeans.inertia_

    return df


#APLICO EL ALGORITMO
for frame in frames:


        #----------------------------------------TRABAJO POR SECTORES-----------------------------

        #ME SACO LOS RICS EN ESE FRAME:
        ric_in_frame= []
        
        for i in frame:
                ric_in_frame.append(constituents_by_day[i])

        #HAGO UNA LISTA SIMPLE QUE CONTENGA TODOS LOS RICS (CON DUPLICADOS)
        ric_in_frame                 = [item for sublist in ric_in_frame for item in sublist]

        #ME QUITO LOS DUPLICADOS Y ELIMINO EL VACIO
        ric_in_frame                 = list(set(ric_in_frame))
        ric_in_frame                 =[x for x in ric_in_frame if x not in error]
        ric_in_frame                 = [x for x in ric_in_frame if x.strip()]

        #GENERO UNA MASCARA BOOLEANA QUE UTILIZARE PARA FILTRAR
        bool_rics_in_frame           = close.columns.isin(ric_in_frame)

        #GENERO UN DF VACIO
        datos_ric                   = pd.DataFrame(index=ric_in_frame, columns = caracteristicas)

        #DATOS QUE UTILIZARE PARA SACAR CARACTERISTICAS POR SECTOR
        data_frame_sec              = close.loc[frame,bool_rics_in_frame]
        data_frame_benchmark_sec    = benchmark.loc[frame,'CLOSE']
        data_frame_volume_sec       = volume.loc[frame,bool_rics_in_frame]
        
        #AÑADO EL SECTOR AL QUE PERTENCE CADA RIC
        datos_ric['sector']         = rics_sector.loc[ric_in_frame,:].sector

        sectors_in_frame            = datos_ric.sector.unique()
        rics_sec                    = {}
        alpha_by_sector             = pd.DataFrame(index=sectors_in_frame, columns=['alpha'])
        ret_adj_by_sector           = pd.DataFrame(index=frame[1:], columns=sectors_in_frame)

        for sector in sectors_in_frame:

            #LISTA DE RICS EN EL SECTOR Y ME LO GUARDO EN UN DICCIONARIO
            rics                   = datos_ric.index[datos_ric['sector'] == sector].tolist()
            rics_sec[sector]       = rics

            #DATOS DE ESE SECTOR (OHLC Y VOLUMEN)
            sector_data_frame      = data_frame_sec.loc[frame,rics]
            sector_data_frame_vol  = data_frame_volume_sec.loc[frame,rics]

            #SACO LOS PESOS DENTRO DEL SECTOR SEGUN VOLUMEN PARA CADA DIA
            weigths                = pd.DataFrame(index=frame, columns=rics)

            for i in frame: 
                vol_date            = sector_data_frame_vol.loc[i,:]
                vol_date_total      = sector_data_frame_vol.loc[i,:].sum(axis=0)
                weigth_i            = vol_date / vol_date_total
                weigths.loc[i]      = weigth_i

            weigths                 = weigths.iloc[1:]

            #RETORNOS LOGARITMICOS DE PRECIO (CLOSE Y BENCHMARK), SIN PONDERAR POR PESO
            sector_ret              = sector_data_frame.pct_change()[1:].astype(float)
            benchmark_ret_sector    = data_frame_benchmark_sec.pct_change()[1:].astype(float)

            #RETORNO DIARIO DEL SECTOR AJUSTADO A LOS PESOS Y ME LOS GUARDO
            ret_adj_day             = sector_ret * weigths
            ret_adj                 = ret_adj_day.sum(axis=1)
            ret_adj_by_sector[sector] = ret_adj

            #RETORNOS LOGARITMICOS DEL SECTOR ENTRE EL PRIMER Y ULTIMO DIA DEL FRAME PARA EL SECTOR Y EL BENCHMARK
            frame_ret               = ret_adj.sum(axis=0)
            frame_benchmark_ret     = benchmark_ret_sector.sum(axis=0)

            #VOLATILIDAD
            volat                   = ret_adj.std()

            #RATIO DE SHARP
            sharp                   = (ret_adj.sum(axis=0) / volat)
            

            #BETA
            cov           = np.cov(ret_adj.dropna(),benchmark_ret_sector.dropna())
            beta          = cov[0,1]/cov[1,1]

            #ALPHA

            alpha = frame_ret - (frame_benchmark_ret + (beta * frame_benchmark_ret))


            alpha_by_sector.loc[sector] = alpha
            
        #OBTENGO EL LISTADO DE SECTORES ORDENADO POR ALPHA
        alpha_by_sector             = alpha_by_sector.sort_values(by='alpha',ascending=False)

        #----------------------------------------TRABAJO POR RICS---------------------------------------

        #SACO LOS DATOS PARA LOS RICS

        data_frame                 = price_ret.loc[frame,bool_rics_in_frame]
        data_frame_benchmark       = benchmark_ret.loc[frame]
        data_frame_volume          = vol_ret.loc[frame,bool_rics_in_frame]


        #SACO LOS RETORNOS LOGARITMICOS PARA EL FRAME:

        datos_ric['ret']            = data_frame.sum(axis=0)               #RETORNO LOGARITMICO DE LOS ACTIVOS
        datos_ric['volume']         = data_frame_volume.sum(axis=0)        #RETORNO LOGARITMICO DEL VOLUMN
        df_ret_benchmark            = data_frame_benchmark.sum(axis=0)

        #BETA

        for i in data_frame.columns:

                if list(np.isnan(close.loc[frame,i])).count(False) > 2 :

                        nan_bool_mask = ~np.isnan(data_frame[i])
                        cov           = np.cov(data_frame[i].dropna(),data_frame_benchmark[nan_bool_mask].dropna())
                        beta          = cov[0,1]/cov[1,1]
                        datos_ric['beta'][i] = beta


        #VOLATILIDAD
        volat                      = data_frame.std()
        datos_ric['volat']         = volat

        #RATIO DE SHARP
        datos_ric['sharp']         = (datos_ric['ret'] / datos_ric['volat'])
 

        #ALPHA DE JENSEN

        for i in datos_ric.index:

                if (datos_ric['beta'][i] == 0 and datos_ric['ret'][i] == 0):

                        continue
                else :
                        datos_ric['alpha'][i] = datos_ric['ret'][i] - (df_ret_benchmark + (datos_ric['beta'][i] * df_ret_benchmark))
        
        datos_ric                  = datos_ric[['alpha','sharp','sector']]
        datos_ric                  = datos_ric.dropna()

        #------------------------- SACO LOS 30 CON MEJOR ALPHA PARA LOS 5 MEJORES SECTORES Y LOS GUARDO EN UN DICT  ---------------------------------                

        top_30_sector              = {}
        top_30_sector_uniq         = []

        for sector in list(alpha_by_sector.index):
        
            #FILTRO LOS ALPHAS MAYORES QUE 0

            if alpha_by_sector.loc[sector][0] >= 0:
            
                top_30_rics           = list(datos_ric[datos_ric['sector']==sector].sort_values('alpha',ascending = False).index)
                top_30_sector[sector] = top_30_rics
                top_30_sector_uniq.append(top_30_rics)

            else:  next
                   
        top_30_sector_uniq        = [item for sublist in top_30_sector_uniq for item in sublist]

        #SI NO HAY NINGUNO CON ALPHA MAYOR QUE 0 NO HAY ALOC

        if len(top_30_sector_uniq) < 10:
                continue
        


        #------------------------- SACO CARACTERISTICAS  ------------------------------------------------------------------------------

        carac_DF                        = pd.DataFrame(index=top_30_sector_uniq)

        #SHARP

        carac_DF['sharp']               = datos_ric.loc[top_30_sector_uniq,'sharp'] 

        #ALPHA

        carac_DF['alpha']               = datos_ric.loc[top_30_sector_uniq,'alpha']


        #RATIO VARIACION PRECIO-VOLUMEN

        ret_price_frame                 = data_frame.sum(axis=0)[top_30_sector_uniq]
        ret_vol_frame                   = data_frame_volume.sum(axis=0)[top_30_sector_uniq]
        carac_DF['%price/%vol']         = ret_price_frame / ret_vol_frame
        carac_DF['%price/%vol']         = carac_DF['%price/%vol'].fillna(0)

        #TRUE RANGE

        #VM_utrend                        = (high.loc[frame,top_30_sector_uniq].iloc[-1] / low.loc[frame,top_30_sector_uniq].iloc[1]) -1
        #VM_downtrend                     = (low.loc[frame,top_30_sector_uniq].iloc[-1] / high.loc[frame,top_30_sector_uniq].iloc[1]) -1
        #VM_utrend_index                  = (benchmark.loc[frame,'HIGH'][-1] / benchmark.loc[frame,'LOW'].iloc[1]) -1
        #VM_downtrend_index               = (benchmark.loc[frame,'LOW'].iloc[-1] / benchmark.loc[frame,'HIGH'].iloc[1]) -1
        #carac_DF['VM_utrend']            = VM_utrend / VM_utrend_index
        #carac_DF['VM_downtrend']         = VM_downtrend / VM_downtrend_index

        #RSI

        prices_rsi                       = close.loc[frame,top_30_sector_uniq].diff()
        trading_days                     = prices_rsi.shape[0] - prices_rsi.isna().sum()
        AvgU                             = prices_rsi.apply(lambda x: x[x>0].sum(),axis=0) / trading_days
        AvgD                             = -(prices_rsi.apply(lambda x: x[x<0].sum(),axis=0) / trading_days)
        RS                               = AvgU/AvgD
        RSI                              = 100 - (100/(1+ RS))
        carac_DF['rsi']                  = RSI

        #SKEAWNESS (https://seekingalpha.com/article/4370533-impact-of-skewness-on-returns)

        carac_DF['skew']                 = data_frame.skew(axis=0)[top_30_sector_uniq]

        #KURTOSIS (https://www.investopedia.com/terms/k/kurtosis.asp#:~:text=For%20investors%2C%20high%20kurtosis%20of,the%20normal%20distribution%20of%20returns.)

        carac_DF['kurtosis']             = data_frame.kurt(axis=0)[top_30_sector_uniq]

        #------------------------------------------------------ APLICO EL CLUSTER----------------------------------

        #LIMPIO DE NA Y PASO TODO A FLOAT64
        carac_DF                         = carac_DF.dropna()
        carac_DF                       = carac_DF.astype('float64')

        #CALCULO LOS CLUSTERS POR CARACTERISTICA Y LOS ORDENO EN FUNCION DE LA CARECTIRISTICA

        top_30_carac                     = {}
        asc                              = ['kurtosis','rsi']
        desc                             = ['alpha','sharp','%price/%vol','skew']

        for i in carac_DF.columns:

                if i in asc:

                                       x = True
                else:
                                       x = False
        
                df_carac                 = pd.DataFrame(carac_DF[i])
                df_carac                 = df_carac.astype('float64')

                #APLICO CLUSTERIZACION
                df                       = apply_cluster(df_carac, 5)

                df['duplicate']          = df['cluster']

                df_carac = df.drop(columns=['duplicate'])
                cluster_df               = (df
                .groupby('cluster')
                .agg({f"{i}":"mean","duplicate":"count"})
                .sort_values(f'{i}',ascending = x)
                .reset_index())
                cluster_df.rename(columns = {'duplicate' : 'rics'}, inplace = True)
                
                
                #OBTENGO EL NUMERO DE LOS CLUSTERS QUE HE ELEGIDO

                sum = 0
                num = 0

                for p in list(cluster_df['rics']):

                    num = num + 1            
                    if sum <= 500:     
                        cluster_choose = list(cluster_df['cluster'])[0:num]
                        sum = sum + p
        
                #RICS QUE ELIGO
        
                rics_alloc               = []
        
                for k in cluster_choose:
                
                        df_cluster_choose = df[df['cluster']==k].sort_values(f"{i}",ascending = x)
                        rics_alloc.append(df_cluster_choose.index.tolist())

                rics_alloc               = [item for sublist in rics_alloc for item in sublist]
                rics_alloc               = rics_alloc[0:250]
                top_30_carac[i]          = df_carac.loc[rics_alloc]

        #-------------------------------- FILTRADO POR CARACTERISTICA -----------------------

        s1                               = set(list(top_30_carac[list(carac_DF.columns)[0]].index))
        s2                               = set(list(top_30_carac[list(carac_DF.columns)[1]].index))
        s3                               = set(list(top_30_carac[list(carac_DF.columns)[2]].index))
        s4                               = set(list(top_30_carac[list(carac_DF.columns)[3]].index))
        s5                               = set(list(top_30_carac[list(carac_DF.columns)[4]].index))
        s6                               = set(list(top_30_carac[list(carac_DF.columns)[5]].index))

        set1                             = s1.intersection(s2)
        set2                             = set1.intersection(s3)
        set3                             = set2.intersection(s4)
        set4                             = set3.intersection(s5)
        set5                             = set4.intersection(s6)

        #------------------------------- VUELVO A HACER CLUSTERS Y SACO ALOCS-----------------

        top_carac                        = carac_DF.loc[set5,:]

        if len(top_carac) < 10:
                continue

        df = apply_cluster(top_carac, 5)

        top_carac['duplicate']           = top_carac['cluster']

        cluster_df_fin                   = (top_carac
        .groupby('cluster')
        .agg({"rsi":"mean","duplicate":"count"})
        .sort_values('rsi',ascending = True)
        .reset_index())
        cluster_df_fin.rename(columns = {'duplicate' : 'rics'}, inplace = True)

        #OBTENGO EL NUMERO DE LOS CLUSTERS QUE HE ELEGIDO

        sum                                = 0
        num                                = 0

        for j in cluster_df_fin['rics']:
        
            num                            = num + 1            
            if sum <= 10:     
                cluster_choose             = cluster_df_fin['cluster'][range(0,num)]
                sum = sum + j


        #RICS QUE ELIGO
        rics_alloc_fin                     = []

        for k in cluster_choose:

                df_cluster_choose_fin      = top_carac[top_carac['cluster']==k].sort_values("rsi",ascending = True)
                rics_alloc_fin.append(df_cluster_choose.index.tolist())

        rics_alloc_fin                     = [item for sublist in rics_alloc_fin for item in sublist]
        rics_alloc_fin                     = rics_alloc[0:10]

        #PREPARO EL DF CON LOS PESOS
        allocation                         = pd.DataFrame()
        allocation['rics']                 = rics_alloc_fin
        allocation['pesos']                = 1/(len(rics_alloc_fin))
        print(frame[-1], allocation)
        
        
        #GUARDO EN EL DICCIONARIO
        last_day_in_frame                  = frame[-1]
        alocs[last_day_in_frame]           = allocation


#------------------------------------------------------ HAGO EL BACKTESTING----------------------------------

cluster             = alocs
s_p                 = benchmark
capital             = 1000000.00
res_0               = 0.1
res                 = capital*res_0
cap                 = capital-res
Broker_fee          = 0.005 # Per Share Max 1% trade value Max.Val = 
SEC                 = 0.0000051 # Per Share Per price Value of agreagted sales 
TAF                 = 0.000119 # Per Share Quantity Sold. Max.Val = 5.95 $
cap_df              = pd.DataFrame()
ret                 = {}
ret[0]              = cap
date_dif            = []
index_dif           = []
dates               = []
l_s_p               = []
list_ric            = []         
dict_close          = {}
w_min               = {}
w_sharpe            = {}
W                   = 30         
dict_s_p_ret        = {}
dict_close_ret      = {}
dict_stop_loss      = {}
cov_                = [0]
beta_               = [0]
dict_parcial        = {}
vix_dict            = {}  


for  i in cluster.keys():

    dates.append(i)

f_epoch             = len(dates)

for k in range(0,f_epoch):

    cluster[dates[k]]['num_acc']                = 0

    cluster[dates[k]]['CAP_1']                  = 0.00000

    cluster[dates[k]]['CAP_D_T']                = 0.00000          

    cluster[dates[k]]['CAP_0']                  = 0.00000

    cluster[dates[k]]['BROKER_FEE']             = 0.00000

    cluster[dates[k]]['SEC_FEE']                = 0.00000

    cluster[dates[k]]['TAF_FEE']                = 0.00000

    cluster[dates[k]]['CLOSE_buy']              = 0.00000

    cluster[dates[k]]['CLOSE_sell']             = 0.00000

    cluster[dates[k]]['RET']                    = 0.00000

    cluster[dates[k]]['W_RET']                  = 0.00000

cluster[dates[0]]['CAP_D_T'] = cap

def get_data(data, element):

        elements  = pd.DataFrame()
        rics_with_data = []
        for i in list(data.keys()):
                try:
                        df_i = data[i][element]
                        elements = pd.concat([elements,df_i], axis=1)
                        rics_with_data.append(i)
                except TypeError:
                        pass   
        elements.columns = rics_with_data
        

        return elements

close               = get_data(data,'CLOSE')
close               = close.fillna(method='ffill')
close               = close.loc[:dates[-1],:]
s_p                 = s_p.loc[:dates[-1],:]
close_ret           = np.log(close) - np.log(close.shift(1))   
benchmark_ret       = np.log(s_p.CLOSE) - np.log(s_p.CLOSE.shift(1))
d_list              = close.index.tolist()

 

lista_rics = []

for i in range(0,f_epoch-1):

        lista_rics.append( alocs[dates[i]].rics )

        dict_s_p_ret[dates[i]]            =     benchmark_ret[dates[i]:dates[i+1]]

        dict_close_ret[dates[i]]          =     close_ret.loc[dates[i]:dates[i+1],lista_rics[i]]

        dict_close[dates[i]]              =     close.loc[dates[i]:dates[i+1],lista_rics[i]]

        dict_stop_loss[dates[i]]          =     close.loc[dates[i]:dates[i+1],lista_rics[i]]

        vix_dict[dates[i]]                =     vix[dates[i]:dates[i+1]]

        dict_s_p_ret[dates[i]]            =     dict_s_p_ret[dates[i]][-30:]

        dict_close[dates[i]]              =     dict_close[dates[i]].iloc[-30:,:]

        dict_close_ret[dates[i]]          =     dict_close_ret[dates[i]].iloc[-30:,:]

        dict_stop_loss[dates[i]]          =     dict_stop_loss[dates[i]].iloc[-30:,:]

        vix_dict[dates[i]]                =     vix_dict[dates[i]].iloc[-30:,:]


print('entro')

vix_dict.pop(dates[167])
f_epoch = len(vix_dict)

for j in range(0,f_epoch-1):

    for k in range(0,W):

        if  vix_dict[dates[j]]['CLOSE'][k] > 30:

                cluster[dates[j]]=cluster[dates[j-1]]

                continue
        else:

            for i in range(0,10):

                # CLOSE_buy para la compra del cluster en la semana 0

                cluster[dates[j]]['CLOSE_buy'][i]                       = close.loc[dates[j],cluster[dates[j]].rics[i]]

                # CLOSE_sell para la venta del cluster en la semana 1

                cluster[dates[j]]['CLOSE_sell'][i]                      = close.loc[dates[j+1],cluster[dates[j]].rics[i]]

                # NUM_ACC = PESOS * CAPITAL DISPONIBLE // PRECIO POR ACCION

                cluster[dates[j]]['num_acc'][i]                         = (cluster[dates[j]].pesos[i]*cluster[dates[j]]['CAP_D_T'][0])//(cluster[dates[j]].CLOSE_buy[i])
            
                # RICS DIFERENTES ENTRE FECHA J Y J+1

                cap_df['DIF']                                           = cluster[dates[j+1]].rics.where(cluster[dates[j+1]].rics.values != cluster[dates[j]].rics.values).notna()

                # EN FECHA J+1 RICS QUE HAN DE VENDERSE

                cap_df['OUT']                                           = cluster[dates[j]].rics[cap_df['DIF']]

                # EN FECHA J+1 RICS QUE HAN DE COMPRARSE

                cap_df['IN']                                            = cluster[dates[j+1]].rics[cap_df['DIF']]

                cap_df=cap_df.fillna(0)

                

                if cap_df.IN[i]== 0:

                        date_dif.append(dates[j])
                        index_dif.append(j)
                        cluster[dates[j+1]].loc[i,'rics']               = cluster[dates[j]].loc[i,'rics']
                        
                else:
                        cluster[dates[j+1]].loc[i,'rics']               = cap_df.IN[i]


                
                if cap_df.OUT[i] == 0:

                        cap_df.OUT[i]                                   = cluster[dates[j]].loc[i,'rics']
                        cluster[dates[j+1]]['CAP_1'][i]                 = close.loc[dates[j],cluster[dates[j]].rics[i]]*cluster[dates[j]].num_acc[i]
                        cluster[dates[j]]['CAP_1'][i]                   = 0
                    
                else:

                # CAP_1: Capital disponible despues de la venta: Precio de venta * numero de acciones       
                        
                        cluster[dates[j]]['CAP_1'][i]                 = cluster[dates[j]].CLOSE_sell[i] * cluster[dates[j]].num_acc[i]

                # Broker: Numero acciones * fee Broker

                if      cluster[dates[j]].num_acc[i] * Broker_fee      >= cluster[dates[j]]['CAP_1'][i] * 0.001:

                        cluster[dates[j]]['BROKER_FEE'][i]            = cluster[dates[j]]['CAP_1'][i] * 0.001

                else:
                        cluster[dates[j]]['BROKER_FEE'][i]            = cluster[dates[j]].num_acc[i] * Broker_fee

                # SEC: Precio accion * Numero acciones * SEC
                
                cluster[dates[j]]['SEC_FEE'][i]                       = close.loc[dates[j],cluster[dates[j]].rics[i]] * cluster[dates[j]].num_acc[i] * SEC

                # TAF: Numero acciones * TAF 

                if cluster[dates[j]].num_acc[i] * TAF                  >= 5.95:

                        cluster[dates[j]]['TAF_FEE'][i]               = 5.95
                
                else: 

                        cluster[dates[j]]['TAF_FEE'][i]               =  cluster[dates[j]].num_acc[i] * TAF

                if cluster[dates[j]]['CAP_1'][i]                        == 0:

                        cluster[dates[j]]['BROKER_FEE'][i]              = 0
                        cluster[dates[j]]['SEC_FEE'][i]                 = 0
                        cluster[dates[j]]['TAF_FEE'][i]                 = 0
                
                # CAP_1: Capital disponible despues de la venta: Precio de venta * numero de acciones menos las comisiones e impuestos

                cluster[dates[j]]['CAP_1'][i]                         = cluster[dates[j]]['CAP_1'][i] - cluster[dates[j]]['BROKER_FEE'][i] - cluster[dates[j]]['SEC_FEE'][i] - cluster[dates[j]]['TAF_FEE'][i]
                
                cluster[dates[j+1]]['CAP_D_T']                        = cluster[dates[j]]['CAP_1'].sum()

                ret[j+1]                                              = cluster[dates[j]]['CAP_1'].sum()                                              
        
        
t1 = time.time()
print("tiempo2(): %f" % (t1 - t0))   

                        
cap_                = pd.DataFrame.from_dict(ret,orient='index',columns=['CAP'])
cap_['DATES']       = dates[0:len(ret)]
cap_                = cap_.reindex(cap_.columns.tolist() + ['RET_CLUSTER','VOLAT','SHARPE','ALPHA'], axis=1)

for j in range(0,f_epoch-1):

        
        cov_.append(np.cov(dict_close_ret[dates[j+1]].sum(axis=1).astype(float), dict_s_p_ret[dates[j+1]].astype(float), rowvar= False))

        beta_.append(cov_[j+1][0,1]/cov_[j+1][1,1])

        for i in range(0,10):
                cluster[dates[j+1]]['RET'][i]                 = (cluster[dates[j]].CLOSE_sell[i]/cluster[dates[j]].CLOSE_buy[i])-1         
                cluster[dates[j+1]]['W_RET'][i]               = (cluster[dates[j+1]]['RET'][i]*cluster[dates[j]].pesos[i]) 

try:
        for j in range(0,f_epoch-1):
       
                cap_['RET_CLUSTER'][j]                        = cluster[dates[j]]['W_RET'].sum()

                cap_['VOLAT'][j]                              = cluster[dates[j]]['RET'].std()

                cap_['ALPHA'][j]                              = (cap_['RET_CLUSTER'][j] - dict_s_p_ret[dates[j]].mean()) + (beta_[j]*dict_s_p_ret[dates[j]].mean())
 
                cap_['SHARPE'][j]                             = (cap_['RET_CLUSTER'][j]  / cap_['VOLAT'][j]).mean()

except KeyError:
        pass
                

cap_                                                          = cap_.set_index('DATES')

#PLOTEO GRAFICOS DE RENDIMIENTOS

retornos =pd.DataFrame(index=benchmark_ret.index,columns=['date','algo','sp'])
capital             = 1000000
for a in list(retornos.index)[1:]:
    capital = capital + (capital * benchmark_ret[a])
    retornos.loc[a,'sp'] = capital
capital             = 1000000.00
for a in list(cap_.index):
    retornos.loc[a,'algo'] = cap_.loc[a,'CAP']

retornos                 = retornos.fillna( method='ffill')
retornos                 = retornos.fillna( method='bfill')
retornos                 = retornos.astype('float64')
retornos['date']         = benchmark_ret.index

retornos.plot(x="date", y=["algo", "sp"])

T_close         = {}
T_stop_loss     = {}
stop_loss       = {}

for g in range(0,f_epoch): 
        
        T_close[dates[g]]                               = dict_close[dates[g]].T
        T_stop_loss[dates[g]]                           = dict_stop_loss[dates[g]].T
        stop_loss[dates[g]]                             = T_stop_loss[dates[g]].T
        stop_loss[dates[g]]['CAP_PARCIAL']              = 0.00000
        stop_loss[dates[g]]['CAP_0']                    = 0.00000
        stop_loss[dates[g]]['DIFF']                     = 0.00000

for j in range(0,f_epoch-1):

    for k in range(0,W):

        for i in range(0,10):
                try:
                        

                        stop_loss[dates[j]].iloc[k,i]                     = dict_close[dates[j]].iloc[k,i]  *  cluster[dates[j]]['num_acc'][i]

                        stop_loss[dates[j]]['CAP_0'][k]                   = ret[j]

                        stop_loss[dates[j]]['CAP_PARCIAL'][k]             = stop_loss[dates[j]].iloc[k,0:10].sum()

                        stop_loss[dates[j]]['DIFF'][k]                    = stop_loss[dates[j]]['CAP_PARCIAL'][k] - stop_loss[dates[j]]['CAP_0'][k]

                except KeyError:
                        pass




#pickle.dump(stop_loss, open('A2_intra_EW.p','wb'))
#pickle.dump(cluster, open('A2_cluster_EW.p','wb'))
# %%
