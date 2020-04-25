# Calculate the item degree and vertex weight distributions for Amazon Book dataset
import numpy as np
import matplotlib.pyplot as plt
from powerlaw.regression import estimate_parameters, goodness_of_fit
import time
import datetime
from sklearn import linear_model
from scipy.stats import norm

def data_form(fin, fout):
    f = open(fin, 'r', encoding='utf8')
    f_out = open(fout, 'w', encoding='utf8')
    pos = 0
    for line in f:
        pos+=1
        if pos%1000000 == 0:
            print(pos)
        c = line.split(',')
        tmp_timestamp = change_time(c[3])
        tmp_timestamp = tmp_timestamp.replace('-','')
        print(c[0], c[1], c[2], tmp_timestamp, file = f_out)
    f.close()
    f_out.close()

def change_time(u):
    u = int(u)
    # u = 1393200000        #unix时间戳
    t = datetime.datetime.fromtimestamp(u)
    t_str = str(t)
    return t_str.split(' ')[0]

def data_processing():
    for csv in ['ratings_Books', 'ratings_CDs_and_Vinyl', 'ratings_Electronics', 'ratings_Movies_and_TV']:
        filename_in = '../RealDatasetValidation/data/' + csv + '.csv'
        new_csv = csv.replace('ratings_', '')
        filename_out = '../RealDatasetValidation/data/amazon/' + new_csv +'.txt'
        print("********************" + new_csv + "*******************")
        print("input", filename_in)
        print("output", filename_out)
        data_form(filename_in, filename_out)
        print("********************END" + new_csv + "*******************")

def sort_by_time(fin, fout):
    f = open(fin, 'r', encoding='utf8')
    f_out = open(fout, 'w', encoding='utf8')
    pos = 0
    dataset = []
    for line in f:
        pos+=1
        if pos%1000000 == 0:
            print(pos)
        c = line.split(' ')
        dataset.append((c[0], c[1], c[2], int(c[3])))
    print("sorting......")
    dataset = sorted(dataset,key=lambda x:x[3])
    for i in range(0, len(dataset)):
        print(dataset[i][0], dataset[i][1], dataset[i][2], str(dataset[i][3]), file=f_out)
    f.close()
    f_out.close()

# 22507155 book
# 7824482 electronic
# 4607047 movie
# 3749004 CD

def graph_main(fin, num, pace):
    stop_time = int(pace * num)
    print('stoptime:', stop_time, "pace:", pace)
    degree = [0]*10000000
    item = [0]*10000000
    weight = [0]*10000000
    # files = 'ratings_Books.csv'
    f = open(fin, 'r', encoding='utf8')
    data = []
    cnt =0
    ytime = 19700101
    print ('Start reading from '+fin)
    for line in f:
        cnt = cnt + 1
        if cnt ==1:
            continue
        if cnt == stop_time:
            ytime = int(line.split(' ')[3])
            print('ytime', ytime)
            break
        if cnt % 1000000 == 0:
            print (cnt)
        line = line.split(' ')
        user = hash(line[0])%10000000
        book = hash(line[1])%10000000
        rate = float(line[2])
        degree[user] = degree[user] + 1
        weight[user] = weight[user] + rate
    deg = np.array(degree)
    bins = np.bincount(deg)
    np.set_printoptions(threshold=np.inf)
    degseq = bins[:20000]
    xind = np.arange(degseq.shape[0])
    phi_deg = DataFit(xind[1:200], degseq[1:200])
    # plt.plot(xind[1:100],degseq[1:100],'bo')
    # plt.title('User Degree Distributions')
    # plt.xlabel('Degree')
    # plt.ylabel('Number')
    # plt.show()
    # plt.plot(np.log(xind[1:100]),np.log(degseq[1:100]),'bo')
    # plt.title('User Degree Distributions (Logarithm)')
    # plt.xlabel('Degree')
    # plt.ylabel('Number')
    # plt.show()
    wei = np.array(weight)
    binw = np.bincount(wei.astype('int'))
    weiseq = binw[:20000]
    phi_wei = DataFit(xind[1:200], weiseq[1:200])
    # plt.plot(xind[1:100],weiseq[1:100],'ro')
    # plt.title('User Vertex Weight Distributions')
    # plt.xlabel('Vertex weight')
    # plt.ylabel('Number')
    # plt.show()
    # plt.plot(np.log(xind[1:100]),np.log(weiseq[1:100]),'ro')
    # plt.title('User Vertex Weight Distributions (Logarithm)')
    # plt.xlabel('Vertex weight')
    # plt.ylabel('Number')
    # plt.show()
    return phi_deg, phi_wei, ytime

def DataFit(X, Y):
    # 模型数据准备
    X_parameter=[]
    Y_parameter=[]
    for single_square_feet ,single_price_value in zip(X,Y):
        X_parameter.append([float(single_square_feet)])
        Y_parameter.append(float(single_price_value))

    # 模型拟合
    regr = linear_model.LinearRegression()
    regr.fit(X_parameter, Y_parameter)

    #加p_value
    try:
        data = Y
        (xmin, alpha, ks_statistics) = estimate_parameters(data)
        p_value = goodness_of_fit(data, xmin, alpha, ks_statistics)
        print("p_value:", p_value, "item alpha:", alpha, "item xmin:", xmin)
    except:
        pass

    # 模型结果与得分
    # print("#", year, ":")
    # print("幂律指数𝜑: \n", 0-regr.coef_[0])
    # print("Intercept:\n",regr.intercept_)
    # The mean square error
    # print("rest: %.8f" % np.mean((regr.predict(X_parameter) - Y_parameter) ** 2))  # 残差平方和

    phi = 0-regr.coef_[0]
    return phi#, p_value, alpha

def data_time():
    for csv in ['ratings_CDs_and_Vinyl', 'ratings_Books', 'ratings_Electronics', 'ratings_Movies_and_TV']:
        csv = csv.replace('ratings_', '')
        filename_in = '../RealDatasetValidation/data/amazon/' + csv +'.txt'
        filename_out = '../RealDatasetValidation/data/amazon/' + csv +'_dump.txt'
        print("********************" + csv + "*******************")
        print("input", filename_in)
        print("output", filename_out)
        sort_by_time(filename_in, filename_out)
        print("********************END" + csv + "*******************")

# 22507155 book
# 7824482 electronic
# 4607047 movie
# 3749004 CD

if __name__=="__main__":
    csv = ['ratings_CDs_and_Vinyl', 'ratings_Books', 'ratings_Electronics', 'ratings_Movies_and_TV']
    filename_in = '../RealDatasetValidation/data/amazon/' + csv[2].replace('ratings_','') +'_dump.txt'
    for i in range(10):
        if i!=0:
            print(graph_main(filename_in, i, int(7824482/10)))