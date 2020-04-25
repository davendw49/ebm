# This is the simulation of our evolving RS model under the FIRST framework of our assumptions on edge weights.
import numpy as np
import random
import matplotlib.pyplot as plt
import powerlaw
import pandas as pd
import pymysql

from sim_model_1 import assumption_1st, get_pvalue_alpha_xmin_1, cal_big_c

if __name__ == "__main__":
    # 打开数据库连接
    db = pymysql.connect("10.10.10.10","groupleader","onlyleaders","ebm" )
    
    # 使用cursor()方法获取操作游标 
    cursor = db.cursor()
    
    
    f = open("para_effect_1.txt", "w", encoding="utf8")
    for itime in range(0,100):
        print("this is iter_", itime)
        '''
        sim_model_1-User
        '''
        # beta_list = [0.1,0.2,0.4,0.6,0.8]
        rate_list = [2,3,4,5,6,7,8]
        # cu_list = [1,5,10,20,30,50]
        # ci_list = [1,5,10,20,30,50]

        # beta
#         alpha_list = []
#         for item in beta_list:
#             a1 = assumption_1st(beta=item,iterations=4000,rating_scale=5,Cu=10,Ci=20,Unum=20,Inum=10,K=10,L=5,C=1)
#             degseq, weiseq, xind = a1.get_distribution()
#             p_value, alpha, xmin = get_pvalue_alpha_xmin_1(degseq)
#             alpha_list.append(str(alpha))
#         # data=[alpha_list]
#         # frame_beta_user=pd.DataFrame(data, index=[r'$ \beta $'+' effect '+r'$ \alpha $'], columns=beta_list)
#         # print("#frame_beta_user#", file=f)
#         # print(frame_beta_user, file=f)
#         sql = "INSERT INTO user_beta_method_1(iter,beta_0_1,beta_0_2,beta_0_4,beta_0_6,beta_0_8) VALUES ('%s', '%s', '%s', '%s', '%s', '%s')" % (str(itime),alpha_list[0],alpha_list[1],alpha_list[2],alpha_list[3],alpha_list[4])
#         try:
#             # 执行sql语句
#             cursor.execute(sql)
#             # 提交到数据库执行
#             db.commit()
#         except:
#             # 如果发生错误则回滚
#             db.rollback()
        # rate
        alpha_list = []
        big_c_list = []
        for item in rate_list:
            a1 = assumption_1st(beta=0.6,iterations=4000,rating_scale=item,Cu=10,Ci=20,Unum=20,Inum=10,K=10,L=5,C=1)
            degseq, weiseq, xind = a1.get_distribution()
            p_value, alpha, xmin = get_pvalue_alpha_xmin_1(degseq)
            alpha_list.append(str(alpha))
            big_c = cal_big_c(a1)
            big_c_list.append(str(big_c))
        # data=[alpha_list]
        # frame_rate_user=pd.DataFrame(data, index=['rate'+' effect '+r'$ \alpha $'], columns=rate_list)
        # print("#frame_rate_user#", file=f)
        # print(frame_rate_user, file=f)
        sql = "INSERT INTO user_rate_method_1_final_2(iter,rate_2,rate_3,rate_4,rate_5,rate_6,rate_7,rate_8, c_2,c_3,c_4,c_5,c_6,c_7,c_8) VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (str(itime),alpha_list[0],alpha_list[1],alpha_list[2],alpha_list[3],alpha_list[4],alpha_list[5],alpha_list[6],big_c_list[0],big_c_list[1],big_c_list[2],big_c_list[3],big_c_list[4],big_c_list[5],big_c_list[6])
        try:
            # 执行sql语句
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
        except:
            # 如果发生错误则回滚
            db.rollback()
        # Cu
        # alpha_list = []
        # for item in cu_list:
        #     print(item)
        #     a1 = assumption_1st(beta=0.6,iterations=4000,rating_scale=5,Cu=item,Ci=20,Unum=20,Inum=50,K=10,L=5,C=1)
        #     degseq, weiseq, xind = a1.get_distribution()
        #     p_value, alpha, xmin = get_pvalue_alpha_xmin_1(degseq)
        #     alpha_list.append(alpha)
        # data=[alpha_list]
        # frame_cu_user=pd.DataFrame(data, index=[r'$ c_{u} $'+' effect '+r'$ \alpha $'], columns=cu_list)
        # print("#frame_cu_user#", file=f)
        # print(frame_cu_user, file=f)
        
        # # Ci
        # alpha_list = []
        # for item in ci_list:
        #     print(item)
        #     a1 = assumption_1st(beta=0.6,iterations=4000,rating_scale=5,Cu=10,Ci=item,Unum=50,Inum=10,K=10,L=5,C=1)
        #     degseq, weiseq, xind = a1.get_distribution()
        #     p_value, alpha, xmin = get_pvalue_alpha_xmin_1(degseq)
        #     alpha_list.append(alpha)
        # data=[alpha_list]
        # frame_ci_user=pd.DataFrame(data, index=[r'$ c_{i} $'+' effect '+r'$ \alpha $'], columns=ci_list)
        # print("#frame_ci_user#", file=f)
        # print(frame_ci_user, file=f)


        '''
        sim_model_1-Item
        '''

        # beta_list = [0.1,0.2,0.4,0.6,0.8]
        rate_list = [2,3,4,5,6,7,8]
        # cu_list = [1,5,10,20,30,50]
        # ci_list = [1,5,10,20,30,50]

        # beta
#         alpha_list = []
#         for item in beta_list:
#             a1 = assumption_1st(beta=item,iterations=4000,rating_scale=5,Cu=10,Ci=20,Unum=20,Inum=10,K=10,L=5,C=1)
#             degseq, weiseq, xind = a1.get_distribution("item")
#             p_value, alpha, xmin = get_pvalue_alpha_xmin_1(degseq)
#             alpha_list.append(str(alpha))
#         # data=[alpha_list]
#         # frame_beta_item=pd.DataFrame(data, index=[r'$ \beta $'+' effect '+r'$ \alpha $'], columns=beta_list)
#         # print("#frame_beta_item#", file=f)
#         # print(frame_beta_item, file=f)
#         sql = "INSERT INTO item_beta_method_1(iter,beta_0_1,beta_0_2,beta_0_4,beta_0_6,beta_0_8) VALUES ('%s', '%s', '%s', '%s', '%s', '%s')" % (str(itime),alpha_list[0],alpha_list[1],alpha_list[2],alpha_list[3],alpha_list[4])
#         try:
#             # 执行sql语句
#             cursor.execute(sql)
#             # 提交到数据库执行
#             db.commit()
#         except:
#             # 如果发生错误则回滚
#             db.rollback()


        # rate
        alpha_list = []
        big_c_list = []
        for item in rate_list:
            a1 = assumption_1st(beta=0.6,iterations=4000,rating_scale=item,Cu=10,Ci=20,Unum=20,Inum=10,K=10,L=5,C=1)
            degseq, weiseq, xind = a1.get_distribution("item")
            p_value, alpha, xmin = get_pvalue_alpha_xmin_1(degseq)
            alpha_list.append(str(alpha))
            big_c = cal_big_c(a1)
            big_c_list.append(str(big_c))
        # data=[alpha_list]
        # frame_rate_item=pd.DataFrame(data, index=['rate'+' effect '+r'$ \alpha $'], columns=rate_list)
        # print("#frame_rate_item#", file=f)
        # print(frame_rate_item, file=f)
        sql = "INSERT INTO item_rate_method_1_final_2(iter,rate_2,rate_3,rate_4,rate_5,rate_6,rate_7,rate_8, c_2,c_3,c_4,c_5,c_6,c_7,c_8) VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (str(itime),alpha_list[0],alpha_list[1],alpha_list[2],alpha_list[3],alpha_list[4],alpha_list[5],alpha_list[6],big_c_list[0],big_c_list[1],big_c_list[2],big_c_list[3],big_c_list[4],big_c_list[5],big_c_list[6])
        try:
            # 执行sql语句
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
        except:
            # 如果发生错误则回滚
            db.rollback()
        # Cu
        # alpha_list = []
        # for item in cu_list:
        #     print(item)
        #     a1 = assumption_1st(beta=0.6,iterations=4000,rating_scale=5,Cu=item,Ci=20,Unum=20,Inum=50,K=10,L=5,C=1)
        #     degseq, weiseq, xind = a1.get_distribution("item")
        #     p_value, alpha, xmin = get_pvalue_alpha_xmin_1(degseq)
        #     alpha_list.append(alpha)
        # data=[alpha_list]
        # frame_cu_item=pd.DataFrame(data, index=[r'$ c_{u} $'+' effect '+r'$ \alpha $'], columns=cu_list)
        # print("#frame_cu_item#", file=f)
        # print(frame_cu_item, file=f)

        # # Ci
        # alpha_list = []
        # for item in ci_list:
        #     print(item)
        #     a1 = assumption_1st(beta=0.6,iterations=4000,rating_scale=5,Cu=10,Ci=item,Unum=50,Inum=10,K=10,L=5,C=1)
        #     degseq, weiseq, xind = a1.get_distribution("item")
        #     p_value, alpha, xmin = get_pvalue_alpha_xmin_1(degseq)
        #     alpha_list.append(alpha)
        # data=[alpha_list]
        # frame_ci_item=pd.DataFrame(data, index=[r'$ c_{i} $'+' effect '+r'$ \alpha $'], columns=ci_list)
        # print("#frame_ci_item#", file=f)
        # print(frame_ci_item, file=f)
    
    f.close()
    # 关闭数据库连接
    db.close()