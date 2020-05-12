# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv

def append_bias_reshape(features,labels):
    m = features.shape[0]
    n = features.shape[1]
    x = np.reshape(np.c_[np.ones(m),features],[m,n+1])
    y = np.reshape(labels,[m,1])

    return x,y

def get_ef(pt_list):
    patient_list = pt_list
    patient_list.sort()
    ef_list = {}
    with open('../stat/ef.labels','r') as f:
        a = csv.reader(f)
        for row in a:
            if row[3] in patient_list:
                if row[3] not in ef_list:
                    ef_list[row[3]] = [row]
                else:
                    ef_list[row[3]].append(row)
    pt_ef = {}
    for key,value in ef_list.items():
        pt_ef[key] = value[-1][1]
    return pt_ef

def get_patient_value(patient_dict):
    avg_pt_dict = {}
    for key,value in patient_dict.items():
        pt_arr = np.zeros(768)
        count = 0
        for item in value:
            count += 1
            pt_arr += item
        pt_arr = pt_arr/count
        avg_pt_dict[key] = pt_arr

    return avg_pt_dict

def categorize(y):
    if y<35:
        return 'I'
    elif y<39:
        return 'II'
    elif y<54:
        return  'III'
    else:
        return  'IV'

if __name__ == '__main__':
    with open('../stat/pt_all.pkl','rb') as f:
        ori_data = pickle.load(f)

    embedding = get_patient_value(ori_data)
    pt_list = []
    for key in embedding:
        pt_list.append(key)
    pt_ef = get_ef(pt_list)

    train_X = []
    train_Y = []
    for key in pt_ef:
        train_X.append(embedding[key])
        train_Y.append(pt_ef[key])

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    train_X,train_Y = append_bias_reshape(train_X,train_Y)
    #
    # m = len(train_X)
    # n = 768+1
    #
    # X = tf.placeholder(tf.float32,name='X',shape=[m,n])
    # Y = tf.placeholder(tf.float32,name='Y')
    #
    # weight = tf.Variable(tf.random_normal([n,1]))
    # bias = tf.Variable(tf.random_normal([1]))
    #
    # Y_hat = tf.matmul(X,weight)
    #
    # loss = tf.reduce_mean(tf.square(Y-Y_hat,name='LOSS'))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    #
    # init_op = tf.global_variables_initializer()
    # total = []
    #
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     writer = tf.summary.FileWriter('graphs',sess.graph)
    #     for i in range(5000):
    #         _, l = sess.run([optimizer,loss],feed_dict={X:train_X,Y:train_Y})
    #         total.append(l)
    #         print('Epoch: {},Loss: {}'.format(i,l))
    #         writer.close()
    #         w_val,b_val = sess.run([weight,bias])
    #
    # with open('../result/weight.pkl','wb') as f:
    #     pickle.dump([w_val,b_val],f)
    # plt.plot(total)
    # plt.show()

    with open('../result/weight.pkl','rb') as f:
        weight = pickle.load(f)

    y_pre = []
    y_true = []
    for i in range(len(train_X)):
        X_pred = train_X[i,:]
        Y_pred = np.matmul(X_pred,weight[0])+weight[1]
        y_pre.append(Y_pred[0])
        y_true.append(float(train_Y[i][0]))

    # print(y_pre)
    # print(y_true)
    count = 0
    with open('../stat/regress.csv','w') as f:
        f.write("true EF,true category,pred EF,pred category\n")
        for i in range(len(y_pre)):
            prec = categorize(y_pre[i])
            truec = categorize(y_true[i])
            if prec==truec:
                count += 1
            f.write("{},{},{},{}\n".format(y_true[i],truec,y_pre[i],prec))

    print

    plt.xlabel('prediction')
    plt.ylabel('true')
    plt.xlim((0,100))
    plt.ylim((0,100))
    plt.scatter(y_pre,y_true,s = 0.3)
    plt.savefig('../stat/regress.png')
    plt.show()