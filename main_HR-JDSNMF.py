import os
import pandas as pd
import tensorflow as tf
import numpy as np
import math
from math import sqrt
from matplotlib import pyplot as plt

SEED = 1
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
tf.compat.v1.disable_eager_execution()


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.compat.v1.random.set_random_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.compat.v1.config.threading.set_inter_op_parallelism_threads(1)
    tf.compat.v1.config.threading.set_intra_op_parallelism_threads(1)


set_global_determinism(seed=SEED)


def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den

class EarlyStopping:
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('Training process is stopped early')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


def frob(z):
    vec = tf.reshape(z, [-1])
    return tf.reduce_sum(tf.multiply(vec, vec))


def minmax_sca(data):
    """
    离差标准化
    param data:传入的数据
    return:标准化之后的数据
    """
    new_data = (data - data.min()) / (data.max() - data.min())
    return new_data


def main():
    np.random.seed(1)
    gene_df = pd.read_csv('pet_df_final_noname.csv', header=None)
    meth_df = pd.read_csv('snp_df_final_noname.csv', header=None)
    pet_df = pd.read_csv('gene_df_final_noname.csv', header=None)

    gene = gene_df.values
    meth = meth_df.values
    pet = pet_df.values

    L1 = pd.read_csv('pet_df_final_hp.csv', header=None)
    L1 = L1.values
    L2 = pd.read_csv('snp_df_final_hp.csv', header=None)
    L2 = L2.values
    L3 = pd.read_csv('gene_df_final_hp.csv', header=None)
    L3 = L3.values
    # Hyperparameters
    max_steps = 15000
    early_stopping = EarlyStopping(patience=200, verbose=1)
    first_reduced_dimension_1 = 25  # K值
    second_reduced_dimension = 20
    third_reduced_dimension = 15
    fourth_reduced_dimension = 10

    lambda_ = 0.001
    la1 = 0.001
    la2 = 0.001
    la3 = 0.001

    n, dm = meth.shape
    _, dg = gene.shape
    _, dp = pet.shape

    tf.compat.v1.set_random_seed(1)
    sess = tf.compat.v1.InteractiveSession()

    DM = tf.compat.v1.placeholder(tf.float32, shape=(None, dm))
    GE = tf.compat.v1.placeholder(tf.float32, shape=(None, dg))
    PE = tf.compat.v1.placeholder(tf.float32, shape=(None, dp))

    L1 = tf.Variable(L1, tf.float32)
    L2 = tf.Variable(L2, tf.float32)
    L3 = tf.Variable(L3, tf.float32)

    # Initialization using SVD
    DM_svd_u_1, _, DM_svd_vh_1 = np.linalg.svd(meth, full_matrices=False)
    DM_svd_u_2, _, DM_svd_vh_2 = np.linalg.svd(DM_svd_u_1, full_matrices=False)
    DM_svd_u_3, _, DM_svd_vh_3 = np.linalg.svd(DM_svd_u_2, full_matrices=False)
    DM_svd_u_4, _, DM_svd_vh_4 = np.linalg.svd(DM_svd_u_3, full_matrices=False)

    GE_svd_u_1, _, GE_svd_vh_1 = np.linalg.svd(gene, full_matrices=False)
    GE_svd_u_2, _, GE_svd_vh_2 = np.linalg.svd(GE_svd_u_1, full_matrices=False)
    GE_svd_u_3, _, GE_svd_vh_3 = np.linalg.svd(GE_svd_u_2, full_matrices=False)
    GE_svd_u_4, _, GE_svd_vh_4 = np.linalg.svd(GE_svd_u_3, full_matrices=False)

    PE_svd_u_1, _, PE_svd_vh_1 = np.linalg.svd(pet, full_matrices=False)
    PE_svd_u_2, _, PE_svd_vh_2 = np.linalg.svd(PE_svd_u_1, full_matrices=False)
    PE_svd_u_3, _, PE_svd_vh_3 = np.linalg.svd(PE_svd_u_2, full_matrices=False)
    PE_svd_u_4, _, PE_svd_vh_4 = np.linalg.svd(PE_svd_u_3, full_matrices=False)

    U = tf.Variable(tf.cast(DM_svd_u_4[:, 0:first_reduced_dimension_1], tf.float32))

    Z21 = tf.Variable(
        tf.cast(DM_svd_u_2[0:first_reduced_dimension_1, 0:second_reduced_dimension], tf.float32))
    Z11 = tf.Variable(
        tf.cast(GE_svd_u_2[0:first_reduced_dimension_1, 0:second_reduced_dimension], tf.float32))
    Z31 = tf.Variable(
        tf.cast(PE_svd_u_2[0:first_reduced_dimension_1, 0:second_reduced_dimension], tf.float32))

    Z22 = tf.Variable(
        tf.cast(DM_svd_u_3[0:second_reduced_dimension, 0:third_reduced_dimension], tf.float32))
    Z12 = tf.Variable(
        tf.cast(GE_svd_u_3[0:second_reduced_dimension, 0:third_reduced_dimension], tf.float32))
    Z32 = tf.Variable(
        tf.cast(PE_svd_u_3[0:second_reduced_dimension, 0:third_reduced_dimension], tf.float32))

    Z23 = tf.Variable(
        tf.cast(DM_svd_u_4[0:third_reduced_dimension, 0:fourth_reduced_dimension], tf.float32))
    Z13 = tf.Variable(
        tf.cast(GE_svd_u_4[0:third_reduced_dimension, 0:fourth_reduced_dimension], tf.float32))
    Z33 = tf.Variable(
        tf.cast(PE_svd_u_4[0:third_reduced_dimension, 0:fourth_reduced_dimension], tf.float32))

    H23 = tf.Variable(tf.cast(DM_svd_vh_1[0:fourth_reduced_dimension, :], tf.float32))
    H13 = tf.Variable(tf.cast(GE_svd_vh_1[0:fourth_reduced_dimension, :], tf.float32))
    H33 = tf.Variable(tf.cast(PE_svd_vh_1[0:fourth_reduced_dimension, :], tf.float32))


    H10 = tf.sigmoid(tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13))))))
    H20 = tf.sigmoid(tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))))
    H30 = tf.sigmoid(tf.matmul(Z31, tf.sigmoid(tf.matmul(Z32, tf.sigmoid(tf.matmul(Z33, H33))))))

    H11 = tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13))))
    H21 = tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))
    H31 = tf.sigmoid(tf.matmul(Z32, tf.sigmoid(tf.matmul(Z33, H33))))

    H12 = tf.sigmoid(tf.matmul(Z13, H13))
    H22 = tf.sigmoid(tf.matmul(Z23, H23))
    H32 = tf.sigmoid(tf.matmul(Z33, H33))

    # loss function
    loss = frob(
        GE - tf.matmul(U, tf.sigmoid(
            tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13)))))))) + \
           frob(
               DM - tf.matmul(U, tf.sigmoid(
                   tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23)))))))) + \
           frob(
               PE - tf.matmul(U, tf.sigmoid(
                   tf.matmul(Z31, tf.sigmoid(tf.matmul(Z32, tf.sigmoid(tf.matmul(Z33, H33)))))))) + \
           lambda_ * (frob(U) + frob(Z11) + frob(Z12) + frob(Z32) + frob(Z13) + frob(H13) + frob(Z21) + frob(Z31) + \
                      frob(
        Z22) + frob(H11) + frob(H21) + frob(H31) + frob(H12) + frob(H22) + frob(H32) + frob(
        Z23) + frob(
        H23) + frob(
        Z33) + frob(
        H33) + la1*tf.linalg.trace(tf.matmul(tf.matmul(H10, tf.cast(L1, tf.float32)), tf.transpose(H10))) + la2*tf.linalg.trace(tf.matmul(tf.matmul(H20, tf.cast(L2, tf.float32)), tf.transpose(H20))) + la3*tf.linalg.trace(tf.matmul(tf.matmul(H30, tf.cast(L3, tf.float32)), tf.transpose(H30)))
                      )

    diff_DM = frob(
        DM - tf.matmul(U, tf.sigmoid(
            tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))))))
    diff_GE = frob(
        GE - tf.matmul(U, tf.sigmoid(
            tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13))))))))
    diff_PE = frob(
        PE - tf.matmul(U, tf.sigmoid(
            tf.matmul(Z31, tf.sigmoid(tf.matmul(Z32, tf.sigmoid(tf.matmul(Z33, H33))))))))

    MF = frob(
        GE - tf.matmul(U, tf.sigmoid(
            tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13)))))))) + \
         frob(
             DM - tf.matmul(U, tf.sigmoid(
                 tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))))))+ \
         frob(
             PE - tf.matmul(U, tf.sigmoid(
                 tf.matmul(Z31, tf.sigmoid(tf.matmul(Z32, tf.sigmoid(tf.matmul(Z33, H33))))))))
    F = lambda_ * (
            frob(U) + frob(Z11) + frob(Z12) + frob(Z13) + frob(H13) + frob(Z21) + frob(Z22) + frob(
        Z23) + frob(H23)) + frob(Z31) + frob(Z32) + frob(Z33)

    L21 = la1*tf.linalg.trace(tf.matmul(tf.matmul(H10, tf.cast(L1, tf.float32)),tf.transpose(H10))) + la2*tf.linalg.trace(tf.matmul(tf.matmul(H20,tf.cast(L2, tf.float32)),tf.transpose(H20))) + la3*tf.linalg.trace(tf.matmul(tf.matmul(H30,tf.cast(L3, tf.float32)),tf.transpose(H30)))

    # O1 = lo1*ATV(H10)+lo2*ATV(H20)+lo3*ATV(H30)
    train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(loss)

    tf.compat.v1.global_variables_initializer().run()

    funval = []
    _, loss_iter = sess.run([train_step, loss], feed_dict={DM: meth, GE: gene, PE: pet})
    funval.append(loss_iter)

    indicate_dict = {}
    loss_list = []
    diff_me_list = []
    diff_ge_list = []
    diff_pe_list = []
    L21_list = []
    # O1_list = []
    MF_list = []
    F21_list = []

    for i in range(max_steps):
        _, loss_iter = sess.run([train_step, loss], feed_dict={DM: meth, GE: gene, PE: pet})
        funval.append(loss_iter)
        loss_1 = sess.run(loss, feed_dict={DM: meth, GE: gene, PE: pet})
        diff_me = sess.run(diff_DM, feed_dict={DM: meth, GE: gene, PE: pet})
        diff_ge = sess.run(diff_GE, feed_dict={DM: meth, GE: gene, PE: pet})
        diff_pe = sess.run(diff_PE, feed_dict={DM: meth, GE: gene, PE: pet})
        MF_1 = sess.run(MF, feed_dict={DM: meth, GE: gene, PE: pet})
        F21 = sess.run(F, feed_dict={DM: meth, GE: gene, PE: pet})
        L21 = sess.run(tf.cast(L21, tf.float32), feed_dict={DM: meth, GE: gene, PE: pet})
        # O1 = sess.run(O1, feed_dict={DM: meth, GE: gene, PE: pet})
        loss_list.append(loss_1)
        diff_me_list.append(diff_me)
        diff_ge_list.append(diff_ge)
        diff_pe_list.append(diff_pe)
        MF_list.append(MF_1)
        F21_list.append(F21)
        L21_list.append(L21)
        # O1_list.append(O1)
        if (i % 1000 == 0) & i != 0:
            print(i, " Loss : %f" % loss_1)
            print(" Average diff_me : %f" % diff_me)
            print(" Average diff_ge : %f" % diff_ge)
            print(" MF: %f" % MF_1)
        if early_stopping.validate(loss_iter):
            break
        if math.isnan(loss_iter):
            break


if __name__ == '__main__':
    main()

