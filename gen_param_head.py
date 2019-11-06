import pickle

para = pickle.load(open('param.dat', 'rb'))

f = open("gen_encoder.h","w")
f.write('double ew[64][2] = {')
for i in range(64):
    for j in range(2):
        f.write("%.8e,\n" % para['encoder.weight'][i,j])
f.write('};')

f = open("gen_encoder.h","a")
f.write('double eb[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['encoder.bias'][i])
f.write('};')
f.close()

f = open("gen_b0l0.h","w")
f.write('double b_0_l_0_na[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.layers.0.connection.norm.a_2'][i])
f.write('};')

f = open("gen_b0l0.h","a")
f.write('double b_0_l_0_nb[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.layers.0.connection.norm.b_2'][i])
f.write('};')

f = open("gen_b0l0.h","a")
f.write('double b_0_l_0_wih[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.0.layers.0.local_rnn.rnn.weight_ih_l0'][k,i])
f.write('};')

f = open("gen_b0l0.h","a")
f.write('double b_0_l_0_whh[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.0.layers.0.local_rnn.rnn.weight_hh_l0'][k,i])
f.write('};')

f = open("gen_b0l0.h","a")
f.write('double b_0_l_0_bih[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.0.layers.0.local_rnn.rnn.bias_ih_l0'][k])
f.write('};')

f = open("gen_b0l0.h","a")
f.write('double b_0_l_0_bhh[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.0.layers.0.local_rnn.rnn.bias_hh_l0'][k])
f.write('};')
f.close()

######################################## block0 layer1
f = open("gen_b0l1.h","w")
f.write('double b_0_l_1_na[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.layers.1.connection.norm.a_2'][i])
f.write('};')

f = open("gen_b0l1.h","a")
f.write('double b_0_l_1_nb[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.layers.1.connection.norm.b_2'][i])
f.write('};')

f = open("gen_b0l1.h","a")
f.write('double b_0_l_1_wih[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.0.layers.1.local_rnn.rnn.weight_ih_l0'][k,i])
f.write('};')

f = open("gen_b0l1.h","a")
f.write('double b_0_l_1_whh[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.0.layers.1.local_rnn.rnn.weight_hh_l0'][k,i])
f.write('};')

f = open("gen_b0l1.h","a")
f.write('double b_0_l_1_bih[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.0.layers.1.local_rnn.rnn.bias_ih_l0'][k])
f.write('};')

f = open("gen_b0l1.h","a")
f.write('double b_0_l_1_bhh[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.0.layers.1.local_rnn.rnn.bias_hh_l0'][k])
f.write('};')
f.close()
#################################  gen_b0p.h  MHP

f = open("gen_b0p.h","w")
f.write('double b_0_c_0_na[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.connections.0.norm.a_2'][i])
f.write('}; \n')

f = open("gen_b0p.h","a")
f.write('double b_0_c_0_nb[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.connections.0.norm.b_2'][i])
f.write('}; \n')

f = open("gen_b0p.h","a")
f.write('double b_0_p_w0[64][64] = {')
for k in range(64):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.0.pooling.linears.0.weight'][k,i])
f.write('}; \n')

f = open("gen_b0p.h","a")
f.write('double b_0_p_b0[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.pooling.linears.0.bias'][k])
f.write('}; \n')

f = open("gen_b0p.h","a")
f.write('double b_0_p_w1[64][64] = {')
for k in range(64):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.0.pooling.linears.1.weight'][k,i])
f.write('}; \n')

f = open("gen_b0p.h","a")
f.write('double b_0_p_b1[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.pooling.linears.1.bias'][k])
f.write('}; \n')

f = open("gen_b0p.h","a")
f.write('double b_0_p_w2[64][64] = {')
for k in range(64):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.0.pooling.linears.2.weight'][k,i])
f.write('}; \n')

f = open("gen_b0p.h","a")
f.write('double b_0_p_b2[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.pooling.linears.2.bias'][k])
f.write('}; \n')

f = open("gen_b0p.h","a")
f.write('double b_0_p_w3[64][64] = {')
for k in range(64):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.0.pooling.linears.3.weight'][k,i])
f.write('}; \n')

f = open("gen_b0p.h","a")
f.write('double b_0_p_b3[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.pooling.linears.3.bias'][k])
f.write('}; \n')
f.close()

#====================gen_b0f.h

f = open("gen_b0f.h","w")
f.write('double b_0_c_1_na[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.connections.1.norm.a_2'][i])
f.write('};')

f = open("gen_b0f.h","a")
f.write('double b_0_c_1_nb[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.connections.1.norm.b_2'][i])
f.write('};')

f = open("gen_b0f.h","a")
f.write('double b_0_f_w1[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.0.feed_forward.w_1.weight'][k,i])
f.write('};')

f = open("gen_b0f.h","a")
f.write('double b_0_f_b1[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.0.feed_forward.w_1.bias'][k])
f.write('};')

f = open("gen_b0f.h","a")
f.write('double b_0_f_w2[64][256] = {')
for k in range(64):
    for i in range(256):
        f.write("%.8e,\n" % para['rt.forward_net.0.feed_forward.w_2.weight'][k,i])
f.write('};')

f = open("gen_b0f.h","a")
f.write('double b_0_f_b2[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.0.feed_forward.w_2.bias'][k])
f.write('};')
f.close()

#============================================BLOCK2
#=====================================================
f = open("gen_b1l0.h","w")
f.write('double b_1_l_0_na[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.layers.0.connection.norm.a_2'][i])
f.write('};')

f = open("gen_b1l0.h","a")
f.write('double b_1_l_0_nb[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.layers.0.connection.norm.b_2'][i])
f.write('};')

f = open("gen_b1l0.h","a")
f.write('double b_1_l_0_wih[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.1.layers.0.local_rnn.rnn.weight_ih_l0'][k,i])
f.write('};')

f = open("gen_b1l0.h","a")
f.write('double b_1_l_0_whh[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.1.layers.0.local_rnn.rnn.weight_hh_l0'][k,i])
f.write('};')

f = open("gen_b1l0.h","a")
f.write('double b_1_l_0_bih[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.1.layers.0.local_rnn.rnn.bias_ih_l0'][k])
f.write('};')

f = open("gen_b1l0.h","a")
f.write('double b_1_l_0_bhh[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.1.layers.0.local_rnn.rnn.bias_hh_l0'][k])
f.write('};')
f.close()

######################################## block1 layer1
f = open("gen_b1l1.h","w")
f.write('double b_1_l_1_na[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.layers.1.connection.norm.a_2'][i])
f.write('};')

f = open("gen_b1l1.h","a")
f.write('double b_1_l_1_nb[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.layers.1.connection.norm.b_2'][i])
f.write('};')

f = open("gen_b1l1.h","a")
f.write('double b_1_l_1_wih[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.1.layers.1.local_rnn.rnn.weight_ih_l0'][k,i])
f.write('};')

f = open("gen_b1l1.h","a")
f.write('double b_1_l_1_whh[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.1.layers.1.local_rnn.rnn.weight_hh_l0'][k,i])
f.write('};')

f = open("gen_b1l1.h","a")
f.write('double b_1_l_1_bih[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.1.layers.1.local_rnn.rnn.bias_ih_l0'][k])
f.write('};')

f = open("gen_b1l1.h","a")
f.write('double b_1_l_1_bhh[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.1.layers.1.local_rnn.rnn.bias_hh_l0'][k])
f.write('};')
f.close()
#################################  gen_b1p.h  MHP

f = open("gen_b1p.h","w")
f.write('double b_1_c_0_na[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.connections.0.norm.a_2'][i])
f.write('}; \n')

f = open("gen_b1p.h","a")
f.write('double b_1_c_0_nb[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.connections.0.norm.b_2'][i])
f.write('}; \n')

f = open("gen_b1p.h","a")
f.write('double b_1_p_w0[64][64] = {')
for k in range(64):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.1.pooling.linears.0.weight'][k,i])
f.write('}; \n')

f = open("gen_b1p.h","a")
f.write('double b_1_p_b0[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.pooling.linears.0.bias'][k])
f.write('}; \n')

f = open("gen_b1p.h","a")
f.write('double b_1_p_w1[64][64] = {')
for k in range(64):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.1.pooling.linears.1.weight'][k,i])
f.write('}; \n')

f = open("gen_b1p.h","a")
f.write('double b_1_p_b1[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.pooling.linears.1.bias'][k])
f.write('}; \n')

f = open("gen_b1p.h","a")
f.write('double b_1_p_w2[64][64] = {')
for k in range(64):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.1.pooling.linears.2.weight'][k,i])
f.write('}; \n')

f = open("gen_b1p.h","a")
f.write('double b_1_p_b2[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.pooling.linears.2.bias'][k])
f.write('}; \n')

f = open("gen_b1p.h","a")
f.write('double b_1_p_w3[64][64] = {')
for k in range(64):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.1.pooling.linears.3.weight'][k,i])
f.write('}; \n')

f = open("gen_b1p.h","a")
f.write('double b_1_p_b3[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.pooling.linears.3.bias'][k])
f.write('}; \n')
f.close()

#====================gen_b1f.h

f = open("gen_b1f.h","w")
f.write('double b_1_c_1_na[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.connections.1.norm.a_2'][i])
f.write('};')

f = open("gen_b1f.h","a")
f.write('double b_1_c_1_nb[64] = {')
for i in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.connections.1.norm.b_2'][i])
f.write('};')

f = open("gen_b1f.h","a")
f.write('double b_1_f_w1[256][64] = {')
for k in range(256):
    for i in range(64):
        f.write("%.8e,\n" % para['rt.forward_net.1.feed_forward.w_1.weight'][k,i])
f.write('};')

f = open("gen_b1f.h","a")
f.write('double b_1_f_b1[256] = {')
for k in range(256):
    f.write("%.8e,\n" % para['rt.forward_net.1.feed_forward.w_1.bias'][k])
f.write('};')

f = open("gen_b1f.h","a")
f.write('double b_1_f_w2[64][256] = {')
for k in range(64):
    for i in range(256):
        f.write("%.8e,\n" % para['rt.forward_net.1.feed_forward.w_2.weight'][k,i])
f.write('};')

f = open("gen_b1f.h","a")
f.write('double b_1_f_b2[64] = {')
for k in range(64):
    f.write("%.8e,\n" % para['rt.forward_net.1.feed_forward.w_2.bias'][k])
f.write('};')
f.close()

f = open("gen_out.h","w")
f.write('double lw[24][64] = {')
for k in range(24):
    for i in range(64):
        f.write("%.8e,\n" % para['linear.weight'][k,i])
f.write('};')

f = open("gen_out.h","a")
f.write('double lb[24] = {')
for k in range(24):
    f.write("%.8e,\n" % para['linear.bias'][k])
f.write('};')
f.close()
