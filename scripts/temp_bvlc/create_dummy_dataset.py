img_path = '/home/mohit/research/cloudcv/caffe_nvidia/caffe/examples/images/cat.jpg'

fi = open('train.txt','w')
for i in range(1000):
    fi.write(img_path+' '+str(i)+'\n')

print 'Done creating dummy dataset'
