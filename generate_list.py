import os
import shutil

splits = ['train', 'val']
for split in splits:
    dirs = os.listdir(split)
    files = []

    for d in dirs:
        # # Create symlink for annotation
        # src = os.path.abspath(os.path.join(split, d, 'proj_depth/velodyne_raw'))
        # dst = os.path.join(d[:10], d, 'proj_depth/velodyne_raw')
        # if os.path.islink(dst):
        #     os.remove(dst)
        # os.symlink(src, dst)
        # # Merger velodyne and gt
        # # dst = src.replace('/velodyne/', '/')
        # # os.rename(src, dst)
        # print('from {} -> {}'.format(src, dst))

        # File list
        fs = os.listdir(os.path.join(split, d, 'proj_depth/groundtruth/image_02'))
        fs.sort()
        for idx, f in enumerate(fs):
            files.append(d[:10]+'/'+d+' ')
            files[-1] += f.strip('.png').strip('.jpg')
            files[-1] += ' 2'
        fs = os.listdir(os.path.join(split, d, 'proj_depth/groundtruth/image_03'))
        fs.sort()
        for idx, f in enumerate(fs):
            files.append(d[:10]+'/'+d+' ')
            files[-1] += f.strip('.png').strip('.jpg')
            files[-1] += ' 3'

    # List for selected val
    # files = os.listdir('image')
    # for i in range(len(files)):
    #     folder = files[i][:10]
    #     files[i] = folder + '/' + files[i]
    #     files[i] = files[i].replace('_image_', ' ')
    #     files[i] = files[i].replace('03.png', '3')
    #     files[i] = files[i].replace('02.png', '2')

    f = open(split+ '_files.txt', 'w')
    for i in files:
        f.write(i+'\n')
    f.close()
